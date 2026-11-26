"""Per-policy rule extractor with hybrid RAG retrieval, strict grounding, and TPM-safe batching.

Extracts structured rules for canonical conditions from data/conditions_to_check.json and saves
verified v2 rules into policies/<pdf_hash>/rules.json. Missing or unmentioned conditions are marked
explicitly as `not_mentioned` (covered=None) rather than fabricated as covered.
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from groq import Groq

from ingest import hybrid_search, pdf_fingerprint, search
from policy_validator import get_groq_api_key, resolve_groq_model

POLICIES_DIR = Path(__file__).parent / "policies"
CONDITIONS_FILE = Path(__file__).parent / "data" / "conditions_to_check.json"

EXTRACTION_MODEL_PREFERENCE = [
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

BATCH_SIZE = 6
MIN_RAG_SIMILARITY = 0.22
SCHEMA_VERSION = 2
TPM_BUDGET = 6200  # Stay safely under Groq free-tier 8,000 TPM cap

# Medical synonyms & policy terminology aliases for hybrid lexical search
CONDITION_SYNONYMS: dict[str, list[str]] = {
    "cataract surgery": ["cataract"],
    "knee replacement": ["knee replacement", "joint replacement", "arthroplasty"],
    "hip replacement": ["hip replacement", "joint replacement", "arthroplasty"],
    "hernia repair": ["hernia"],
    "cholecystectomy gall bladder": ["cholecystectomy", "gall bladder", "gallbladder", "cholelithiasis"],
    "appendectomy": ["appendectomy", "appendicitis"],
    "hysterectomy": ["hysterectomy", "uterus"],
    "tonsillectomy": ["tonsillectomy", "tonsillitis", "adenoidectomy"],
    "cardiac bypass surgery": ["bypass", "coronary", "cabg", "heart disease", "cardiac"],
    "angioplasty stent": ["angioplasty", "stent", "coronary", "ptca"],
    "kidney dialysis": ["dialysis", "renal failure"],
    "cancer chemotherapy": ["chemotherapy", "cancer", "malignancy", "oncology", "radiotherapy"],
    "stroke treatment": ["stroke", "cerebrovascular"],
    "brain tumor surgery": ["brain tumor", "benign tumor", "intracranial"],
    "diabetes": ["diabetes", "pre-existing"],
    "hypertension high blood pressure": ["hypertension", "blood pressure", "pre-existing"],
    "thyroid disorders": ["thyroid", "pre-existing"],
    "asthma": ["asthma", "bronchial", "pre-existing"],
    "kidney stones": ["kidney stone", "calculus", "urogenital", "lithotripsy"],
    "PCOD polycystic ovarian disease": ["polycystic", "pcod", "pcos"],
    "fibroids": ["fibroids", "fibromyoma", "myomectomy"],
    "osteoarthritis": ["osteoarthritis", "arthritis", "osteoporosis"],
    "GERD acid reflux": ["gerd", "reflux", "ulcer", "duodenum"],
    "sinusitis": ["sinusitis", "rhinitis", "sinus"],
    "varicose veins": ["varicose"],
    "haemorrhoids piles": ["haemorrhoids", "hemorrhoids", "piles", "fissure", "fistula"],
    "maternity pregnancy childbirth": ["maternity", "childbirth", "pregnancy", "caesarean"],
    "IVF infertility treatment": ["infertility", "sterility", "ivf", "assisted reproduction"],
    "cosmetic plastic surgery": ["cosmetic", "plastic surgery", "aesthetic", "change of appearance"],
    "routine dental treatment": ["dental", "tooth", "teeth"],
    "obesity bariatric surgery": ["obesity", "bariatric", "weight control", "bmi"],
    "refractive eye surgery LASIK": ["refractive", "lasik", "dioptre", "eyesight"],
    "hearing aids spectacles": ["hearing aid", "spectacles", "contact lens"],
    "vaccination immunization": ["vaccination", "immunization", "inoculation"],
    "OPD outpatient consultation": ["outpatient", "opd"],
    "hazardous adventure sports": ["hazardous", "adventure sport"],
    "alcohol substance abuse": ["alcohol", "substance abuse", "drug abuse", "addictive"],
    "self-inflicted injury suicide": ["self-injury", "self-inflicted", "suicide"],
    "hospitalization room rent": ["room rent", "single private room", "room category"],
    "ICU intensive care charges": ["intensive care", "icu", "iccu"],
    "day care procedures": ["day care", "daycare"],
    "ambulance road": ["road ambulance", "ambulance"],
    "emergency air ambulance": ["air ambulance"],
    "AYUSH ayurveda treatment": ["ayush", "ayurveda", "homeopathy", "unani", "siddha"],
    "pre hospitalization expenses": ["pre-hospitalization", "pre hospitalization"],
    "post hospitalization expenses": ["post-hospitalization", "post hospitalization"],
    "organ donor expenses": ["organ donor"],
    "domiciliary hospitalization at home": ["domiciliary"],
}

BATCH_EXTRACTION_PROMPT = """You extract STRICTLY GROUNDED structured coverage rules from health insurance policy excerpts.

CONDITIONS TO EVALUATE:
{conditions_list}

POLICY EXCERPTS (with page numbers):
---
{excerpts}
---

For EACH condition listed above, return a JSON object keyed by the exact condition string.

CRITICAL GROUNDING RULES — READ CAREFULLY:
1. ONLY use facts explicitly present in the excerpts above. NEVER use outside knowledge or assumptions.
2. Note on Joint Replacement / Cataract / Specified Lists: If an excerpt explicitly covers/lists "Joint replacement surgeries" under a waiting period, apply it to knee/hip replacement; if a list says "Listed conditions are excluded until expiry of 24 months", mark status="covered", covered=true, waiting_period_days=730.
3. If a condition (or its medical category) is NOT mentioned anywhere in the provided excerpts, you MUST set:
   "status": "not_mentioned", "covered": null, "sub_limit_inr": null, "co_pay_percent": null, "waiting_period_days": null, "evidence_quote": "", "page_reference": null, "notes": "Not mentioned in retrieved policy excerpts."
4. PERMANENT EXCLUSIONS under "Standard Exclusions" or "Specific Exclusions" with NO expiry months ===> "status": "excluded", "covered": false, "waiting_period_days": null.
5. Waiting period codes:
   - Initial 30-day wait -> 30
   - Specified disease (commonly 24 months) -> 730
   - Pre-Existing Disease / PED (commonly 36 months) -> 1095
   - Covered from Day 1 / accident / no wait explicitly stated -> 0
   - Wait not stated in excerpts -> null
6. evidence_quote MUST be a verbatim short excerpt (max 180 chars) copied from the excerpts, and page_reference MUST be the [Page N] number where it appears.

Output STRICT valid JSON matching:
{{
  "rules": {{
    "<exact condition name>": {{
      "status": "covered" | "excluded" | "not_mentioned",
      "covered": true | false | null,
      "sub_limit_inr": <integer or null>,
      "co_pay_percent": <number 0-100 or null>,
      "waiting_period_days": <integer or null>,
      "evidence_quote": "<verbatim quote or empty string>",
      "page_reference": <integer page number or null>,
      "notes": "<one concise grounded sentence>"
    }}
  }}
}}"""


def get_policy_dir(pdf_path: Path) -> Path:
    d = POLICIES_DIR / pdf_fingerprint(pdf_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_retry_delay(err_msg: str, attempt: int) -> float:
    match = re.search(r"try again in\s*(?:(\d+)m)?([\d.]+)s", err_msg, re.IGNORECASE)
    if match:
        mins = int(match.group(1) or 0)
        secs = float(match.group(2) or 2.0)
        return min(35.0, mins * 60 + secs + 0.5)
    return min(25.0, 2.5 * (attempt + 1))


def _is_cached_rules_valid(rules_path: Path) -> bool:
    if not rules_path.exists():
        return False
    try:
        data = json.loads(rules_path.read_text())
    except Exception:
        return False

    meta = data.get("_meta", {})
    if meta.get("schema_version") != SCHEMA_VERSION:
        return False
    if meta.get("failed_count", 0) > 0:
        return False

    rules = data.get("rules", {})
    if not rules:
        return False
    for rule in rules.values():
        notes = (rule.get("notes") or "").lower()
        if rule.get("status") == "extraction_failed" or "extraction error" in notes:
            return False
    return True


def _keywords_for_condition(cond: str) -> list[str]:
    kws = list(CONDITION_SYNONYMS.get(cond, []))
    for tok in cond.lower().split():
        if len(tok) >= 4 and tok not in {"surgery", "repair", "treatment", "disease", "disorders", "expenses", "routine", "charges"}:
            kws.append(tok)
    return list(dict.fromkeys(kws))


def _collect_batch_excerpts(pdf_path: Path, category: str, conditions: list[str]) -> tuple[str, set[int]]:
    """Retrieve deduplicated hybrid chunks + relevant waiting-period header anchors."""
    unique_chunks: dict[tuple[int, str], dict] = {}

    def _add_hits(hits: list[dict]):
        for h in hits:
            key = (int(h["page"]), h["text"])
            existing = unique_chunks.get(key)
            if existing is None or (h.get("keyword_hits", 0), h["similarity"]) > (
                existing.get("keyword_hits", 0),
                existing["similarity"],
            ):
                unique_chunks[key] = h

    for cond in conditions:
        kws = _keywords_for_condition(cond)
        hits = hybrid_search(pdf_path, cond, keywords=kws, k=3, min_similarity=MIN_RAG_SIMILARITY)
        _add_hits(hits)

    # For procedure/condition batches, include policy waiting-period anchor clauses if present
    # (e.g., when Page 29 has the disease table while Page 28 defines the 24m/36m duration)
    if category in {"common_procedures", "common_conditions"} and unique_chunks:
        anchor_hits = hybrid_search(
            pdf_path,
            "Specified Disease Procedure waiting period months Pre-Existing Diseases",
            keywords=["specified disease/procedure", "pre-existing diseases", "expiry of 24 months", "expiry of 36 months"],
            k=2,
            min_similarity=0.35,
        )
        _add_hits(anchor_hits)

    if not unique_chunks:
        return "[no relevant excerpts found in document]", set()

    ordered = sorted(
        unique_chunks.values(),
        key=lambda x: (-x.get("keyword_hits", 0), -x["similarity"], x["page"]),
    )[:10]
    ordered_by_page = sorted(ordered, key=lambda x: x["page"])
    kept_parts: list[str] = []
    pages_present: set[int] = set()
    total_len = 0
    for c in ordered_by_page:
        part = f"[Page {c['page']}]: {c['text']}"
        if kept_parts and total_len + len(part) + 2 > 4200:
            continue
        kept_parts.append(part)
        pages_present.add(int(c["page"]))
        total_len += len(part) + 2
    return "\n\n".join(kept_parts), pages_present


def _normalize_extracted_rule(condition: str, raw: dict | None, valid_pages: set[int], category: str) -> dict:
    """Enforce schema invariants, grounding check, and waiting-period semantic normalization."""
    if not isinstance(raw, dict):
        return {
            "status": "not_mentioned",
            "covered": None,
            "sub_limit_inr": None,
            "co_pay_percent": None,
            "waiting_period_days": None,
            "evidence_quote": "",
            "page_reference": None,
            "notes": "Not mentioned in retrieved policy excerpts.",
            "_category": category,
        }

    status = str(raw.get("status") or "").lower().strip()
    covered = raw.get("covered")
    wait = raw.get("waiting_period_days")
    sub_limit = raw.get("sub_limit_inr")
    copay = raw.get("co_pay_percent")
    quote = (raw.get("evidence_quote") or "").strip()[:200]
    page_ref = raw.get("page_reference")
    notes = (raw.get("notes") or "").strip()

    try:
        page_ref = int(page_ref) if page_ref is not None else None
    except (TypeError, ValueError):
        page_ref = None

    try:
        wait = int(wait) if wait is not None else None
    except (TypeError, ValueError):
        wait = None

    try:
        sub_limit = int(float(sub_limit)) if sub_limit is not None else None
    except (TypeError, ValueError):
        sub_limit = None

    try:
        copay = float(copay) if copay is not None else None
    except (TypeError, ValueError):
        copay = None

    if status not in {"covered", "excluded", "not_mentioned"}:
        if covered is True:
            status = "covered"
        elif covered is False:
            status = "excluded"
        else:
            status = "not_mentioned"

    # Fix waiting-period misinterpretation ("excluded until 24 months" => covered after waiting period)
    if status == "excluded" and wait and wait > 0:
        notes_l = notes.lower()
        permanent_markers = ("permanently excluded", "permanent exclusion", "never covered", "not covered under any")
        if not any(pm in notes_l for pm in permanent_markers):
            status = "covered"
            covered = True
            months = max(1, wait // 30)
            notes = f"Covered after {months}-month waiting period. {notes}".strip()

    # Grounding verification: positive/negative claims MUST cite a retrieved page and quote
    if status in {"covered", "excluded"}:
        if not valid_pages or not page_ref or page_ref not in valid_pages or len(quote) < 6:
            status = "not_mentioned"
            covered = None
            sub_limit = None
            copay = None
            wait = None
            quote = ""
            page_ref = None
            notes = "Not explicitly verified with page evidence in retrieved excerpts."
        else:
            covered = (status == "covered")
            if copay is None:
                copay = 0.0
            if wait is None and status == "covered":
                wait = 0
    else:
        covered = None
        sub_limit = None
        copay = None
        wait = None
        quote = ""
        page_ref = None
        if not notes:
            notes = "Not mentioned in retrieved policy excerpts."

    return {
        "status": status,
        "covered": covered,
        "sub_limit_inr": sub_limit,
        "co_pay_percent": copay,
        "waiting_period_days": wait,
        "evidence_quote": quote,
        "page_reference": page_ref,
        "notes": notes,
        "_category": category,
    }


def _extract_batch(
    client: Groq,
    model: str,
    pdf_path: Path,
    category: str,
    batch: list[tuple[str, str]],
    token_window: list[tuple[float, int]],
) -> dict[str, dict]:
    conditions = [cond for _, cond in batch]
    excerpts, valid_pages = _collect_batch_excerpts(pdf_path, category, conditions)

    if not valid_pages:
        return {
            cond: _normalize_extracted_rule(cond, None, valid_pages, cat)
            for cat, cond in batch
        }

    cond_list_str = "\n".join(f"- {cond}" for cond in conditions)
    prompt = BATCH_EXTRACTION_PROMPT.format(conditions_list=cond_list_str, excerpts=excerpts)
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    est_tokens = max(800, len(prompt) // 4 + 800)

    last_err = None
    for attempt in range(5):
        now = time.monotonic()
        token_window[:] = [(ts, tok) for ts, tok in token_window if now - ts < 62.0]
        used_recent = sum(tok for _, tok in token_window)
        if token_window and used_recent + est_tokens > TPM_BUDGET:
            sleep_for = max(1.0, (token_window[0][0] + 62.0) - now)
            time.sleep(sleep_for)
            now = time.monotonic()
            token_window[:] = [(ts, tok) for ts, tok in token_window if now - ts < 62.0]

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Output only valid JSON matching the requested schema."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=1800,
                **extra,
            )
            used = getattr(getattr(resp, "usage", None), "total_tokens", None) or est_tokens
            token_window.append((time.monotonic(), int(used)))

            payload = json.loads(resp.choices[0].message.content)
            parsed_rules = payload.get("rules", payload)
            if not isinstance(parsed_rules, dict):
                raise ValueError("LLM response did not contain a JSON object for rules.")

            lower_map = {str(k).lower().strip(): v for k, v in parsed_rules.items()}
            out: dict[str, dict] = {}
            for cat, cond in batch:
                raw = parsed_rules.get(cond) or lower_map.get(cond.lower().strip())
                out[cond] = _normalize_extracted_rule(cond, raw, valid_pages, cat)
            return out
        except Exception as exc:
            last_err = exc
            delay = _parse_retry_delay(str(exc), attempt)
            time.sleep(delay)

    return {
        cond: {
            "status": "extraction_failed",
            "covered": None,
            "sub_limit_inr": None,
            "co_pay_percent": None,
            "waiting_period_days": None,
            "evidence_quote": "",
            "page_reference": None,
            "notes": f"Extraction failed after retries: {last_err}",
            "_category": cat,
        }
        for cat, cond in batch
    }


def extract_all(
    pdf_path: Path,
    force: bool = False,
    validation_info: dict | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Extract structured rules per category in batches under Groq rate-limit budgets."""
    pol_dir = get_policy_dir(pdf_path)
    rules_path = pol_dir / "rules.json"

    if not force and _is_cached_rules_valid(rules_path):
        if progress_callback:
            progress_callback(1, 1, "Loaded verified cached policy rules")
        return rules_path

    meta = json.loads(CONDITIONS_FILE.read_text())
    batches: list[tuple[str, list[tuple[str, str]]]] = []
    for category, items in meta.items():
        if category == "_meta":
            continue
        cat_items = [(category, cond) for cond in items]
        for i in range(0, len(cat_items), BATCH_SIZE):
            batches.append((category, cat_items[i : i + BATCH_SIZE]))

    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment / .env / Streamlit secrets")
    client = Groq(api_key=api_key, timeout=30.0, max_retries=0)
    model = resolve_groq_model(client, "GROQ_EXTRACTION_MODEL", EXTRACTION_MODEL_PREFERENCE)

    rules: dict[str, dict] = {}
    token_window: list[tuple[float, int]] = []
    for idx, (category, batch) in enumerate(batches, start=1):
        if progress_callback:
            progress_callback(idx - 1, len(batches), f"Analyzing policy clauses (batch {idx}/{len(batches)})...")
        batch_rules = _extract_batch(client, model, pdf_path, category, batch, token_window)
        rules.update(batch_rules)

    if progress_callback:
        progress_callback(len(batches), len(batches), "Finalizing structured policy index...")

    covered_count = sum(1 for r in rules.values() if r.get("status") == "covered")
    excluded_count = sum(1 for r in rules.values() if r.get("status") == "excluded")
    not_mentioned_count = sum(1 for r in rules.values() if r.get("status") == "not_mentioned")
    failed_count = sum(1 for r in rules.values() if r.get("status") == "extraction_failed")

    output = {
        "_meta": {
            "schema_version": SCHEMA_VERSION,
            "pdf_hash": pdf_fingerprint(pdf_path),
            "pdf_filename": pdf_path.name,
            "extracted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "extraction_model": model,
            "num_conditions": len(rules),
            "covered_count": covered_count,
            "excluded_count": excluded_count,
            "not_mentioned_count": not_mentioned_count,
            "failed_count": failed_count,
            "validation": validation_info or {
                "is_health_insurance_document": True,
                "document_type": "policy_wording",
            },
        },
        "rules": rules,
    }
    rules_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    return rules_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python rules_extractor.py <path-to-pdf> [--force]")
        sys.exit(1)
    pdf = Path(sys.argv[1])
    if not pdf.exists():
        print(f"error: not found: {pdf}")
        sys.exit(1)
    force_flag = "--force" in sys.argv
    out_path = extract_all(pdf, force=force_flag)
    saved = json.loads(out_path.read_text())
    print("Saved:", out_path)
    print("Meta:", json.dumps(saved["_meta"], indent=2))

