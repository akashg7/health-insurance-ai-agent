"""Document gatekeeper validating that an uploaded PDF is a health insurance policy document."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from ingest import extract_preview_text, parse_pdf_to_pages

load_dotenv()

HEALTH_INSURANCE_KEYWORDS = {
    "health insurance",
    "sum insured",
    "waiting period",
    "hospitalization",
    "hospitalisation",
    "pre-existing",
    "pre existing",
    "co-pay",
    "copay",
    "co-payment",
    "cashless",
    "irdai",
    "policyholder",
    "insured person",
    "exclusions",
    "deductible",
    "day care",
    "daycare",
    "sub-limit",
    "sublimit",
    "room rent",
    "domiciliary",
    "reimbursement",
    "inpatient",
    "icu",
    "policy wording",
    "mediclaim",
    "indemnity",
    "uin",
}

CORE_POLICY_TERMS = {
    "sum insured",
    "waiting period",
    "pre-existing",
    "pre existing",
    "irdai",
    "hospitalization",
    "hospitalisation",
    "policy wording",
    "cashless",
    "mediclaim",
}

VALIDATOR_MODEL_PREFERENCE = [
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
]


class NonPolicyDocumentError(ValueError):
    """Raised when an uploaded PDF does not appear to be a health insurance document."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


def get_groq_api_key() -> str | None:
    """Resolve GROQ_API_KEY from environment, .env files, or Streamlit secrets."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key.strip()

    for candidate in (Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                if line.startswith("GROQ_API_KEY="):
                    val = line.split("=", 1)[1].strip("\"' ")
                    if val:
                        return val

    try:
        import streamlit as st
        secret = st.secrets.get("GROQ_API_KEY")
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return None


def resolve_groq_model(client: Groq, env_var: str, preferred: list[str]) -> str:
    """Pick a configured or available Groq chat model without failing on deprecated model IDs."""
    override = os.environ.get(env_var)
    try:
        available = {m.id for m in client.models.list().data}
    except Exception:
        return override or preferred[0]

    if override and override in available:
        return override
    for candidate in preferred:
        if candidate in available:
            return candidate
    for model_id in sorted(available):
        if not any(x in model_id for x in ("whisper", "orpheus", "guard", "allam")):
            return model_id
    return preferred[0]


def _classify_with_llm(preview_text: str, filename: str) -> dict | None:
    api_key = get_groq_api_key()
    if not api_key:
        return None

    client = Groq(api_key=api_key, timeout=25.0, max_retries=0)
    model = resolve_groq_model(client, "GROQ_VALIDATOR_MODEL", VALIDATOR_MODEL_PREFERENCE)

    prompt = f"""Analyze the following PDF excerpts from '{filename}' and determine whether this document is a Health Insurance Policy document (policy wording, policy schedule/certificate, health insurance product brochure, or health rider/add-on document).

REJECT (set is_health_insurance_document: false) if it is:
- A resume, CV, invoice, receipt, medical report, lab test, book, tutorial, presentation, or article.
- Motor/car insurance, life/term insurance (without medical/hospitalization terms), home insurance, or unrelated financial document.

EXCERPTS:
---
{preview_text}
---

Respond with valid JSON ONLY:
{{
  "is_health_insurance_document": true or false,
  "document_type": "policy_wording" | "policy_schedule" | "product_brochure" | "rider_addon" | "non_health_insurance" | "unrelated_document",
  "insurer_or_issuer": "<company name or null>",
  "plan_name": "<plan/product name or null>",
  "reason": "<one concise factual sentence>"
}}"""

    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You classify documents strictly using valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=300,
                **extra,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "rate_limit" in msg:
                time.sleep(2.5 * (attempt + 1))
                continue
            return None
    return None


def validate_policy_pdf(pdf_path: Path, pages: list[dict] | None = None) -> tuple[dict, list[dict]]:
    """Validate readability and health-insurance domain relevance. Returns (validation_info, pages)."""
    if pages is None:
        pages, total_pages = parse_pdf_to_pages(pdf_path)
    else:
        total_pages = len(pages)

    full_text_lower = "\n".join(p["text"] for p in pages).lower()
    total_chars = sum(len(p["text"]) for p in pages)

    matched_keywords = sorted(k for k in HEALTH_INSURANCE_KEYWORDS if k in full_text_lower)
    matched_core = sorted(k for k in CORE_POLICY_TERMS if k in full_text_lower)

    # Layer 1: fast local domain check (0 API calls for obviously unrelated PDFs)
    if len(matched_keywords) < 3:
        raise NonPolicyDocumentError(
            f"'{pdf_path.name}' does not look like a health insurance policy document "
            f"(only {len(matched_keywords)} health-insurance terms detected). "
            "Please upload an actual health insurance policy wording, schedule, or brochure.",
            details={
                "is_health_insurance_document": False,
                "document_type": "unrelated_document",
                "matched_keywords": matched_keywords,
                "reason": "Insufficient health insurance terminology across document text.",
            },
        )

    # Layer 2: semantic verification via Groq LLM
    preview = extract_preview_text(pages, max_chars=3000)
    llm_result = _classify_with_llm(preview, pdf_path.name)

    if llm_result is not None:
        if not llm_result.get("is_health_insurance_document", False):
            doc_type = llm_result.get("document_type", "unrelated_document")
            reason = llm_result.get("reason", "Document is not a health insurance policy.")
            raise NonPolicyDocumentError(
                f"Rejected '{pdf_path.name}' ({doc_type.replace('_', ' ')}): {reason}",
                details={**llm_result, "matched_keywords": matched_keywords},
            )
        return {
            "is_health_insurance_document": True,
            "document_type": llm_result.get("document_type", "policy_wording"),
            "insurer_or_issuer": llm_result.get("insurer_or_issuer"),
            "plan_name": llm_result.get("plan_name"),
            "reason": llm_result.get("reason", "Verified as health insurance document."),
            "matched_keywords_count": len(matched_keywords),
            "num_pages": total_pages,
            "total_chars": total_chars,
        }, pages

    # Conservative fallback if LLM validator is temporarily unreachable
    if len(matched_keywords) >= 6 and len(matched_core) >= 2:
        return {
            "is_health_insurance_document": True,
            "document_type": "policy_wording",
            "insurer_or_issuer": None,
            "plan_name": None,
            "reason": "Verified via domain keyword heuristics (LLM validator offline).",
            "matched_keywords_count": len(matched_keywords),
            "num_pages": total_pages,
            "total_chars": total_chars,
        }, pages

    raise NonPolicyDocumentError(
        f"Could not verify '{pdf_path.name}' as a health insurance policy document.",
        details={
            "is_health_insurance_document": False,
            "document_type": "unverified_document",
            "matched_keywords": matched_keywords,
            "reason": "Failed semantic policy verification and did not meet strict fallback criteria.",
        },
    )
