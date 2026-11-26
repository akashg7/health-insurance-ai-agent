"""Coverage verification and deterministic claim payout calculation tools."""

from datetime import date, datetime
import re
from langchain_core.tools import tool

from ._context import get_context

GENERIC_STOPWORDS = {
    "surgery", "surgeries", "treatment", "treatments", "disease", "diseases",
    "disorder", "disorders", "repair", "care", "charges", "expenses", "procedure",
    "procedures", "routine", "test", "policy", "cover", "coverage", "medical",
    "condition", "conditions", "at", "home", "in", "for", "and", "of", "the",
    "bill", "claim", "is", "are", "my", "check", "related",
}

CONDITION_ALIASES: dict[str, str] = {
    "joint replacement": "knee replacement",
    "knee surgery": "knee replacement",
    "total knee replacement": "knee replacement",
    "tkr": "knee replacement",
    "hip surgery": "hip replacement",
    "total hip replacement": "hip replacement",
    "thr": "hip replacement",
    "gall bladder": "cholecystectomy gall bladder",
    "gallbladder": "cholecystectomy gall bladder",
    "gallstones": "cholecystectomy gall bladder",
    "gallstone": "cholecystectomy gall bladder",
    "appendicitis": "appendectomy",
    "childbirth": "maternity pregnancy childbirth",
    "delivery": "maternity pregnancy childbirth",
    "pregnancy": "maternity pregnancy childbirth",
    "maternity": "maternity pregnancy childbirth",
    "c section": "maternity pregnancy childbirth",
    "cesarean": "maternity pregnancy childbirth",
    "infertility": "IVF infertility treatment",
    "ivf": "IVF infertility treatment",
    "lasik": "refractive eye surgery LASIK",
    "eye laser": "refractive eye surgery LASIK",
    "refractive": "refractive eye surgery LASIK",
    "dental": "routine dental treatment",
    "tooth": "routine dental treatment",
    "root canal": "routine dental treatment",
    "cosmetic": "cosmetic plastic surgery",
    "plastic surgery": "cosmetic plastic surgery",
    "bariatric": "obesity bariatric surgery",
    "weight loss surgery": "obesity bariatric surgery",
    "piles": "haemorrhoids piles",
    "hemorrhoids": "haemorrhoids piles",
    "fissure": "haemorrhoids piles",
    "fistula": "haemorrhoids piles",
    "kidney stone": "kidney stones",
    "lithotripsy": "kidney stones",
    "opd": "OPD outpatient consultation",
    "outpatient": "OPD outpatient consultation",
    "blood pressure": "hypertension high blood pressure",
    "bp": "hypertension high blood pressure",
    "sugar": "diabetes",
    "diabetic": "diabetes",
    "icu": "ICU intensive care charges",
    "intensive care": "ICU intensive care charges",
    "daycare": "day care procedures",
    "day care": "day care procedures",
    "room rent": "hospitalization room rent",
    "air ambulance": "emergency air ambulance",
    "road ambulance": "ambulance road",
    "ambulance": "ambulance road",
    "pcos": "PCOD polycystic ovarian disease",
    "pcod": "PCOD polycystic ovarian disease",
    "gerd": "GERD acid reflux",
    "acid reflux": "GERD acid reflux",
    "ayurveda": "AYUSH ayurveda treatment",
    "ayurvedic": "AYUSH ayurveda treatment",
    "homeopathy": "AYUSH ayurveda treatment",
    "ayush": "AYUSH ayurveda treatment",
    "hearing aid": "hearing aids spectacles",
    "spectacles": "hearing aids spectacles",
    "glasses": "hearing aids spectacles",
    "dialysis": "kidney dialysis",
    "chemotherapy": "cancer chemotherapy",
    "chemo": "cancer chemotherapy",
    "bypass": "cardiac bypass surgery",
    "angioplasty": "angioplasty stent",
    "stent": "angioplasty stent",
    "thyroid": "thyroid disorders",
    "tonsil": "tonsillectomy",
    "tonsils": "tonsillectomy",
    "fibroid": "fibroids",
    "domiciliary": "domiciliary hospitalization at home",
    "pre hospitalization": "pre hospitalization expenses",
    "post hospitalization": "post hospitalization expenses",
    "organ donor": "organ donor expenses",
}


def _clean_text(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", s.lower()).split())


def _significant_tokens(s: str) -> set[str]:
    return {
        tok
        for tok in _clean_text(s).split()
        if len(tok) >= 3 and tok not in GENERIC_STOPWORDS
    }


def _find_condition(condition: str, rules: dict | None = None) -> tuple[str | None, dict | None]:
    """Match a condition query against extracted policy rules without false generic word matches."""
    if rules is None:
        rules = get_context().rules
    if not rules:
        return None, None

    cl = _clean_text(condition)
    if not cl:
        return None, None

    # 1. Exact match (cleaned)
    for k, v in rules.items():
        if _clean_text(k) == cl:
            return k, v

    # 2. Domain alias lookup
    if cl in CONDITION_ALIASES:
        target = CONDITION_ALIASES[cl]
        if target in rules:
            return target, rules[target]
    for alias, target in CONDITION_ALIASES.items():
        if alias in cl and target in rules:
            return target, rules[target]

    # 3. Significant substring match
    sig_cl = _significant_tokens(cl)
    if sig_cl:
        for k, v in rules.items():
            kl = _clean_text(k)
            if (len(cl) >= 4 and cl in kl) or (len(kl) >= 4 and kl in cl):
                return k, v

    # 4. Specific non-stopword Jaccard overlap (prevents 'dental surgery' matching 'cataract surgery')
    if not sig_cl:
        return None, None

    best_key, best_rule, best_score = None, None, 0.0
    for k, v in rules.items():
        sig_k = _significant_tokens(k)
        if not sig_k:
            continue
        overlap = sig_cl & sig_k
        if not overlap:
            continue
        jaccard = len(overlap) / len(sig_cl | sig_k)
        if jaccard > best_score:
            best_key, best_rule, best_score = k, v, jaccard

    return (best_key, best_rule) if best_key is not None else (None, None)


def compute_claim_payout(
    rule: dict,
    user: dict,
    bill_amount: float,
    condition_label: str,
    matched_rule_name: str,
    as_of_date: date | None = None,
) -> dict:
    """Pure deterministic financial engine incorporating sub-limits, co-pays, sum insured, and waiting period."""
    if bill_amount <= 0:
        return {"error": "Bill amount must be greater than 0 INR."}

    covered_flag = rule.get("covered")
    status = rule.get("status")
    page_ref = rule.get("page_reference")
    quote = (rule.get("evidence_quote") or "")[:200]

    if covered_flag is None or status in {"not_mentioned", "extraction_failed"}:
        return {
            "condition": condition_label,
            "matched_rule_name": matched_rule_name,
            "status": status or "not_mentioned",
            "covered": None,
            "error": (
                f"Coverage for '{matched_rule_name}' could not be confirmed in the extracted rules "
                "of this document. Verify wording via search_policy_docs before estimating any payout."
            ),
        }

    if not covered_flag:
        return {
            "condition": condition_label,
            "matched_rule_name": matched_rule_name,
            "status": "excluded",
            "covered": False,
            "covered_amount": 0.0,
            "co_pay_amount": 0.0,
            "payable_to_user": 0.0,
            "eligible_today": False,
            "reason": rule.get("notes") or "Excluded under the policy.",
            "policy_reference": quote,
            "page_reference": page_ref,
        }

    sum_insured = float(user.get("sum_insured") or 500_000)
    sub_limit = rule.get("sub_limit_inr")
    has_sublimit = sub_limit is not None and float(sub_limit) > 0
    cap = float(sub_limit) if has_sublimit else sum_insured

    covered = min(float(bill_amount), cap, sum_insured)

    cond_copay = float(rule.get("co_pay_percent") or 0) / 100.0
    user_copay = float(user.get("co_pay") or 0)
    effective_copay = max(0.0, min(1.0, max(cond_copay, user_copay)))

    copay_amount = covered * effective_copay
    raw_payable = max(0.0, min(covered - copay_amount, sum_insured))

    # Check waiting-period eligibility against policyholder inception date
    wait_days = int(rule.get("waiting_period_days") or 0)
    peds = [str(p).lower().strip() for p in (user.get("pre_existing") or [])]
    cond_l = _clean_text(matched_rule_name)
    if any(p and (p in cond_l or cond_l in p) for p in peds):
        wait_days = max(wait_days, 1095)

    today = as_of_date or date.today()
    policy_start_raw = user.get("policy_start")
    elapsed_days = None
    days_remaining = 0
    waiting_status = "no_waiting_period"
    eligible_today: bool | None = True

    if wait_days > 0:
        if policy_start_raw:
            try:
                start_dt = datetime.strptime(str(policy_start_raw)[:10], "%Y-%m-%d").date()
                elapsed_days = max(0, (today - start_dt).days)
                if elapsed_days < wait_days:
                    eligible_today = False
                    days_remaining = wait_days - elapsed_days
                    waiting_status = "waiting_period_not_completed"
                else:
                    eligible_today = True
                    waiting_status = "cleared"
            except ValueError:
                eligible_today = None
                waiting_status = "verify_policy_start_date"
        else:
            eligible_today = None
            waiting_status = "verify_policy_start_date"

    payable_today = round(raw_payable, 2) if eligible_today is not False else 0.0

    return {
        "condition": condition_label,
        "matched_rule_name": matched_rule_name,
        "status": "covered",
        "covered": True,
        "bill_amount": float(bill_amount),
        "sum_insured_inr": int(sum_insured),
        "sub_limit_applied": int(sub_limit) if has_sublimit else f"None (capped by Sum Insured ₹{int(sum_insured):,})",
        "covered_amount": round(covered, 2),
        "co_pay_percent": round(effective_copay * 100, 2),
        "co_pay_amount": round(copay_amount, 2),
        "payable_to_user": payable_today,
        "estimated_payout_after_wait": round(raw_payable, 2),
        "waiting_period_days": wait_days,
        "waiting_period_status": waiting_status,
        "eligible_today": eligible_today,
        "days_since_policy_start": elapsed_days,
        "waiting_days_remaining": days_remaining,
        "policy_reference": quote,
        "page_reference": page_ref,
    }


@tool
def check_coverage(condition: str) -> dict:
    """Check if a specific medical condition or procedure is covered under the active uploaded policy.

    Use this first when the user asks 'is X covered?' or asks about waiting periods or sub-limits.
    Returns matched_rule_name, status ('covered', 'excluded', 'not_mentioned'), sub_limit_inr,
    co_pay_percent, waiting_period_days, evidence_quote, and page_reference.
    If status is 'not_mentioned' or covered is null, MUST follow up with search_policy_docs.
    """
    ctx = get_context()
    if not ctx.rules:
        return {
            "error": "No verified health insurance policy rules are loaded in this session. Upload a valid policy PDF first."
        }

    matched_key, rule = _find_condition(condition, ctx.rules)
    if rule is None or matched_key is None:
        return {
            "status": "not_mentioned",
            "covered": None,
            "condition": condition,
            "policy_document": ctx.policy_label,
            "notes": (
                f"No pre-extracted rule matched '{condition}' in '{ctx.policy_label}'. "
                "Call search_policy_docs to check full policy wording before concluding."
            ),
        }

    status = rule.get("status")
    if status in {"not_mentioned", "extraction_failed"} or rule.get("covered") is None:
        return {
            "condition": condition,
            "matched_rule_name": matched_key,
            "status": status or "not_mentioned",
            "covered": None,
            "policy_document": ctx.policy_label,
            "notes": (
                f"'{matched_key}' was not verified in the key extracted excerpts of '{ctx.policy_label}'. "
                "MUST call search_policy_docs to inspect the document text; do NOT assume it is covered."
            ),
        }

    return {
        "condition": condition,
        "matched_rule_name": matched_key,
        "policy_document": ctx.policy_label,
        **rule,
    }


@tool
def calculate_claim(condition: str, bill_amount: float, user_id: str = "") -> dict:
    """Deterministically calculate the expected claim payout for a medical condition and bill amount.

    Combines the active policy's coverage rule (sub-limit, co-pay, waiting period) with the
    active policyholder profile (sum insured, user co-pay, policy inception date).
    """
    from .user import _get_user_data

    ctx = get_context()
    if not ctx.rules:
        return {"error": "No verified policy loaded in this session."}

    matched_key, rule = _find_condition(condition, ctx.rules)
    if rule is None or matched_key is None:
        return {
            "error": (
                f"No rule found for '{condition}' in the active policy. "
                "Check wording with search_policy_docs first."
            ),
            "status": "not_found",
        }

    uid = user_id.strip() or ctx.user_id
    user = _get_user_data(uid) or {}
    return compute_claim_payout(rule, user, float(bill_amount), condition, matched_key)
