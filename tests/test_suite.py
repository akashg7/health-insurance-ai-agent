"""Comprehensive pytest suite verifying PDF validation, retrieval, matching, claims, and isolation."""

from datetime import date
from pathlib import Path
import json
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
BOT_DIR = ROOT_DIR / "health_insurance_bot"
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from ingest import EmptyPDFError, ingest, search  # noqa: E402
from policy_validator import NonPolicyDocumentError, validate_policy_pdf  # noqa: E402
from rules_extractor import _is_cached_rules_valid, _normalize_extracted_rule  # noqa: E402
from tools._context import PolicyContext, bind_context, get_context  # noqa: E402
from tools.coverage import _find_condition, compute_claim_payout  # noqa: E402
from tools.documents import list_required_documents  # noqa: E402
from tools.hospitals import find_network_hospitals  # noqa: E402
from tools.policy_search import search_policy_docs  # noqa: E402


def _make_minimal_pdf(path: Path, text: str) -> Path:
    esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 11 Tf 50 740 Td ({esc}) Tj ET"
    objs = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj",
        f"4 0 obj << /Length {len(stream)} >> stream\n{stream}\nendstream endobj",
        "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
    ]
    body = "%PDF-1.4\n"
    offsets = []
    for o in objs:
        offsets.append(len(body))
        body += o + "\n"
    xref_pos = len(body)
    body += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n"
    for off in offsets:
        body += f"{off:010d} 00000 n \n"
    body += f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF"
    path.write_bytes(body.encode("latin-1"))
    return path


def test_empty_pdf_rejected(tmp_path: Path):
    pdf = _make_minimal_pdf(tmp_path / "scanned.pdf", "")
    with pytest.raises(EmptyPDFError):
        validate_policy_pdf(pdf)


def test_random_resume_pdf_rejected(tmp_path: Path):
    pdf = _make_minimal_pdf(
        tmp_path / "resume.pdf",
        (
            "Jane Doe Senior Backend Software Engineer Experience Built scalable microservices "
            "using Python FastAPI Kubernetes PostgreSQL Redis AWS Docker React TypeScript "
            "Led distributed systems architecture and CI/CD pipelines across cloud infrastructure "
            "Education Bachelor of Technology Computer Science Skills Python SQL Docker AWS"
        ),
    )
    with pytest.raises(NonPolicyDocumentError):
        validate_policy_pdf(pdf)


def test_rule_normalization_prevents_unverified_covered_hallucination():
    # LLM falsely tries to return covered=True with no page or empty quote
    raw = {
        "status": "covered",
        "covered": True,
        "evidence_quote": "",
        "page_reference": 0,
        "notes": "Assumed covered",
    }
    norm = _normalize_extracted_rule("knee replacement", raw, valid_pages={28, 29}, category="common_procedures")
    assert norm["status"] == "not_mentioned"
    assert norm["covered"] is None
    assert norm["page_reference"] is None


def test_rule_normalization_flips_waiting_period_misread_not_permanent_exclusion():
    raw = {
        "status": "excluded",
        "covered": False,
        "waiting_period_days": 730,
        "evidence_quote": "Excluded until expiry of 24 months continuous coverage.",
        "page_reference": 28,
        "notes": "Excluded until expiry of 24 months of continuous coverage.",
    }
    norm = _normalize_extracted_rule("cataract surgery", raw, valid_pages={28, 29}, category="common_procedures")
    assert norm["status"] == "covered"
    assert norm["covered"] is True
    assert norm["waiting_period_days"] == 730


def test_poisoned_cache_detection(tmp_path: Path):
    bad_v1 = tmp_path / "bad_v1.json"
    bad_v1.write_text(json.dumps({"_meta": {"num_conditions": 1}, "rules": {"knee": {"covered": True, "notes": "extraction error: 429"}}}))
    assert _is_cached_rules_valid(bad_v1) is False

    bad_v2 = tmp_path / "bad_v2.json"
    bad_v2.write_text(json.dumps({"_meta": {"schema_version": 2, "failed_count": 1}, "rules": {"knee": {"status": "extraction_failed"}}}))
    assert _is_cached_rules_valid(bad_v2) is False

    good_v2 = tmp_path / "good_v2.json"
    good_v2.write_text(json.dumps({"_meta": {"schema_version": 2, "failed_count": 0}, "rules": {"knee": {"status": "not_mentioned", "covered": None, "notes": "Not mentioned"}}}))
    assert _is_cached_rules_valid(good_v2) is True


def test_fuzzy_condition_matcher_prevents_cross_category_false_positives():
    sample_rules = {
        "cataract surgery": {"status": "covered", "covered": True},
        "knee replacement": {"status": "covered", "covered": True},
        "routine dental treatment": {"status": "excluded", "covered": False},
        "brain tumor surgery": {"status": "not_mentioned", "covered": None},
    }
    key_dental, _ = _find_condition("dental surgery", sample_rules)
    assert key_dental == "routine dental treatment"

    key_joint, _ = _find_condition("joint replacement surgery", sample_rules)
    assert key_joint == "knee replacement"

    key_unrelated, _ = _find_condition("hair transplant cosmetic", sample_rules)
    assert key_unrelated != "cataract surgery"
    assert key_unrelated != "brain tumor surgery"


def test_deterministic_claim_calculator_scenarios():
    cataract_rule = {
        "status": "covered",
        "covered": True,
        "sub_limit_inr": 40_000,
        "co_pay_percent": 0,
        "waiting_period_days": 730,
        "evidence_quote": "Cataract sub-limit 40000",
        "page_reference": 29,
    }
    senior_user = {"sum_insured": 500_000, "co_pay": 0.10, "policy_start": "2023-01-01", "pre_existing": []}

    # Cleared waiting period -> 40k cap * (1 - 0.10) = 36,000
    res = compute_claim_payout(cataract_rule, senior_user, 85_000, "cataract surgery", "cataract surgery", as_of_date=date(2026, 11, 26))
    assert res["covered"] is True
    assert res["eligible_today"] is True
    assert res["covered_amount"] == 40_000.0
    assert res["co_pay_amount"] == 4_000.0
    assert res["payable_to_user"] == 36_000.0

    # Active waiting period -> payable_to_user = 0, estimated_payout_after_wait = 36,000
    new_user = {"sum_insured": 500_000, "co_pay": 0.10, "policy_start": "2026-06-01", "pre_existing": []}
    res_wait = compute_claim_payout(cataract_rule, new_user, 85_000, "cataract surgery", "cataract surgery", as_of_date=date(2026, 11, 26))
    assert res_wait["eligible_today"] is False
    assert res_wait["payable_to_user"] == 0.0
    assert res_wait["estimated_payout_after_wait"] == 36_000.0
    assert res_wait["waiting_days_remaining"] > 500

    # Sum Insured cap exceeded
    knee_rule = {"status": "covered", "covered": True, "sub_limit_inr": None, "co_pay_percent": 0, "waiting_period_days": 0, "page_reference": 29}
    low_si_user = {"sum_insured": 300_000, "co_pay": 0.20, "policy_start": "2020-01-01", "pre_existing": []}
    res_si = compute_claim_payout(knee_rule, low_si_user, 450_000, "knee replacement", "knee replacement", as_of_date=date(2026, 11, 26))
    assert res_si["covered_amount"] == 300_000.0
    assert res_si["payable_to_user"] == 240_000.0

    # Unverified rule refuses calculation
    unverified_rule = {"status": "not_mentioned", "covered": None}
    res_unv = compute_claim_payout(unverified_rule, senior_user, 50_000, "stem cell", "stem cell")
    assert "error" in res_unv


def test_context_isolation_across_sessions():
    ctx_a = PolicyContext(pdf_hash="hash_a", user_id="U1001")
    ctx_b = PolicyContext(pdf_hash="hash_b", user_id="U1002", custom_user={"user_id": "self", "name": "Custom"})

    with bind_context(ctx_a):
        assert get_context().pdf_hash == "hash_a"
        assert get_context().user_id == "U1001"
        with bind_context(ctx_b):
            assert get_context().pdf_hash == "hash_b"
            assert get_context().custom_user["name"] == "Custom"
        assert get_context().pdf_hash == "hash_a"


def test_vector_search_clamping_and_similarity_threshold():
    sample_pdf = BOT_DIR / "data" / "policy.pdf"
    if not sample_pdf.exists():
        pytest.skip("Sample policy.pdf missing")

    ingest(sample_pdf)
    ctx = PolicyContext(pdf_path=sample_pdf)
    with bind_context(ctx):
        hits = search_policy_docs.invoke({"query": "cataract waiting period", "k": 5})
        assert hits and hits[0].get("page", 0) > 0
        assert 0.0 <= hits[0]["similarity_score"] <= 1.0

        # Extremely unrelated gibberish with high threshold returns empty from low-level search
        gib = search(sample_pdf, "xyzzy plugh qwerty orbital angular momentum banana", k=3, min_similarity=0.65)
        assert gib == []


def test_sample_directory_and_checklists_tools():
    hosp = find_network_hospitals.invoke({"city": "Bangalore", "specialty": "ophthalmology"})
    assert hosp["count"] >= 1
    assert "sample" in hosp["data_source"].lower()

    docs = list_required_documents.invoke({"claim_type": "reimbursement"})
    assert docs["claim_type"] == "reimbursement_hospitalization"
    assert len(docs["documents"]) >= 5
