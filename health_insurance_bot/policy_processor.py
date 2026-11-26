"""Single entry point to validate, ingest, and extract structured coverage rules from a PDF."""

import json
import sys
from pathlib import Path
from typing import Callable

from ingest import get_or_create_collection, ingest, pdf_fingerprint
from policy_validator import validate_policy_pdf
from rules_extractor import _is_cached_rules_valid, extract_all, get_policy_dir


def process(
    pdf_path: Path,
    force: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Validate PDF readability & health-insurance domain, index in ChromaDB, and extract rules."""
    pdf_hash = pdf_fingerprint(pdf_path)
    pol_dir = get_policy_dir(pdf_path)
    rules_path = pol_dir / "rules.json"

    # Fast path: if verified v2 rules exist, ensure ChromaDB is populated locally without LLM calls
    if not force and _is_cached_rules_valid(rules_path):
        meta = json.loads(rules_path.read_text()).get("_meta", {})
        coll, already = get_or_create_collection(pdf_path)
        if already and coll.count() > 0:
            if progress_callback:
                progress_callback(10, 10, f"Loaded verified policy '{pdf_path.name}'")
            return {
                "pdf_hash": pdf_hash,
                "policy_dir": str(pol_dir),
                "rules_path": str(rules_path),
                "num_chunks": coll.count(),
                "validation": meta.get("validation", {"is_health_insurance_document": True}),
                "cached": True,
            }
        # Cold container (e.g. Streamlit Cloud): embed locally into ChromaDB without re-extracting rules
        if progress_callback:
            progress_callback(3, 10, f"Indexing '{pdf_path.name}' into local vector store...")
        n_chunks = ingest(pdf_path)
        if progress_callback:
            progress_callback(10, 10, f"Loaded verified policy '{pdf_path.name}'")
        return {
            "pdf_hash": pdf_hash,
            "policy_dir": str(pol_dir),
            "rules_path": str(rules_path),
            "num_chunks": n_chunks,
            "validation": meta.get("validation", {"is_health_insurance_document": True}),
            "cached": True,
        }

    if progress_callback:
        progress_callback(0, 10, f"Validating '{pdf_path.name}'...")
    validation_info, pages = validate_policy_pdf(pdf_path)

    if progress_callback:
        progress_callback(2, 10, f"Indexing {len(pages)} pages into local vector store...")
    n_chunks = ingest(pdf_path, pages=pages)

    def _scaled_progress(done: int, total: int, msg: str):
        if progress_callback:
            pct_step = 2 + int((done / max(total, 1)) * 8)
            progress_callback(pct_step, 10, msg)

    rules_path = extract_all(
        pdf_path,
        force=force,
        validation_info=validation_info,
        progress_callback=_scaled_progress,
    )

    return {
        "pdf_hash": pdf_hash,
        "policy_dir": str(pol_dir),
        "rules_path": str(rules_path),
        "num_chunks": n_chunks,
        "validation": validation_info,
        "cached": False,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python policy_processor.py <path-to-pdf> [--force]")
        sys.exit(1)
    pdf = Path(sys.argv[1])
    if not pdf.exists():
        print(f"error: not found: {pdf}")
        sys.exit(1)
    result = process(
        pdf,
        force="--force" in sys.argv,
        progress_callback=lambda d, t, m: print(f"  [{d}/{t}] {m}", flush=True),
    )
    print("\n=== done ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

