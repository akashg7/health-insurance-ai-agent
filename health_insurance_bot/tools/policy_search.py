"""Thresholded hybrid RAG search over the uploaded policy PDF."""

from langchain_core.tools import tool

from ._context import get_context
from ingest import hybrid_search

MIN_SEARCH_SIMILARITY = 0.20


@tool
def search_policy_docs(query: str, k: int = 5) -> list[dict]:
    """Search the active uploaded health insurance PDF for clauses matching a query.

    Use this for:
      - Open-ended 'why' or 'explain' questions about policy wording, definitions, or rules.
      - Any condition/procedure where check_coverage returned status='not_mentioned'.
      - Verifying exact wording on exclusions, sub-limits, waiting periods, or add-ons.

    Returns matching chunks with page numbers and similarity scores, OR a 'no_relevant_match'
    status if the active PDF contains no relevant clauses.
    """
    ctx = get_context()
    if ctx.pdf_path is None:
        return [{"status": "error", "message": "No policy PDF loaded in this session."}]

    keywords = [w for w in query.lower().split() if len(w) >= 4]
    hits = hybrid_search(
        ctx.pdf_path,
        query,
        keywords=keywords,
        k=k,
        min_similarity=MIN_SEARCH_SIMILARITY,
    )
    if not hits:
        return [
            {
                "status": "no_relevant_match",
                "document": ctx.pdf_path.name,
                "message": (
                    f"No clauses relevant to '{query}' were found in '{ctx.pdf_path.name}' "
                    f"(above similarity threshold {MIN_SEARCH_SIMILARITY})."
                ),
            }
        ]

    return [
        {
            "document": ctx.pdf_path.name,
            "page": h["page"],
            "similarity_score": round(float(h["similarity"]), 3),
            "text": h["text"],
        }
        for h in hits
    ]
