"""PDF -> ChromaDB ingestion pipeline using cosine similarity, hybrid keyword search, and local MiniLM embeddings.

Idempotent per PDF SHA-256 hash. Rejects empty / scanned image-only PDFs cleanly
and supports similarity-thresholded retrieval + lexical keyword matching.
"""

import hashlib
import sys
import threading
from pathlib import Path

import chromadb
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).parent / "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
MIN_PDF_TEXT_CHARS = 150

_chroma_client = None
_embedder = None
_lock = threading.Lock()


class EmptyPDFError(ValueError):
    """Raised when a PDF is unreadable, encrypted, empty, or contains scanned images without text."""


def _get_chroma():
    global _chroma_client
    with _lock:
        if _chroma_client is None:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def _get_embedder():
    global _embedder
    with _lock:
        if _embedder is None:
            try:
                _embedder = SentenceTransformer(EMBED_MODEL)
            except Exception:
                _embedder = SentenceTransformer(EMBED_MODEL, local_files_only=True)
    return _embedder


def pdf_fingerprint(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]


def collection_name_for(pdf_path: Path) -> str:
    return f"policy_{pdf_fingerprint(pdf_path)}"


def parse_pdf_to_pages(pdf_path: Path) -> tuple[list[dict], int]:
    """Extract selectable text page-by-page. Returns (non_empty_pages, total_page_count)."""
    pages: list[dict] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append({"page": i, "text": text})
    except Exception as exc:
        raise EmptyPDFError(f"Could not open or read PDF '{pdf_path.name}': {exc}") from exc

    total_chars = sum(len(p["text"]) for p in pages)
    if not pages or total_chars < MIN_PDF_TEXT_CHARS:
        raise EmptyPDFError(
            f"'{pdf_path.name}' contains too little selectable text ({total_chars} chars across {total_pages} pages). "
            "Scanned or image-only PDFs are not supported without OCR."
        )
    return pages, total_pages


def extract_preview_text(pages: list[dict], max_chars: int = 3200) -> str:
    """Build a representative text preview spanning early and middle pages for document classification."""
    if not pages:
        return ""
    indices = [0]
    if len(pages) > 1:
        indices.append(1)
    if len(pages) > 2:
        indices.append(2)
    if len(pages) > 5:
        indices.append(len(pages) // 2)
    if len(pages) > 10:
        indices.append((len(pages) * 3) // 4)

    seen = set()
    parts: list[str] = []
    remaining = max_chars
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        page_obj = pages[idx]
        snippet = page_obj["text"][: min(900, remaining)]
        if snippet:
            parts.append(f"[Page {page_obj['page']}]:\n{snippet}")
            remaining -= len(snippet)
        if remaining <= 100:
            break
    return "\n\n".join(parts)


def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for p in pages:
        for j, ch in enumerate(splitter.split_text(p["text"])):
            cleaned = ch.strip()
            if cleaned:
                chunks.append({
                    "id": f"p{p['page']:03d}_c{j:02d}",
                    "text": cleaned,
                    "page": p["page"],
                })
    return chunks


def get_or_create_collection(pdf_path: Path):
    """Return (cosine_collection, already_existed). Recreates legacy non-cosine collections."""
    client = _get_chroma()
    name = collection_name_for(pdf_path)
    existing = {c.name for c in client.list_collections()}
    if name in existing:
        coll = client.get_collection(name)
        space = (coll.metadata or {}).get("hnsw:space")
        if space == "cosine":
            return coll, True
        client.delete_collection(name)
    return client.create_collection(name=name, metadata={"hnsw:space": "cosine"}), False


def ingest(pdf_path: Path, pages: list[dict] | None = None) -> int:
    """Parse, chunk, embed, and index a readable PDF into a cosine ChromaDB collection."""
    coll, already = get_or_create_collection(pdf_path)
    if already and coll.count() > 0:
        return coll.count()

    if pages is None:
        pages, _ = parse_pdf_to_pages(pdf_path)

    chunks = chunk_pages(pages)
    if not chunks:
        raise EmptyPDFError(f"No text chunks could be created from '{pdf_path.name}'.")

    embedder = _get_embedder()
    vectors = embedder.encode(
        [c["text"] for c in chunks],
        show_progress_bar=False,
        batch_size=32,
    ).tolist()

    coll.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[{"page": c["page"]} for c in chunks],
    )
    return coll.count()


def search(
    pdf_path: Path,
    query: str,
    k: int = 5,
    min_similarity: float | None = None,
) -> list[dict]:
    """Cosine semantic search returning chunks sorted by relevance with bounded [0, 1] score."""
    coll, _ = get_or_create_collection(pdf_path)
    total = coll.count()
    if total == 0:
        return []

    n_results = max(1, min(k, total))
    embedder = _get_embedder()
    qvec = embedder.encode([query], show_progress_bar=False).tolist()[0]
    res = coll.query(query_embeddings=[qvec], n_results=n_results)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    hits: list[dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        similarity = round(max(0.0, min(1.0, 1.0 - float(dist))), 4)
        if min_similarity is not None and similarity < min_similarity:
            continue
        hits.append({
            "text": doc,
            "page": meta["page"],
            "distance": round(float(dist), 4),
            "similarity": similarity,
        })
    return hits


def hybrid_search(
    pdf_path: Path,
    query: str,
    keywords: list[str] | None = None,
    k: int = 5,
    min_similarity: float = 0.15,
) -> list[dict]:
    """Combine cosine semantic search with exact lexical keyword hits across policy chunks."""
    semantic_hits = search(pdf_path, query, k=k, min_similarity=min_similarity)
    by_key: dict[tuple[int, str], dict] = {
        (int(h["page"]), h["text"]): {**h, "keyword_hits": 0}
        for h in semantic_hits
    }

    kw_list = [kw.lower().strip() for kw in (keywords or []) if len(kw.strip()) >= 3]
    if kw_list:
        coll, _ = get_or_create_collection(pdf_path)
        all_rows = coll.get(include=["documents", "metadatas"])
        docs = all_rows.get("documents") or []
        metas = all_rows.get("metadatas") or []
        for doc, meta in zip(docs, metas):
            doc_l = doc.lower()
            kw_count = sum(1 for kw in kw_list if kw in doc_l)
            if kw_count > 0:
                page = int(meta["page"])
                key = (page, doc)
                if key in by_key:
                    by_key[key]["keyword_hits"] = max(by_key[key]["keyword_hits"], kw_count)
                else:
                    by_key[key] = {
                        "text": doc,
                        "page": page,
                        "distance": 0.4,
                        "similarity": min(0.95, 0.55 + 0.1 * kw_count),
                        "keyword_hits": kw_count,
                    }

    ranked = sorted(
        by_key.values(),
        key=lambda x: (x["keyword_hits"], x["similarity"]),
        reverse=True,
    )
    return ranked[:k]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python ingest.py <path-to-pdf> [smoke-test-query]")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    if not pdf.exists():
        print(f"error: file not found: {pdf}")
        sys.exit(1)

    count = ingest(pdf)
    print(f"Indexed {count} chunks for {pdf.name}")

    if len(sys.argv) > 2:
        q = " ".join(sys.argv[2:])
        print(f"\n--- smoke-test query: {q!r} ---")
        for i, hit in enumerate(hybrid_search(pdf, q, keywords=q.split(), k=3), 1):
            preview = hit["text"][:180].replace("\n", " ")
            print(f"\n[hit {i}] page {hit['page']}  sim={hit['similarity']:.3f}  kw={hit['keyword_hits']}")
            print(f"  {preview}...")
