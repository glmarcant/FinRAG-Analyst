"""
FinRAG Analyst — Chunking & Indexing Pipeline
=============================================
Implements element-based structural chunking following:
  Jimeno Yepes et al. (2024) "Financial Report Chunking for Effective RAG"

Usage:
    python chunker.py                         # index all docs in config
    python chunker.py --inspect               # print chunk stats only, no indexing
    python chunker.py --baseline              # also build fixed-size baseline index
"""

import re
import uuid
import json
import argparse
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from typing import Optional

from bs4 import BeautifulSoup
from unstructured.partition.html import partition_html
from unstructured.documents.elements import (
    Title, Table, NarrativeText, ListItem,
    Header, Footer, Text
)
from sentence_transformers import SentenceTransformer
from extract_text import extract_primary_document

# ── CONFIG ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    # Vector store: "faiss" or "chroma"
    "vector_store": "faiss",

    # Paths (absolute so the script works from any working directory)
    "index_dir": str(PROJECT_ROOT / "indexes"),
    "chunks_dir": str(PROJECT_ROOT / "chunks"),

    # Embedding model (free, runs on CPU, good for financial text)
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",

    # Chunking parameters (from the paper)
    "max_chars": 2048,               # max chars before forcing a new chunk
    "min_chars": 100,                # skip chunks shorter than this

    # Baseline chunking (for comparison / evaluation)
    "baseline_chunk_tokens": 512,
    "baseline_overlap_tokens": 50,

    # SEC EDGAR request header (required — EDGAR blocks requests without this)
    "sec_user_agent": "FinRAG-Analyst finrag@university.edu",

    # Leave empty to auto-discover all files in data/processed/
    "documents": [],
}

TICKER_TO_COMPANY = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "JPM":  "JPMorgan",
}

RAW_DIR = PROJECT_ROOT / "data/raw/sec-edgar-filings"


def _read_period_date(submission: Path) -> str:
    """Read CONFORMED PERIOD OF REPORT from the SEC header (e.g. '20250331')."""
    try:
        with open(submission, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "CONFORMED PERIOD OF REPORT" in line:
                    return line.split(":")[-1].strip()
                if line.startswith("<DOCUMENT>"):
                    break
    except Exception:
        pass
    return ""


def _period_label(date_str: str, doc_type: str) -> str:
    """Convert a raw YYYYMMDD date string to a human period label."""
    if len(date_str) == 8:
        year  = int(date_str[:4])
        month = int(date_str[4:6])
        if doc_type == "10-K":
            return f"FY{year}"
        quarter = (month - 1) // 3 + 1
        return f"Q{quarter}-{year}"
    return "Unknown"


def discover_documents() -> list[tuple]:
    """
    Auto-discover all filings from data/raw/sec-edgar-filings/.
    Returns a list of (company, doc_type, period, path_to_full-submission.txt).
    """
    docs = []
    if not RAW_DIR.exists():
        return docs
    for ticker_dir in sorted(RAW_DIR.iterdir()):
        if not ticker_dir.is_dir():
            continue
        company = TICKER_TO_COMPANY.get(ticker_dir.name.upper(), ticker_dir.name)
        for doctype_dir in sorted(ticker_dir.iterdir()):
            if not doctype_dir.is_dir():
                continue
            doc_type = doctype_dir.name.upper()
            for accession_dir in sorted(doctype_dir.iterdir()):
                submission = accession_dir / "full-submission.txt"
                if not submission.exists():
                    continue
                date_str = _read_period_date(submission)
                period   = _period_label(date_str, doc_type)
                docs.append((company, doc_type, period, str(submission)))
    return docs

# ── HTML CLEANING ──────────────────────────────────────────────────────────────

def clean_sec_html(html: str) -> str:
    """
    Strip EDGAR-specific noise before passing to unstructured:
      - Inline XBRL tags (ix:*) — unwrapped, text preserved
      - Hidden elements (display:none)
      - Empty tags
      - Boilerplate navigation / cover page repetition
    """
    soup = BeautifulSoup(html, "html.parser")

    # Unwrap XBRL inline tags — keep their text content
    for tag in soup.find_all(re.compile(r"^ix:")):
        tag.unwrap()

    # Remove hidden elements
    for tag in soup.find_all(
        style=lambda s: s and "display:none" in s.replace(" ", "").lower()
    ):
        tag.decompose()

    # Remove script / style blocks
    for tag in soup.find_all(["script", "style", "meta", "link"]):
        tag.decompose()

    # Remove empty tags
    for tag in soup.find_all():
        if not tag.get_text(strip=True) and tag.name not in ["br", "hr", "img"]:
            tag.decompose()

    return str(soup)


# ── DOCUMENT LOADING ───────────────────────────────────────────────────────────

def load_html(source: str, doc_type: str = "") -> str:
    """Load HTML from a URL, a plain .htm file, or an EDGAR full-submission.txt container."""
    if source.startswith("http://") or source.startswith("https://"):
        print(f"  Fetching from URL: {source[:80]}...")
        headers = {"User-Agent": CONFIG["sec_user_agent"]}
        resp = requests.get(source, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")
    print(f"  Reading: {source}")
    content = path.read_text(encoding="utf-8", errors="replace")

    if path.name == "full-submission.txt" and doc_type:
        html = extract_primary_document(content, doc_type)
        if html is None or len(html) < 500:
            raise ValueError(f"Extracted HTML looks empty or missing for {doc_type} in {source}")
        # iXBRL filings wrap HTML inside <XBRL>...</XBRL> — strip the outer wrapper
        if re.match(r"\s*<XBRL", html, re.IGNORECASE):
            inner = re.search(r"(<html[\s>].*</html>)", html, re.DOTALL | re.IGNORECASE)
            if inner:
                html = inner.group(1)
        return html
    return content


# ── ELEMENT-BASED STRUCTURAL CHUNKING ─────────────────────────────────────────

def extract_elements(html: str):
    """Run unstructured element detection on cleaned HTML."""
    cleaned = clean_sec_html(html)
    elements = partition_html(text=cleaned)
    return elements


def chunk_elements(
    elements,
    company: str,
    doc_type: str,
    period: str,
    source_id: str,
) -> list[dict]:
    """
    Implement the element-based merging logic from Jimeno Yepes et al.:
      - Title  → always flush current buffer, start new chunk
      - Table  → always its own standalone chunk (never merged)
      - Text / List → merge until MAX_CHARS, then flush
    Each chunk gets a rich metadata dict.
    """
    max_chars = CONFIG["max_chars"]
    min_chars = CONFIG["min_chars"]

    chunks = []
    current_text = ""
    current_title = ""
    current_page = None

    def flush(page_num=None):
        """Save buffered text as a chunk if it meets min length."""
        nonlocal current_text
        text = current_text.strip()
        if len(text) >= min_chars:
            chunks.append(_make_chunk(
                text=text,
                section_title=current_title,
                element_type="text",
                company=company,
                doc_type=doc_type,
                period=period,
                source_id=source_id,
                page_number=page_num or current_page,
            ))
        current_text = ""

    for el in elements:
        page = getattr(el.metadata, "page_number", None)
        if page:
            current_page = page

        if isinstance(el, Title):
            # Flush whatever was accumulating, then start fresh with this title
            flush(page)
            current_title = str(el).strip()
            current_text = current_title + "\n"

        elif isinstance(el, Table):
            # Tables always get their own chunk — never merged with text
            flush(page)
            table_text = str(el).strip()
            if len(table_text) >= min_chars:
                chunks.append(_make_chunk(
                    text=table_text,
                    section_title=current_title,
                    element_type="table",
                    company=company,
                    doc_type=doc_type,
                    period=period,
                    source_id=source_id,
                    page_number=page,
                ))

        elif isinstance(el, Footer):
            continue

        elif isinstance(el, (NarrativeText, ListItem, Header)):
            el_text = str(el).strip()
            if not el_text:
                continue

            # Would adding this element exceed max_chars?
            candidate = current_text + el_text + "\n"
            if len(candidate) > max_chars and current_text.strip():
                flush(page)
                current_text = el_text + "\n"
            else:
                current_text = candidate

        elif isinstance(el, Text):
            # Include if it looks substantial (not just page numbers / whitespace)
            el_text = str(el).strip()
            if len(el_text) > 40:
                candidate = current_text + el_text + "\n"
                if len(candidate) > max_chars and current_text.strip():
                    flush(page)
                    current_text = el_text + "\n"
                else:
                    current_text = candidate

    flush()  # flush any remaining buffer
    return chunks


def _make_chunk(
    text: str,
    section_title: str,
    element_type: str,
    company: str,
    doc_type: str,
    period: str,
    source_id: str,
    page_number: Optional[int],
) -> dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "text": text,
        "metadata": {
            "company": company,
            "doc_type": doc_type,       # "10-K", "10-Q", "transcript"
            "period": period,            # e.g. "FY2023", "Q1-2024"
            "section_title": section_title,
            "element_type": element_type,  # "text" or "table"
            "page_number": page_number,
            "source_id": source_id,
            "char_length": len(text),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        },
    }


# ── BASELINE CHUNKER (for evaluation comparison) ───────────────────────────────

def chunk_fixed_size(
    full_text: str,
    company: str,
    doc_type: str,
    period: str,
    source_id: str,
    chunk_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[dict]:
    """
    Fixed-size token-based chunking — baseline from the paper (Base 512).
    Used to compare against element-based chunking in evaluation.
    """
    words = full_text.split()
    step = chunk_tokens - overlap_tokens
    chunks = []

    for i in range(0, len(words), step):
        window = words[i : i + chunk_tokens]
        text = " ".join(window).strip()
        if len(text) < CONFIG["min_chars"]:
            continue
        chunks.append(_make_chunk(
            text=text,
            section_title="",
            element_type="fixed",
            company=company,
            doc_type=doc_type,
            period=period,
            source_id=source_id,
            page_number=None,
        ))

    return chunks


# ── EMBEDDING ──────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """Embed all chunk texts and return as float32 numpy array."""
    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via dot product
    )
    return embeddings.astype(np.float32)


# ── VECTOR STORE: FAISS ────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray, chunks: list[dict], index_name: str):
    """Build and save a FAISS flat inner-product index."""
    import faiss

    index_dir = Path(CONFIG["index_dir"]) / index_name
    index_dir.mkdir(parents=True, exist_ok=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product = cosine sim (embeddings normalised)
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / "index.faiss"))

    # Save metadata (chunk text + metadata) separately
    with open(index_dir / "chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"  FAISS index saved → {index_dir}  ({index.ntotal} vectors, dim={dim})")
    return index


def load_faiss_index(index_name: str):
    """Load a saved FAISS index and its chunk metadata."""
    import faiss

    index_dir = Path(CONFIG["index_dir"]) / index_name
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "chunks.json") as f:
        chunks = json.load(f)
    return index, chunks


# ── VECTOR STORE: CHROMADB ─────────────────────────────────────────────────────

def build_chroma_index(embeddings: np.ndarray, chunks: list[dict], index_name: str):
    """Build and persist a ChromaDB collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(Path(CONFIG["index_dir"]) / index_name))
    # Delete existing collection if rebuilding
    try:
        client.delete_collection(index_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=index_name,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB expects lists, not numpy arrays
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"  ChromaDB collection saved → {CONFIG['index_dir']}/{index_name}  ({len(chunks)} docs)")
    return collection


def load_chroma_index(index_name: str):
    """Load an existing ChromaDB collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(Path(CONFIG["index_dir"]) / index_name))
    collection = client.get_collection(index_name)
    return collection


# ── CHUNK STATS / INSPECTION ───────────────────────────────────────────────────

def print_chunk_stats(chunks: list[dict], label: str = ""):
    """Print a summary of chunk statistics — useful for debugging."""
    if not chunks:
        print("  No chunks to report.")
        return

    lengths = [c["metadata"]["char_length"] for c in chunks]
    types = Counter(c["metadata"]["element_type"] for c in chunks)
    companies = Counter(c["metadata"]["company"] for c in chunks)

    print(f"\n{'='*55}")
    print(f"  Chunk stats — {label}")
    print(f"{'='*55}")
    print(f"  Total chunks   : {len(chunks)}")
    print(f"  Avg length     : {int(np.mean(lengths))} chars")
    print(f"  Min / Max      : {min(lengths)} / {max(lengths)} chars")
    print(f"  Element types  : {dict(types)}")
    print(f"  By company     : {dict(companies)}")
    print(f"{'='*55}\n")


def save_chunks_json(chunks: list[dict], name: str):
    """Save raw chunks to disk for inspection / evaluation."""
    out_dir = Path(CONFIG["chunks_dir"])
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"  Raw chunks saved → {path}")


# ── MAIN PIPELINE ──────────────────────────────────────────────────────────────

def run_pipeline(inspect_only: bool = False, build_baseline: bool = False):
    # NOTE: intentionally "or" not "+"; combining both would index the same filing twice.
    docs = CONFIG["documents"] or discover_documents()
    if not docs:
        print("\n[!] No documents found in data/raw/sec-edgar-filings/ and none configured.\n")
        return

    if not inspect_only:
        # Clear stale chunks and indexes before rebuilding
        chunks_dir = Path(CONFIG["chunks_dir"])
        if chunks_dir.exists():
            removed = list(chunks_dir.glob("*.json"))
            for f in removed:
                f.unlink()
            if removed:
                print(f"Deleted {len(removed)} old chunk file(s).")

        index_dir = Path(CONFIG["index_dir"])
        if index_dir.exists():
            import shutil
            for d in index_dir.iterdir():
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
            print("Cleared old indexes.")

    print(f"\nLoading embedding model: {CONFIG['embedding_model']}")
    model = SentenceTransformer(CONFIG["embedding_model"])

    all_chunks = []
    all_baseline_chunks = []

    for company, doc_type, period, source in docs:
        source_id = f"{company}_{doc_type}_{period}".replace(" ", "_")
        print(f"\n[{source_id}]")

        try:
            html = load_html(source, doc_type)
        except Exception as e:
            print(f"  ERROR loading document: {e}")
            continue

        # Element-based chunking
        print("  Extracting elements...")
        elements = extract_elements(html)
        type_counts = Counter(type(el).__name__ for el in elements)
        print(f"  Elements found: {dict(type_counts)}")

        chunks = chunk_elements(elements, company, doc_type, period, source_id)
        print(f"  → {len(chunks)} structural chunks")
        all_chunks.extend(chunks)
        save_chunks_json(chunks, source_id)

        # Fixed-size baseline (optional)
        if build_baseline:
            full_text = " ".join(str(el) for el in elements)
            baseline = chunk_fixed_size(
                full_text, company, doc_type, period, source_id,
                chunk_tokens=CONFIG["baseline_chunk_tokens"],
                overlap_tokens=CONFIG["baseline_overlap_tokens"],
            )
            print(f"  → {len(baseline)} baseline chunks (fixed-512)")
            all_baseline_chunks.extend(baseline)

    if not all_chunks:
        print("\nNo chunks produced. Check your document sources.")
        return

    print_chunk_stats(all_chunks, "element-based")
    if build_baseline:
        print_chunk_stats(all_baseline_chunks, "baseline fixed-512")

    if inspect_only:
        print("Inspect-only mode — skipping indexing.")
        return

    # ── Embed and index ────────────────────────────────────────────────────────
    print("\nBuilding element-based index...")
    embeddings = embed_chunks(all_chunks, model)

    if CONFIG["vector_store"] == "faiss":
        build_faiss_index(embeddings, all_chunks, "finrag_structural")
    else:
        build_chroma_index(embeddings, all_chunks, "finrag_structural")

    if build_baseline and all_baseline_chunks:
        print("\nBuilding baseline index...")
        baseline_embeddings = embed_chunks(all_baseline_chunks, model)
        if CONFIG["vector_store"] == "faiss":
            build_faiss_index(baseline_embeddings, all_baseline_chunks, "finrag_baseline")
        else:
            build_chroma_index(baseline_embeddings, all_baseline_chunks, "finrag_baseline")

    print("\n✓ Pipeline complete.\n")


# ── RETRIEVAL HELPER (used by the RAG query layer) ─────────────────────────────

def retrieve(
    query: str,
    model: SentenceTransformer,
    top_k: int = 10,
    index_name: str = "finrag_structural",
    filter_company: Optional[str] = None,
    filter_doc_type: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve top-k chunks for a query.
    Works with whichever vector store is configured.
    Optionally filter by company or doc_type.

    Returns a list of dicts: {"text": ..., "metadata": ..., "score": ...}
    """
    query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)

    if CONFIG["vector_store"] == "faiss":
        index, chunks = load_faiss_index(index_name)
        scores, indices = index.search(query_vec, top_k * 3)  # over-fetch for filtering
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = chunks[idx]
            meta = chunk["metadata"]
            if filter_company and meta.get("company") != filter_company:
                continue
            if filter_doc_type and meta.get("doc_type") != filter_doc_type:
                continue
            results.append({**chunk, "score": float(score)})
            if len(results) >= top_k:
                break

    else:  # ChromaDB
        collection = load_chroma_index(index_name)
        where = {}
        if filter_company:
            where["company"] = filter_company
        if filter_doc_type:
            where["doc_type"] = filter_doc_type

        query_kwargs = dict(query_embeddings=query_vec.tolist(), n_results=top_k)
        if where:
            query_kwargs["where"] = where

        res = collection.query(**query_kwargs)
        results = []
        for i, doc_id in enumerate(res["ids"][0]):
            results.append({
                "chunk_id": doc_id,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score": 1 - res["distances"][0][i],  # chroma returns distance
            })

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinRAG chunking & indexing pipeline")
    parser.add_argument("--inspect",  action="store_true", help="Print stats only, skip indexing")
    parser.add_argument("--baseline", action="store_true", help="Also build fixed-size baseline index")
    args = parser.parse_args()

    run_pipeline(inspect_only=args.inspect, build_baseline=args.baseline)
