"""
FinRAG Analyst — Generation Layer
==================================
Wires retrieval → LLM call → answer + citations.

Usage (from src/generation/):
    python generator.py
"""

import sys
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

# ── Load API key from .env ─────────────────────────────────────────────────────
# Walk up from src/generation/ to find the project root .env
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found. Check your .env file in the project root.")

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Import retrieval from ingestion layer ──────────────────────────────────────
_ingestion_dir = str(Path(__file__).resolve().parent.parent / "ingestion_retrieval")
if _ingestion_dir not in sys.path:
    sys.path.insert(0, _ingestion_dir)
from chunker import retrieve, CONFIG
from sentence_transformers import SentenceTransformer

# Load embedding model once at module level (expensive to reload)
print("Loading embedding model...")
_model = SentenceTransformer(CONFIG["embedding_model"])
print("Model ready.\n")


# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are FinRAG Analyst, an expert financial analyst assistant.

You answer questions strictly based on the provided source passages from SEC filings 
(10-K, 10-Q) and financial documents. 

Rules you must follow:
1. Answer ONLY using information from the provided sources. Do not use prior knowledge.
2. Always cite your sources by referencing the company, document type, and period 
   (e.g., "According to Apple's 10-K FY2024...").
3. If the sources do not contain enough information to answer the question, say: 
   "The available documents do not contain sufficient information to answer this question."
4. Be precise with numbers — quote exact figures from the sources when available.
5. Keep answers concise and structured.
"""


# ── COMPANY DETECTION ──────────────────────────────────────────────────────────

def detect_company(query: str) -> Optional[str]:
    """Return company name if unambiguously mentioned in the query."""
    q = query.lower()
    if "apple" in q or "aapl" in q:
        return "Apple"
    if "tesla" in q or "tsla" in q:
        return "Tesla"
    if "jpmorgan" in q or "jp morgan" in q or "jpm" in q or "chase" in q:
        return "JPMorgan"
    return None


# ── QUERY EXPANSION ────────────────────────────────────────────────────────────

def expand_query(query: str, model: str = "gpt-4o-mini") -> str:
    """
    Use the LLM to rewrite the query with financial synonyms and alternative
    phrasings, improving recall for embedding-based retrieval.
    Apple's fiscal calendar offset makes this especially important.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You are a financial search query expander. "
                "Rewrite the user's query as a single expanded search string that includes: "
                "synonyms (e.g. 'net income' → also 'profit, earnings, net earnings'), "
                "alternative time references (e.g. 'last quarter of 2025' → also "
                "'Q4 2025, Q1 2026, first quarter fiscal 2026, October December 2025'), "
                "and both formal and informal phrasings. "
                "Output only the expanded query string, nothing else."
            )},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=128,
    )
    return response.choices[0].message.content.strip()


# ── CORE FUNCTION ──────────────────────────────────────────────────────────────

def answer(
    query: str,
    top_k: int = 10,
    filter_company: Optional[str] = None,
    filter_doc_type: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → LLM call → return answer + sources.
    ...
    """
    # ── Step 1: Expand query then retrieve relevant chunks ────────────────────
    expanded = expand_query(query, model=model)
    print(f"\nExpanded query: {expanded}")          # ← add
    
    effective_company = filter_company or detect_company(query)
    print(f"Filter company: {effective_company}")   # ← add
    
    chunks = retrieve(
        query=expanded,
        model=_model,
        top_k=top_k,
        filter_company=effective_company,
        filter_doc_type=filter_doc_type,
    )
    # Prioritise table chunks (they contain the actual numbers) then by score
    chunks = sorted(
        chunks,
        key=lambda c: (0 if c["metadata"]["element_type"] == "table" else 1, -c["score"])
    )
    
    print(f"Chunks retrieved: {len(chunks)}")       # ← add
    for i, c in enumerate(chunks, 1):               # ← add
        print(f"  [{i}] {c['metadata']['period']} | {c['text'][:100]}")  # ← add

    if not chunks:
        return {
            "question": query,
            "answer": "No relevant documents found in the index.",
            "sources": [],
        }

    # ── Step 2: Build context from retrieved chunks ────────────────────────────
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source_label = (
            f"[Source {i}] "
            f"{meta['company']} | {meta['doc_type']} {meta['period']} | "
            f"Section: {meta['section_title'] or 'N/A'} | "
            f"Type: {meta['element_type']}"
        )
        context_parts.append(f"{source_label}\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    # ── Step 3: Build the user message ────────────────────────────────────────
    user_message = f"""Answer the following question using only the sources provided below.

Question: {query}

Sources:
{context}

Answer:"""

    # ── Step 4: Call the LLM ──────────────────────────────────────────────────
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,    # deterministic — important for financial Q&A
        max_tokens=2048,
    )

    answer_text = response.choices[0].message.content.strip()

    # ── Step 5: Format sources for output ─────────────────────────────────────
    sources = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        sources.append({
            "source_num": i,
            "company":       meta["company"],
            "doc_type":      meta["doc_type"],
            "period":        meta["period"],
            "section_title": meta.get("section_title") or "N/A",
            "element_type":  meta["element_type"],
            "score":         round(chunk["score"], 4),
            "text_preview":  chunk["text"][:200] + "...",
        })

    return {
        "question": query,
        "answer":   answer_text,
        "sources":  sources,
    }


# ── PRETTY PRINT ───────────────────────────────────────────────────────────────

def print_result(result: dict):
    print("\n" + "=" * 65)
    print(f"QUESTION: {result['question']}")
    print("=" * 65)
    print(f"\nANSWER:\n{result['answer']}")
    print("\n" + "-" * 65)
    print("SOURCES:")
    for s in result["sources"]:
        print(
            f"  [{s['source_num']}] score={s['score']} | "
            f"{s['company']} {s['doc_type']} {s['period']} | "
            f"{s['element_type']} | section: {s['section_title']}"
        )
        print(f"       {s['text_preview']}")
    print("=" * 65 + "\n")


# ── QUICK TEST ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "What was Apple's net income for the last quarter of 2025?"
        ]


    for question in test_questions:
        result = answer(question, top_k=10)
        print_result(result)
