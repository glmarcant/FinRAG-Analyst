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


# ── CORE FUNCTION ──────────────────────────────────────────────────────────────

def answer(
    query: str,
    top_k: int = 5,
    filter_company: Optional[str] = None,
    filter_doc_type: Optional[str] = None,
    model: str = "gpt-4o-mini",    # cheap and fast for development
) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → LLM call → return answer + sources.

    Returns:
        {
            "question": str,
            "answer": str,
            "sources": [{"company", "doc_type", "period", "section_title", 
                         "element_type", "score", "text"}, ...]
        }
    """
    # ── Step 1: Retrieve relevant chunks ──────────────────────────────────────
    chunks = retrieve(
        query=query,
        model=_model,
        top_k=top_k,
        filter_company=filter_company,
        filter_doc_type=filter_doc_type,
    )

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
        "What was Apple's total revenue in 2024?",
        "What are Tesla's main risk factors?",
        "What is JPMorgan's net income?",
    ]

    for question in test_questions:
        result = answer(question, top_k=5)
        print_result(result)
