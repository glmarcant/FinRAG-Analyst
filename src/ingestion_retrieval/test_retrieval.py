"""
test_retrieval.py
─────────────────
Sanity check for the FAISS index built by chunker.py.
Runs a few test queries and prints the top retrieved chunks.

Usage (from src/ingestion/):
    python test_retrieval.py
"""

import sys
sys.path.insert(0, ".")

from sentence_transformers import SentenceTransformer
from chunker import retrieve, CONFIG

model = SentenceTransformer(CONFIG["embedding_model"])

TEST_QUERIES = [
    "What was Apple's total revenue in 2024?",
    "What are Tesla's main risk factors?",
    "What is JPMorgan's net income?",
    "How does Apple describe its competition?",
    "What is Tesla's capital expenditure?",
]

for query in TEST_QUERIES:
    print(f"\n{'='*60}")
    print(f"  QUERY: {query}")
    print(f"{'='*60}")

    results = retrieve(query, model, top_k=3)

    if not results:
        print("  No results returned — check that the index exists.")
        continue

    for i, r in enumerate(results):
        meta = r["metadata"]
        print(f"\n  [{i+1}] score={r['score']:.4f} | {meta['company']} {meta['doc_type']} {meta['period']}")
        print(f"       section: {meta['section_title'] or '(none)'}")
        print(f"       type: {meta['element_type']} | {meta['char_length']} chars")
        print(f"       {r['text'][:300].strip()}...")
