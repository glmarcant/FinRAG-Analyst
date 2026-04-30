"""
extract_text.py
───────────────
FinRAG Analyst — Phase 1: Text Extraction
Extracts readable text from SEC EDGAR full-submission.txt filings.

The full-submission.txt file is a raw EDGAR multi-document container.
It bundles several documents (10-K/10-Q HTML, exhibits, XML) together
with SGML-style headers. This script:
  1. Walks data/raw/sec-edgar-filings/<TICKER>/<TYPE>/<ACCESSION>/full-submission.txt
  2. Extracts the primary filing document (10-K or 10-Q) from the container
  3. Strips HTML tags using BeautifulSoup to get plain text
  4. Saves one .txt + one .json metadata file per filing to data/extracted/

Usage (run from project root):
    python src/ingestion/extract_text.py

Authors: Giulia Marcantonio & Maria Emilia Granda
Course:  DATA522
"""

import re
import json
import logging
from pathlib import Path
from bs4 import BeautifulSoup

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths (relative to project root) ──────────────────────────────────────────
RAW_DIR       = Path("data/raw/sec-edgar-filings")
EXTRACTED_DIR = Path("data/extracted")


# ── EDGAR document extraction ──────────────────────────────────────────────────

def extract_primary_document(submission_text: str, doc_type: str):
    """
    Parse the EDGAR full-submission.txt container and return the HTML/text
    content of the primary filing document (10-K or 10-Q).

    EDGAR containers look like:
        <DOCUMENT>
        <TYPE>10-K
        <SEQUENCE>1
        <TEXT>
        ... actual HTML content ...
        </TEXT>
        </DOCUMENT>

    We want SEQUENCE 1 with TYPE matching 10-K or 10-Q (the main filing),
    not exhibits or XML summaries.

    Args:
        submission_text: Full content of full-submission.txt as a string.
        doc_type:        "10-K" or "10-Q" (from folder structure).

    Returns:
        The raw HTML/text string of the primary document, or None if not found.
    """
    document_blocks = re.split(r"<DOCUMENT>", submission_text, flags=re.IGNORECASE)

    for block in document_blocks:
        type_match = re.search(r"<TYPE>\s*(\S+)", block, re.IGNORECASE)
        if not type_match:
            continue

        block_type = type_match.group(1).strip().upper()

        # Match exact type only — skip exhibits like 10-K/A, EX-31, etc.
        if block_type != doc_type.upper():
            continue

        text_match = re.search(r"<TEXT>(.*?)</TEXT>", block, re.DOTALL | re.IGNORECASE)
        if text_match:
            return text_match.group(1).strip()

    return None


def html_to_plain_text(html_content: str) -> str:
    """
    Convert HTML content to plain text using BeautifulSoup.

    Strategy:
      - Remove <script> and <style> blocks entirely
      - Insert newlines at paragraph/block boundaries
      - Use .get_text() for remaining tags

    Args:
        html_content: Raw HTML string from the EDGAR document.

    Returns:
        Plain text string.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    for tag in soup.find_all(["p", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5"]):
        tag.insert_before("\n")

    return soup.get_text(separator=" ")


def is_html(content: str) -> bool:
    """Heuristic check: does this content look like HTML?"""
    return bool(re.search(r"<html|<HTML|<body|<BODY|<table|<TABLE", content))


# ── Per-filing processing ──────────────────────────────────────────────────────

def process_filing(ticker: str, doc_type: str, accession: str, submission_path: Path):
    """
    Process a single full-submission.txt filing and write extracted output.

    Args:
        ticker:          e.g. "AAPL"
        doc_type:        e.g. "10-K"
        accession:       e.g. "0000320193-24-000123"
        submission_path: Path to the full-submission.txt file

    Returns:
        Metadata dict if successful, None if failed.
    """
    logger.info(f"  Processing {ticker} / {doc_type} / {accession}")

    try:
        raw = submission_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"    Could not read file: {e}")
        return None

    primary_doc = extract_primary_document(raw, doc_type)

    if primary_doc is None:
        logger.warning(f"    Could not find primary {doc_type} document block — skipping.")
        return None

    html = is_html(primary_doc)
    plain_text = html_to_plain_text(primary_doc) if html else primary_doc

    word_count = len(plain_text.split())
    char_count = len(plain_text)

    # Output filename: AAPL_10-K_0000320193-24-000123
    stem     = f"{ticker}_{doc_type}_{accession}"
    out_txt  = EXTRACTED_DIR / f"{stem}.txt"
    out_json = EXTRACTED_DIR / f"{stem}.json"

    out_txt.write_text(plain_text, encoding="utf-8")

    metadata = {
        "ticker":      ticker,
        "doc_type":    doc_type,
        "accession":   accession,
        "source_file": str(submission_path),
        "word_count":  word_count,
        "char_count":  char_count,
        "was_html":    html,
    }
    out_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info(f"    Saved: {out_txt.name} ({word_count:,} words)")
    return metadata


# ── Main batch loop ────────────────────────────────────────────────────────────

def main():
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        logger.error(f"Raw directory not found: {RAW_DIR}")
        return

    manifest = []
    total, success, failed = 0, 0, 0

    # Walk: sec-edgar-filings/<TICKER>/<DOC_TYPE>/<ACCESSION>/full-submission.txt
    for ticker_dir in sorted(RAW_DIR.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name.upper()

        for doctype_dir in sorted(ticker_dir.iterdir()):
            if not doctype_dir.is_dir():
                continue
            doc_type = doctype_dir.name.upper()

            for accession_dir in sorted(doctype_dir.iterdir()):
                if not accession_dir.is_dir():
                    continue

                submission_file = accession_dir / "full-submission.txt"
                if not submission_file.exists():
                    logger.warning(f"  No full-submission.txt in {accession_dir.name}")
                    continue

                total += 1
                result = process_filing(ticker, doc_type, accession_dir.name, submission_file)

                if result:
                    manifest.append(result)
                    success += 1
                else:
                    failed += 1

    manifest_path = EXTRACTED_DIR / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info(f"\n{'='*50}")
    logger.info(f"Extraction complete.")
    logger.info(f"  Total filings : {total}")
    logger.info(f"  Successful    : {success}")
    logger.info(f"  Failed        : {failed}")
    logger.info(f"  Manifest      : {manifest_path}")


if __name__ == "__main__":
    main()
