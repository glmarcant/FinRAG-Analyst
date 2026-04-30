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

# ── Paths (absolute, anchored to project root) ────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR       = _PROJECT_ROOT / "data/raw/sec-edgar-filings"
RAW_BASE      = _PROJECT_ROOT / "data/raw"
EXTRACTED_DIR = _PROJECT_ROOT / "data/extracted"
PROCESSED_DIR = _PROJECT_ROOT / "data/processed"

COMPANY_TO_TICKER = {
    "apple":     "AAPL",
    "tesla":     "TSLA",
    "jpmorgan":  "JPM",
    "jp morgan": "JPM",
}


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


def _read_period_date(submission_path: Path) -> str:
    """Read CONFORMED PERIOD OF REPORT from the SEC header (e.g. '20250331')."""
    try:
        with open(submission_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "CONFORMED PERIOD OF REPORT" in line:
                    return line.split(":")[-1].strip()
                if line.startswith("<DOCUMENT>"):
                    break
    except Exception:
        pass
    return ""


def _period_label(date_str: str, doc_type: str) -> str:
    """Convert YYYYMMDD to a human period label like FY2024 or Q1-2025."""
    if len(date_str) == 8:
        year  = int(date_str[:4])
        month = int(date_str[4:6])
        if doc_type == "10-K":
            return f"FY{year}"
        quarter = (month - 1) // 3 + 1
        return f"Q{quarter}-{year}"
    return "Unknown"


# ── Direct HTML processing (files dropped into data/raw/*.html) ───────────────

def parse_html_filename(filename: str) -> tuple:
    """
    Infer (ticker, doc_type, period) from filenames like:
      'Apple Inc. (Form_ 10-K_FY2025, Received_ ...)'
      'Apple Inc. (Form_ 10-Q_Q2FY2026, Received_ ...)'
      'Apple Inc. (Form_ 10-Q, Period_ 2026-03-29, Received_ ...)'
    Returns ("UNKNOWN", "10-K", "Unknown") for unrecognised patterns.
    """
    name = filename.lower()

    ticker = "UNKNOWN"
    for company, t in COMPANY_TO_TICKER.items():
        if company in name:
            ticker = t
            break

    doc_type = "10-Q" if ("10-q" in name or "10_q" in name) else "10-K"

    # Q{n}FY{year} — e.g. Q2FY2026 (most common EDGAR Online 10-Q format)
    qfy = re.search(r"q(\d)fy(\d{4})", name)
    # Q{n}-{year} or Q{n}_{year} — e.g. Q2-2025
    qdash = re.search(r"q(\d)[_\-](\d{4})", name)
    # Period_ YYYY-MM-DD — derive quarter from month
    period_date = re.search(r"period[_\s]+(\d{4})-(\d{2})-\d{2}", name)
    # FY{year} — annual filing fallback
    fy = re.search(r"fy(\d{4})", name)

    if qfy:
        period = f"Q{qfy.group(1)}-{qfy.group(2)}"
    elif qdash:
        period = f"Q{qdash.group(1)}-{qdash.group(2)}"
    elif period_date:
        year    = int(period_date.group(1))
        quarter = (int(period_date.group(2)) - 1) // 3 + 1
        period  = f"Q{quarter}-{year}"
    elif fy:
        period = f"FY{fy.group(1)}"
    else:
        period = "Unknown"

    return ticker, doc_type, period


def process_direct_html(html_path: Path):
    """
    Process a plain HTML filing dropped directly into data/raw/.
    Copies the file to data/processed/ with a standardised name so the
    chunker can discover it via TICKER_DOCTYPE_PERIOD.html naming.
    """
    ticker, doc_type, period = parse_html_filename(html_path.name)
    logger.info(f"  Direct HTML: {html_path.name} -> {ticker}/{doc_type}/{period}")

    out_html = PROCESSED_DIR / f"{ticker}_{doc_type}_{period}.html"
    try:
        out_html.write_bytes(html_path.read_bytes())
    except Exception as e:
        logger.error(f"    Could not copy file: {e}")
        return None

    logger.info(f"    Saved: {out_html.name}")
    return {"ticker": ticker, "doc_type": doc_type, "period": period}


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

    period = _period_label(_read_period_date(submission_path), doc_type)

    # iXBRL filings wrap the HTML inside <XBRL>...</XBRL> — extract the inner HTML
    html_for_chunker = primary_doc
    if re.match(r"\s*<XBRL", primary_doc, re.IGNORECASE):
        inner = re.search(r"(<html[\s>].*</html>)", primary_doc, re.DOTALL | re.IGNORECASE)
        if inner:
            html_for_chunker = inner.group(1)

    # Output filename: AAPL_10-K_0000320193-24-000123
    stem     = f"{ticker}_{doc_type}_{accession}"
    out_txt  = EXTRACTED_DIR / f"{stem}.txt"
    out_json = EXTRACTED_DIR / f"{stem}.json"
    out_html = PROCESSED_DIR / f"{stem}.html"

    out_txt.write_text(plain_text, encoding="utf-8")
    out_html.write_text(html_for_chunker, encoding="utf-8")

    metadata = {
        "ticker":      ticker,
        "doc_type":    doc_type,
        "accession":   accession,
        "period":      period,
        "source_file": str(submission_path),
        "word_count":  word_count,
        "char_count":  char_count,
        "was_html":    html,
    }
    out_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info(f"    Saved: {out_txt.name} ({word_count:,} words)  →  {out_html.name}")
    return metadata


# ── Main batch loop ────────────────────────────────────────────────────────────

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    total, success, failed = 0, 0, 0

    # --- Path 1: EDGAR full-submission.txt directory tree ---
    if RAW_DIR.exists():
        EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
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
                    success += 1 if result else 0
                    failed  += 0 if result else 1

    # --- Path 2: Plain HTML files dropped directly into data/raw/ ---
    for html_file in sorted(RAW_BASE.glob("*.html")):
        total += 1
        result = process_direct_html(html_file)
        success += 1 if result else 0
        failed  += 0 if result else 1

    logger.info(f"\n{'='*50}")
    logger.info(f"Extraction complete: {success}/{total} filings processed.")


if __name__ == "__main__":
    main()
