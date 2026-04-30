"""
clean_text.py
─────────────
FinRAG Analyst — Phase 1: Text Cleaning
Cleans raw extracted text from SEC EDGAR filings in preparation for chunking.

Reads .txt files from data/extracted/ (produced by extract_text.py),
applies a cleaning pipeline, and writes cleaned files to data/processed/.

NOTE: This module does NOT chunk the text. Chunking is the next step.

Usage (run from project root):
    python src/ingestion/clean_text.py

Authors: Giulia Marcantonio & Maria Emilia Granda
Course:  DATA522
"""

import re
import json
import logging
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
EXTRACTED_DIR = Path("data/extracted")
PROCESSED_DIR = Path("data/processed")


# ── Cleaning functions ─────────────────────────────────────────────────────────

def normalise_unicode(text: str) -> str:
    """Replace common Unicode characters with ASCII equivalents."""
    replacements = {
        "\u2018": "'", "\u2019": "'",   # curly single quotes
        "\u201C": '"', "\u201D": '"',   # curly double quotes
        "\u2013": "-", "\u2014": "-",   # en/em dash
        "\u2026": "...",                # ellipsis
        "\u00A0": " ",                  # non-breaking space
        "\u2022": "-", "\u2023": "-", "\u25CF": "-",  # bullets
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def remove_non_printable(text: str) -> str:
    """Strip non-printable/control characters, preserving standard whitespace."""
    return re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)


def remove_edgar_artifacts(text: str) -> str:
    """
    Remove EDGAR-specific artifacts that appear after HTML stripping.

    Targets:
      - Leftover SGML tags like <SEQUENCE>, <FILENAME>, <DESCRIPTION>
      - SEC header blocks (lines starting with SEC FILE NUMBER, FILED AS OF etc.)
      - XBRL remnants and inline tags like ix:nonNumeric
      - Table of contents dot leaders
    """
    # Remove remaining SGML-style header tags
    text = re.sub(r"<[A-Z\-]+>.*?\n", "", text)

    # Remove XBRL/iXBRL inline tags (text content is kept, just tags removed)
    text = re.sub(r"ix:[a-zA-Z]+", "", text)

    # Remove SEC header metadata lines (common patterns)
    text = re.sub(
        r"(?im)^(FILED AS OF DATE|DATE AS OF CHANGE|FILER|COMPANY DATA|"
        r"FILING VALUES|BUSINESS ADDRESS|MAIL ADDRESS|FORMER COMPANY|"
        r"SEC FILE NUMBER|FILM NUMBER|FORM TYPE|PUBLIC DOCUMENT COUNT).*$",
        "",
        text,
    )

    # Remove dot leaders from table of contents (e.g. "Revenue ......... 42")
    text = re.sub(r"\.{3,}", " ", text)

    return text


def remove_headers_footers(text: str) -> str:
    """Remove repeating page numbers and common page header/footer patterns."""
    # Lone page numbers on their own line
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    # "Page X of Y" variants
    text = re.sub(r"(?i)page\s+\d+\s+(of\s+\d+)?", "", text)
    # Dash-padded page numbers
    text = re.sub(r"[-–—]\s*\d{1,4}\s*[-–—]", "", text)
    return text


def fix_hyphenation(text: str) -> str:
    """Rejoin words split across lines by a hyphen (common in PDF-origin HTML)."""
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def remove_excessive_punctuation(text: str) -> str:
    """Remove underscore lines, long dash dividers, and equals dividers."""
    text = re.sub(r"_{3,}", " ", text)
    text = re.sub(r"-{4,}", " ", text)
    text = re.sub(r"={4,}", " ", text)
    return text


def normalise_whitespace(text: str) -> str:
    """
    Standardise whitespace — always applied last.
      1. Collapse horizontal whitespace to single space
      2. Collapse 3+ newlines to 2
      3. Strip overall leading/trailing whitespace
    """
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Master pipeline ────────────────────────────────────────────────────────────

def clean_text(raw_text: str) -> str:
    """
    Apply the full cleaning pipeline in the correct order.

    Order:
      1. Unicode normalisation   (so regex patterns work correctly)
      2. Non-printable removal
      3. EDGAR artifact removal  (SGML tags, XBRL, header metadata)
      4. Header/footer removal   (page numbers)
      5. Hyphenation fix         (before whitespace normalisation)
      6. Excessive punctuation
      7. Whitespace normalisation (always last)
    """
    text = normalise_unicode(raw_text)
    text = remove_non_printable(text)
    text = remove_edgar_artifacts(text)
    text = remove_headers_footers(text)
    text = fix_hyphenation(text)
    text = remove_excessive_punctuation(text)
    text = normalise_whitespace(text)
    return text


# ── Batch processing ───────────────────────────────────────────────────────────

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(EXTRACTED_DIR.glob("*.txt"))
    # Exclude internal manifest files
    txt_files = [f for f in txt_files if not f.name.startswith("_")]

    if not txt_files:
        logger.warning(f"No .txt files found in '{EXTRACTED_DIR}'. Run extract_text.py first.")
        return

    logger.info(f"Found {len(txt_files)} file(s) to clean.\n")

    report = []

    for txt_file in txt_files:
        logger.info(f"Cleaning: {txt_file.name}")

        raw_text     = txt_file.read_text(encoding="utf-8")
        original_len = len(raw_text)

        cleaned      = clean_text(raw_text)
        cleaned_len  = len(cleaned)
        reduction    = (1 - cleaned_len / original_len) * 100 if original_len > 0 else 0

        # Write cleaned text
        out_txt = PROCESSED_DIR / txt_file.name
        out_txt.write_text(cleaned, encoding="utf-8")
        logger.info(f"  Saved: {out_txt.name} ({original_len:,} -> {cleaned_len:,} chars, {reduction:.1f}% reduction)")

        # Copy and update metadata JSON
        json_src = EXTRACTED_DIR / txt_file.with_suffix(".json").name
        if json_src.exists():
            meta = json.loads(json_src.read_text(encoding="utf-8"))
            meta["cleaned_char_count"]      = cleaned_len
            meta["cleaning_reduction_pct"]  = round(reduction, 2)
            (PROCESSED_DIR / json_src.name).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        report.append({
            "file":           txt_file.name,
            "original_chars": original_len,
            "cleaned_chars":  cleaned_len,
            "reduction_pct":  round(reduction, 2),
        })

    report_path = PROCESSED_DIR / "_cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(f"\n{'='*50}")
    logger.info(f"Cleaning complete. {len(report)}/{len(txt_files)} files processed.")
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
