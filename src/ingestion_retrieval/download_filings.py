"""
FinRAG Analyst — SEC Filing Downloader
Downloads 10-K and 10-Q filings for Apple, Tesla, and JPMorgan
from SEC EDGAR into organized folders.

Usage:
    pip install sec-edgar-downloader
    python download_filings.py
"""

from sec_edgar_downloader import Downloader
import os

# ── Configuration ────────────────────────────────────────────────
# Replace with your name and university email (required by SEC EDGAR)
YOUR_NAME = "Giulia Marcantonio"
YOUR_EMAIL = "your_email@unc.edu"

# Companies: (ticker, full name)
COMPANIES = [
    ("AAPL", "Apple"),
    ("TSLA", "Tesla"),
    ("JPM",  "JPMorgan"),
]

# How many filings to download per type
NUM_10K = 2   # last 2 annual reports (~2 years of data)
NUM_10Q = 4   # last 4 quarterly reports (~1 year of data)

# Output folder
OUTPUT_DIR = "./sec_filings"
# ─────────────────────────────────────────────────────────────────


def download_all():
    dl = Downloader(YOUR_NAME, YOUR_EMAIL, OUTPUT_DIR)

    for ticker, name in COMPANIES:
        print(f"\n{'='*50}")
        print(f"  Downloading filings for {name} ({ticker})")
        print(f"{'='*50}")

        # Download 10-K (annual reports)
        print(f"  → Fetching {NUM_10K} x 10-K filings...")
        dl.get("10-K", ticker, limit=NUM_10K)

        # Download 10-Q (quarterly reports)
        print(f"  → Fetching {NUM_10Q} x 10-Q filings...")
        dl.get("10-Q", ticker, limit=NUM_10Q)

        print(f"  ✓ {name} done.")

    print(f"\n{'='*50}")
    print(f"  All filings saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*50}\n")
    summarize(OUTPUT_DIR)


def summarize(base_dir):
    """Print a summary of what was downloaded."""
    print("  Downloaded file summary:\n")
    for ticker, name in COMPANIES:
        print(f"  [{name} — {ticker}]")
        for form in ["10-K", "10-Q"]:
            folder = os.path.join(base_dir, "sec-edgar-filings", ticker, form)
            if os.path.exists(folder):
                filings = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
                print(f"    {form}: {len(filings)} filing(s) downloaded")
            else:
                print(f"    {form}: folder not found (check ticker/network)")
        print()


if __name__ == "__main__":
    download_all()
