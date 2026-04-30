"""
Ingest module for downloading and processing 10-K filings from SEC EDGAR.

This module handles:
- Fetching 10-K filings for specified tickers using edgartools
- Downloading and parsing filing documents
- Storing raw filing data for further processing
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import edgar
from edgar import Company
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from . import config

console = Console()


def setup_edgar_identity():
    """Setup edgar tools with SEC required user agent."""
    edgar.set_identity(config.SEC_USER_AGENT)


def extract_tenk_sections(tenk) -> Dict[str, Optional[str]]:
    """Extract key sections from TenK object."""
    sections = {}

    try:
        sections["business"] = getattr(tenk, 'business', None)
    except Exception:
        sections["business"] = None

    try:
        sections["risk_factors"] = getattr(tenk, 'risk_factors', None)
    except Exception:
        sections["risk_factors"] = None

    try:
        sections["management_discussion"] = getattr(tenk, 'management_discussion', None)
    except Exception:
        sections["management_discussion"] = None

    return sections


def create_filing_json(filing, ticker: str, sections: Dict[str, Optional[str]]) -> Dict:
    """Create JSON structure for filing data."""
    # Handle date formatting - may be string or datetime object
    filing_date = filing.filing_date
    if hasattr(filing_date, 'strftime'):
        filing_date = filing_date.strftime('%Y-%m-%d')
    elif isinstance(filing_date, str):
        filing_date = filing_date
    else:
        filing_date = None

    period_date = filing.period_of_report
    if hasattr(period_date, 'strftime'):
        period_date = period_date.strftime('%Y-%m-%d')
    elif isinstance(period_date, str):
        period_date = period_date
    else:
        period_date = None

    return {
        "ticker": ticker,
        "cik": str(filing.cik),
        "form_type": "10-K",
        "filing_date": filing_date,
        "period_of_report": period_date,
        "accession_number": filing.accession_number,
        "sections": sections
    }


def get_filename(ticker: str, accession_number: str) -> str:
    """Generate filename for filing JSON."""
    clean_accession = accession_number.replace('-', '')
    return f"{ticker}_{clean_accession}.json"


def file_exists(ticker: str, accession_number: str) -> bool:
    """Check if filing file already exists."""
    filename = get_filename(ticker, accession_number)
    filepath = config.FILINGS_JSON_DIR / filename
    return filepath.exists()


def save_filing_json(filing_data: Dict, ticker: str, accession_number: str):
    """Save filing data as JSON file."""
    filename = get_filename(ticker, accession_number)
    filepath = config.FILINGS_JSON_DIR / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(filing_data, f, indent=2, ensure_ascii=False)


def get_filing_status(sections: Dict[str, Optional[str]]) -> str:
    """Determine filing status based on section content."""
    valid_sections = 0

    for section_content in sections.values():
        if section_content and len(section_content.strip()) > 500:
            valid_sections += 1

    if valid_sections == 3:
        return "✅"
    elif valid_sections > 0:
        return "⚠️"
    else:
        return "❌"


def get_section_char_count(content: Optional[str]) -> str:
    """Get character count for section content."""
    if not content:
        return "0"
    return str(len(content.strip()))


def process_ticker(ticker: str, progress: Progress, task_id: TaskID) -> List[Dict]:
    """Process all filings for a single ticker."""
    results = []

    try:
        company = Company(ticker)
        filings = company.get_filings(form="10-K")

        # Take last N filings
        recent_filings = filings[:config.NUM_FILINGS_PER_TICKER]

        for filing in recent_filings:
            try:
                # Check if already exists
                if file_exists(ticker, filing.accession_number):
                    console.print(f"[yellow]Skipping existing file for {ticker} {filing.accession_number}[/yellow]")
                    continue

                # Get TenK object
                tenk = filing.obj()
                if not tenk:
                    console.print(f"[red]Could not get TenK object for {ticker} {filing.accession_number}[/red]")
                    continue

                # Extract sections
                sections = extract_tenk_sections(tenk)

                # Create filing data
                filing_data = create_filing_json(filing, ticker, sections)

                # Save to file
                save_filing_json(filing_data, ticker, filing.accession_number)

                # Add to results
                result = {
                    "ticker": ticker,
                    "filing_date": filing_data["filing_date"],
                    "period": filing_data["period_of_report"],
                    "business_chars": get_section_char_count(sections["business"]),
                    "risk_chars": get_section_char_count(sections["risk_factors"]),
                    "mda_chars": get_section_char_count(sections["management_discussion"]),
                    "status": get_filing_status(sections)
                }
                results.append(result)

                progress.advance(task_id)

            except Exception as e:
                console.print(f"[red]Error processing filing {filing.accession_number} for {ticker}: {e}[/red]")
                continue

    except Exception as e:
        console.print(f"[red]Error processing ticker {ticker}: {e}[/red]")

    return results


def create_summary_table(all_results: List[Dict]):
    """Create and display summary table."""
    table = Table(title="Filing Download Summary")

    table.add_column("Ticker", style="cyan")
    table.add_column("Filing Date", style="green")
    table.add_column("Period", style="green")
    table.add_column("Business (chars)", justify="right")
    table.add_column("Risk Factors (chars)", justify="right")
    table.add_column("MD&A (chars)", justify="right")
    table.add_column("Status", justify="center")

    for result in all_results:
        table.add_row(
            result["ticker"],
            result["filing_date"],
            result["period"],
            result["business_chars"],
            result["risk_chars"],
            result["mda_chars"],
            result["status"]
        )

    console.print(table)


def main():
    """Main function to download and process 10-K filings."""
    console.print(f"[bold blue]Starting SEC EDGAR 10-K ingestion...[/bold blue]")
    console.print(f"Tickers: {config.TICKERS}")
    console.print(f"Filings per ticker: {config.NUM_FILINGS_PER_TICKER}")

    # Setup edgar identity
    setup_edgar_identity()

    # Calculate total filings
    total_filings = len(config.TICKERS) * config.NUM_FILINGS_PER_TICKER
    all_results = []

    with Progress() as progress:
        task_id = progress.add_task(
            f"Processing {len(config.TICKERS)} tickers...",
            total=total_filings
        )

        for ticker in config.TICKERS:
            console.print(f"[bold]Processing {ticker}...[/bold]")
            ticker_results = process_ticker(ticker, progress, task_id)
            all_results.extend(ticker_results)

    # Display summary
    console.print(f"\n[bold green]Completed! Downloaded {len(all_results)} filings.[/bold green]")
    create_summary_table(all_results)


if __name__ == "__main__":
    main()