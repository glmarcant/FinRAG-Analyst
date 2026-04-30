"""
Financial data extraction and processing module.

This module handles:
- Extracting structured financial data from 10-K filings
- Parsing financial statements and key metrics
- Storing financial data in DuckDB for analysis
"""

import pandas as pd
import duckdb
from typing import List, Dict, Optional

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


def setup_duckdb_connection():
    """Create DuckDB connection and setup financials table."""
    conn = duckdb.connect(str(config.DUCKDB_PATH))

    # Create financials table if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS financials (
            ticker VARCHAR,
            cik VARCHAR,
            statement_type VARCHAR,
            line_item VARCHAR,
            period_end DATE,
            value DOUBLE
        )
    """)

    return conn


def fetch_financials_for_ticker(ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Fetch financial statements for a ticker."""
    try:
        company = Company(ticker)
        financials = company.get_financials()

        # Get the three financial statements
        income_df = financials.income_statement().to_dataframe()
        balance_df = financials.balance_sheet().to_dataframe()
        cashflow_df = financials.cash_flow_statement().to_dataframe()

        return {
            'income_statement': income_df,
            'balance_sheet': balance_df,
            'cash_flow': cashflow_df,
            'cik': str(company.cik)
        }

    except Exception as e:
        console.print(f"[red]Error fetching financials for {ticker}: {e}[/red]")
        return None


def transform_to_long_format(df: pd.DataFrame, ticker: str, cik: str, statement_type: str) -> pd.DataFrame:
    """Transform financial DataFrame to long format."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['ticker', 'cik', 'statement_type', 'line_item', 'period_end', 'value'])

    # Find date columns (they contain "(FY)" pattern)
    date_columns = [col for col in df.columns if '(FY)' in str(col)]

    if not date_columns:
        console.print(f"[yellow]No date columns found for {ticker} {statement_type}[/yellow]")
        return pd.DataFrame(columns=['ticker', 'cik', 'statement_type', 'line_item', 'period_end', 'value'])

    # Create the long format DataFrame
    records = []

    for _, row in df.iterrows():
        # Use the 'label' column as the line item (more readable than concept)
        line_item = row.get('label', row.get('concept', 'Unknown'))

        for date_col in date_columns:
            try:
                # Extract date from column name (e.g., "2025-09-27 (FY)" -> "2025-09-27")
                period_date = date_col.split(' ')[0]
                value = row[date_col]

                # Skip null values
                if pd.isna(value):
                    continue

                # Convert value to float if possible
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue

                records.append({
                    'ticker': ticker,
                    'cik': cik,
                    'statement_type': statement_type,
                    'line_item': str(line_item),
                    'period_end': period_date,
                    'value': value
                })
            except Exception as e:
                continue

    if not records:
        console.print(f"[yellow]No valid financial data found for {ticker} {statement_type}[/yellow]")
        return pd.DataFrame(columns=['ticker', 'cik', 'statement_type', 'line_item', 'period_end', 'value'])

    long_df = pd.DataFrame(records)

    # Convert period_end to datetime
    try:
        long_df['period_end'] = pd.to_datetime(long_df['period_end'])
    except Exception as e:
        console.print(f"[yellow]Warning: Could not parse dates for {ticker} {statement_type}: {e}[/yellow]")
        # Keep as string if datetime conversion fails

    return long_df


def delete_existing_ticker_data(conn: duckdb.DuckDBPyConnection, ticker: str):
    """Delete existing data for a ticker to ensure idempotency."""
    conn.execute("DELETE FROM financials WHERE ticker = ?", [ticker])


def insert_financial_data(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Insert financial data into DuckDB."""
    if not df.empty:
        conn.execute("INSERT INTO financials SELECT * FROM df")


def process_ticker_financials(ticker: str, progress: Progress, task_id: TaskID) -> bool:
    """Process financial data for a single ticker."""
    try:
        # Fetch financials
        financials_data = fetch_financials_for_ticker(ticker)
        if not financials_data:
            return False

        cik = financials_data['cik']
        all_dfs = []

        # Transform each statement to long format
        for stmt_type, df in financials_data.items():
            if stmt_type == 'cik':
                continue

            long_df = transform_to_long_format(df, ticker, cik, stmt_type)
            if not long_df.empty:
                all_dfs.append(long_df)

        if not all_dfs:
            console.print(f"[yellow]No financial data found for {ticker}[/yellow]")
            return False

        # Concatenate all DataFrames
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Setup connection and insert data
        conn = setup_duckdb_connection()

        # Delete existing data for this ticker
        delete_existing_ticker_data(conn, ticker)

        # Insert new data
        insert_financial_data(conn, combined_df)

        conn.close()
        progress.advance(task_id)

        console.print(f"[green]✅ Processed {len(combined_df)} records for {ticker}[/green]")
        return True

    except Exception as e:
        console.print(f"[red]Error processing {ticker}: {e}[/red]")
        return False


def create_validation_table(conn: duckdb.DuckDBPyConnection) -> Table:
    """Create validation summary table."""
    query = """
        SELECT
            ticker,
            COUNT(*) AS rows,
            MIN(period_end) AS min_period,
            MAX(period_end) AS max_period
        FROM financials
        GROUP BY ticker
        ORDER BY ticker
    """

    results = conn.execute(query).fetchall()

    table = Table(title="Financial Data Summary")
    table.add_column("Ticker", style="cyan")
    table.add_column("Total Rows", justify="right")
    table.add_column("Min Period", style="green")
    table.add_column("Max Period", style="green")

    for row in results:
        ticker, rows, min_period, max_period = row
        min_str = str(min_period) if min_period else "N/A"
        max_str = str(max_period) if max_period else "N/A"
        table.add_row(ticker, str(rows), min_str, max_str)

    return table


def create_revenue_table(conn: duckdb.DuckDBPyConnection) -> Table:
    """Create revenue summary table showing latest revenue for each ticker."""
    query = """
        SELECT
            ticker,
            line_item,
            period_end,
            value
        FROM financials
        WHERE (line_item LIKE '%Revenue%' OR line_item LIKE '%Net Sales%' OR line_item LIKE '%Total Revenue%')
        AND period_end = (
            SELECT MAX(period_end)
            FROM financials f2
            WHERE f2.ticker = financials.ticker
            AND (f2.line_item LIKE '%Revenue%' OR f2.line_item LIKE '%Net Sales%' OR f2.line_item LIKE '%Total Revenue%')
        )
        ORDER BY ticker, value DESC
    """

    results = conn.execute(query).fetchall()

    table = Table(title="Latest Revenue by Ticker")
    table.add_column("Ticker", style="cyan")
    table.add_column("Line Item", style="blue")
    table.add_column("Period End", style="green")
    table.add_column("Value ($M)", justify="right", style="bold")

    for row in results:
        ticker, line_item, period_end, value = row
        period_str = str(period_end) if period_end else "N/A"
        value_str = f"{value/1_000_000:,.0f}" if value else "N/A"
        table.add_row(ticker, line_item[:50], period_str, value_str)

    return table


def main():
    """Main function to extract and store financial data."""
    console.print(f"[bold blue]Starting financial data extraction...[/bold blue]")
    console.print(f"Tickers: {config.TICKERS}")

    # Setup edgar identity
    setup_edgar_identity()

    # Process each ticker
    with Progress() as progress:
        task_id = progress.add_task(
            f"Processing {len(config.TICKERS)} tickers...",
            total=len(config.TICKERS)
        )

        successful_tickers = []
        for ticker in config.TICKERS:
            console.print(f"[bold]Processing {ticker}...[/bold]")
            if process_ticker_financials(ticker, progress, task_id):
                successful_tickers.append(ticker)

    # Display validation tables
    if successful_tickers:
        conn = setup_duckdb_connection()

        console.print(f"\n[bold green]Completed! Processed {len(successful_tickers)} tickers successfully.[/bold green]")

        # Show validation table
        validation_table = create_validation_table(conn)
        console.print(validation_table)

        # Show revenue table
        revenue_table = create_revenue_table(conn)
        console.print(revenue_table)

        conn.close()
    else:
        console.print(f"[bold red]No tickers processed successfully.[/bold red]")


if __name__ == "__main__":
    main()