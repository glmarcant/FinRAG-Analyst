"""
Retrieval module for semantic search and context extraction.

This module handles:
- Semantic search over indexed filing content
- Retrieving relevant document chunks based on queries
- Context preparation for RAG generation
"""

from typing import List, Dict, Optional, Any
import re

import chromadb
import duckdb
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich.json import JSON

from . import config

console = Console()


def search_filings(
    query: str,
    ticker: Optional[str] = None,
    section: Optional[str] = None,
    form_type: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search filings using semantic vector search.

    Args:
        query: Search query text
        ticker: Filter by ticker (optional)
        section: Filter by section (optional)
        form_type: Filter by form type (optional)
        top_k: Number of results to return

    Returns:
        List of search results with metadata and scores
    """
    # Load embedding model
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_collection("sec_filings")

    # Build where clause for filtering - ChromaDB requires $eq operator for exact matches
    where_clause = None
    if ticker or section or form_type:
        where_clause = {"$and": []}
        if ticker:
            where_clause["$and"].append({"ticker": {"$eq": ticker}})
        if section:
            where_clause["$and"].append({"section": {"$eq": section}})
        if form_type:
            where_clause["$and"].append({"form_type": {"$eq": form_type}})

        # If only one condition, simplify
        if len(where_clause["$and"]) == 1:
            where_clause = where_clause["$and"][0]

    # Perform search
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_clause
    )

    # Format results
    formatted_results = []

    if results['metadatas'] and results['metadatas'][0]:
        for i in range(len(results['metadatas'][0])):
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]
            distance = results['distances'][0][i]

            # Convert distance to score (1 - distance for similarity score)
            score = 1.0 - distance

            formatted_results.append({
                "text": document,
                "ticker": metadata.get('ticker', 'N/A'),
                "filing_date": metadata.get('filing_date', 'N/A'),
                "section": metadata.get('section', 'N/A'),
                "accession_number": metadata.get('accession_number', 'N/A'),
                "score": round(score, 4)
            })

    return formatted_results


def query_financials(sql: str) -> List[Dict[str, Any]]:
    """
    Execute SQL query against financials database.

    Args:
        sql: SQL query string (must start with SELECT)

    Returns:
        List of query results as dictionaries

    Raises:
        ValueError: If query is not a safe SELECT statement
    """
    # Validate SQL query for security
    sql_clean = sql.strip()
    if not sql_clean.lower().startswith('select'):
        raise ValueError("Only SELECT queries are allowed. Found query starting with: " +
                        sql_clean.split()[0] if sql_clean else "empty query")

    # Additional validation - check for dangerous keywords
    dangerous_keywords = ['insert', 'update', 'delete', 'drop', 'create', 'alter',
                         'attach', 'detach', 'pragma', 'vacuum']
    sql_lower = sql_clean.lower()
    for keyword in dangerous_keywords:
        if keyword in sql_lower:
            raise ValueError(f"Potentially dangerous SQL keyword '{keyword}' detected in query")

    # Connect to DuckDB and execute query
    conn = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)

    try:
        # Limit results to 100 rows maximum
        limited_sql = f"SELECT * FROM ({sql_clean}) LIMIT 100"
        result = conn.execute(limited_sql).fetchall()
        columns = [desc[0] for desc in conn.description]

        # Convert to list of dictionaries
        formatted_results = []
        for row in result:
            row_dict = dict(zip(columns, row))
            formatted_results.append(row_dict)

        return formatted_results

    finally:
        conn.close()


def get_schema_description() -> str:
    """
    Get description of the financials table schema.

    Returns:
        String describing the database schema
    """
    schema_description = """
FINANCIALS TABLE SCHEMA:

Table: financials
Columns:
- ticker (VARCHAR): Company ticker symbol (e.g., 'AAPL', 'MSFT', 'TSLA')
- cik (VARCHAR): SEC Central Index Key
- statement_type (VARCHAR): Type of financial statement ('income_statement', 'balance_sheet', 'cash_flow')
- line_item (VARCHAR): Financial line item name
- period_end (DATE): End date of the reporting period
- value (DOUBLE): Numerical value for the line item

Common line_items include:
- Revenue related: 'Revenues', 'Net Sales', 'Total Revenue', 'Revenue From Contract With Customer'
- Income related: 'Net Income', 'Operating Income', 'Income Before Income Tax'
- Balance sheet: 'Total Assets', 'Total Liabilities', 'Stockholders Equity'
- Cash flow: 'Net Cash Flow From Operating Activities', 'Free Cash Flow'

Example queries:
- SELECT ticker, line_item, period_end, value FROM financials WHERE line_item ILIKE '%revenue%'
- SELECT ticker, AVG(value) FROM financials WHERE line_item LIKE '%Net Income%' GROUP BY ticker
- SELECT * FROM financials WHERE ticker = 'AAPL' AND statement_type = 'income_statement'
"""
    return schema_description.strip()


def display_search_results(query: str, results: List[Dict[str, Any]], title: str):
    """Display search results in a formatted table."""
    console.print(f"\n[bold blue]{title}[/bold blue]")
    console.print(f"Query: [cyan]'{query}'[/cyan]")

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    table = Table(title=f"Search Results ({len(results)} found)")
    table.add_column("Ticker", style="cyan")
    table.add_column("Section", style="blue")
    table.add_column("Filing Date", style="green")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Text Preview", style="dim", max_width=60)

    for result in results:
        # Truncate text preview
        text_preview = result["text"][:150] + "..." if len(result["text"]) > 150 else result["text"]

        table.add_row(
            result["ticker"],
            result["section"],
            result["filing_date"],
            str(result["score"]),
            text_preview
        )

    console.print(table)


def display_financial_results(sql: str, results: List[Dict[str, Any]], title: str):
    """Display financial query results."""
    console.print(f"\n[bold blue]{title}[/bold blue]")
    console.print(f"SQL: [cyan]{sql}[/cyan]")

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    # Display as JSON for better readability of financial data
    console.print(f"\n[bold]Results ({len(results)} rows):[/bold]")

    # Show first few results in table format if reasonable number of columns
    if results and len(results[0]) <= 6:
        # Create table for structured display
        table = Table(title=f"Financial Query Results")

        # Add columns
        for col in results[0].keys():
            table.add_column(str(col), style="cyan" if col == "ticker" else "white")

        # Add rows (limit to first 10 for readability)
        for i, row in enumerate(results[:10]):
            table.add_row(*[str(v) if v is not None else "NULL" for v in row.values()])

        console.print(table)

        if len(results) > 10:
            console.print(f"[dim]... and {len(results) - 10} more rows[/dim]")
    else:
        # Show as JSON for complex results
        console.print(JSON.from_data(results[:5]))  # Show first 5 results
        if len(results) > 5:
            console.print(f"[dim]... and {len(results) - 5} more rows[/dim]")


if __name__ == "__main__":
    console.print("[bold green]Testing SEC RAG Retrieval Functions[/bold green]\n")

    # Test 1: Search with filters
    console.print("=" * 80)
    try:
        results1 = search_filings(
            "artificial intelligence risks",
            ticker="AAPL",
            section="risk_factors"
        )
        display_search_results(
            "artificial intelligence risks",
            results1,
            "Test 1: Filtered Search (AAPL, risk_factors)"
        )
    except Exception as e:
        console.print(f"[red]Test 1 failed: {e}[/red]")

    # Test 2: General search
    console.print("\n" + "=" * 80)
    try:
        results2 = search_filings("competition", top_k=3)
        display_search_results(
            "competition",
            results2,
            "Test 2: General Search (competition)"
        )
    except Exception as e:
        console.print(f"[red]Test 2 failed: {e}[/red]")

    # Test 3: Financial query
    console.print("\n" + "=" * 80)
    try:
        sql_query = """
        SELECT ticker, line_item, period_end, value
        FROM financials
        WHERE line_item ILIKE '%revenue%'
        AND period_end >= '2022-01-01'
        ORDER BY period_end DESC LIMIT 10
        """
        results3 = query_financials(sql_query)
        display_financial_results(
            sql_query.strip(),
            results3,
            "Test 3: Financial Query (Revenue Data)"
        )
    except Exception as e:
        console.print(f"[red]Test 3 failed: {e}[/red]")

    # Test 4: Schema description
    console.print("\n" + "=" * 80)
    console.print("[bold blue]Database Schema Description[/bold blue]")
    schema = get_schema_description()
    console.print(schema)