"""
Vector indexing module for RAG system.

This module handles:
- Creating embeddings for filing text chunks using sentence-transformers
- Building and managing ChromaDB vector index
- Indexing document sections for semantic search
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from . import config

console = Console()


def initialize_embedding_model() -> SentenceTransformer:
    """Initialize the sentence transformer model."""
    console.print(f"[blue]Loading embedding model: {config.EMBEDDING_MODEL_NAME}[/blue]")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return model


def initialize_chromadb() -> chromadb.Collection:
    """Initialize ChromaDB with persistent storage."""
    console.print(f"[blue]Initializing ChromaDB at: {config.CHROMA_DIR}[/blue]")
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="sec_filings",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def initialize_text_splitter() -> RecursiveCharacterTextSplitter:
    """Initialize text splitter for chunking."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )
    return text_splitter


def get_filing_json_files() -> List[Path]:
    """Get all JSON filing files."""
    json_files = list(config.FILINGS_JSON_DIR.glob("*.json"))
    console.print(f"[green]Found {len(json_files)} filing JSON files[/green]")
    return json_files


def load_filing_json(file_path: Path) -> Dict:
    """Load a filing JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_chunks_from_section(text: str, text_splitter: RecursiveCharacterTextSplitter) -> List[str]:
    """Create chunks from a text section."""
    if not text or not text.strip():
        return []

    chunks = text_splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def create_document_metadata(filing_data: Dict, section: str, chunk_index: int) -> Dict:
    """Create metadata for a document chunk."""
    return {
        "ticker": filing_data["ticker"],
        "cik": filing_data["cik"],
        "form_type": filing_data["form_type"],
        "filing_date": filing_data["filing_date"],
        "period_of_report": filing_data["period_of_report"],
        "accession_number": filing_data["accession_number"],
        "section": section,
        "chunk_index": chunk_index
    }


def create_document_id(ticker: str, accession_number: str, section: str, chunk_index: int) -> str:
    """Create unique document ID."""
    # Remove dashes from accession number for cleaner ID
    clean_accession = accession_number.replace('-', '')
    return f"{ticker}_{clean_accession}_{section}_{chunk_index}"


def delete_existing_filing(collection: chromadb.Collection, accession_number: str):
    """Delete existing documents for a filing to ensure idempotency."""
    try:
        # ChromaDB delete with where clause
        collection.delete(where={"accession_number": accession_number})
        console.print(f"[yellow]Deleted existing documents for filing {accession_number}[/yellow]")
    except Exception as e:
        # If no documents exist, this will throw an error - that's okay
        console.print(f"[dim]No existing documents to delete for {accession_number}[/dim]")


def process_filing(
    file_path: Path,
    text_splitter: RecursiveCharacterTextSplitter,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    progress: Progress,
    task_id: TaskID
) -> Tuple[int, Dict[str, int]]:
    """Process a single filing file and return chunk counts."""
    try:
        # Load filing data
        filing_data = load_filing_json(file_path)
        accession_number = filing_data["accession_number"]
        ticker = filing_data["ticker"]

        console.print(f"[bold]Processing {ticker} - {accession_number}[/bold]")

        # Delete existing documents for this filing
        delete_existing_filing(collection, accession_number)

        all_documents = []
        all_metadatas = []
        all_ids = []
        section_counts = {"business": 0, "risk_factors": 0, "management_discussion": 0}

        # Process each section
        for section_name, section_text in filing_data["sections"].items():
            if not section_text or not section_text.strip():
                continue

            # Create chunks for this section
            chunks = create_chunks_from_section(section_text, text_splitter)

            for chunk_index, chunk in enumerate(chunks):
                # Create metadata
                metadata = create_document_metadata(filing_data, section_name, chunk_index)

                # Create unique ID
                doc_id = create_document_id(ticker, accession_number, section_name, chunk_index)

                all_documents.append(chunk)
                all_metadatas.append(metadata)
                all_ids.append(doc_id)
                section_counts[section_name] += 1

        if not all_documents:
            console.print(f"[yellow]No text chunks found for {ticker} - {accession_number}[/yellow]")
            progress.advance(task_id)
            return 0, section_counts

        # Generate embeddings in batches
        batch_size = 32
        for i in range(0, len(all_documents), batch_size):
            batch_docs = all_documents[i:i + batch_size]
            batch_metadata = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]

            # Generate embeddings
            embeddings = model.encode(batch_docs).tolist()

            # Add to ChromaDB
            collection.add(
                documents=batch_docs,
                metadatas=batch_metadata,
                ids=batch_ids,
                embeddings=embeddings
            )

        progress.advance(task_id)
        console.print(f"[green]✅ Indexed {len(all_documents)} chunks for {ticker}[/green]")
        return len(all_documents), section_counts

    except Exception as e:
        console.print(f"[red]Error processing {file_path}: {e}[/red]")
        progress.advance(task_id)
        return 0, {"business": 0, "risk_factors": 0, "management_discussion": 0}


def create_summary_table(total_filings: int, total_chunks: int, section_summary: Dict[str, int]) -> Table:
    """Create summary table showing indexing results."""
    table = Table(title="Indexing Summary")

    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="bold")

    table.add_row("Total Filings", str(total_filings))
    table.add_row("Total Chunks", str(total_chunks))
    table.add_row("", "")  # Separator
    table.add_row("Business Chunks", str(section_summary["business"]))
    table.add_row("Risk Factors Chunks", str(section_summary["risk_factors"]))
    table.add_row("MD&A Chunks", str(section_summary["management_discussion"]))

    return table


def test_search_query(collection: chromadb.Collection, query_text: str = "supply chain risk") -> Table:
    """Test search with a sample query."""
    console.print(f"[blue]Testing search with query: '{query_text}'[/blue]")

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=3
        )

        table = Table(title=f"Top 3 Results for: '{query_text}'")
        table.add_column("Ticker", style="cyan")
        table.add_column("Section", style="blue")
        table.add_column("Filing Date", style="green")
        table.add_column("Text Preview", style="dim", max_width=50)

        if results['metadatas'] and results['metadatas'][0]:
            for i in range(len(results['metadatas'][0])):
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i] if results['documents'] else "N/A"

                # Truncate document preview
                doc_preview = document[:100] + "..." if len(document) > 100 else document

                table.add_row(
                    metadata.get('ticker', 'N/A'),
                    metadata.get('section', 'N/A'),
                    metadata.get('filing_date', 'N/A'),
                    doc_preview
                )
        else:
            table.add_row("No results found", "", "", "")

        return table

    except Exception as e:
        console.print(f"[red]Error testing search: {e}[/red]")
        table = Table(title=f"Search Test Failed")
        table.add_column("Error", style="red")
        table.add_row(str(e))
        return table


def main():
    """Main function to index SEC filings."""
    console.print(f"[bold blue]Starting SEC filing indexing...[/bold blue]")

    # Initialize components
    model = initialize_embedding_model()
    collection = initialize_chromadb()
    text_splitter = initialize_text_splitter()

    # Get filing files
    json_files = get_filing_json_files()

    if not json_files:
        console.print(f"[red]No JSON files found in {config.FILINGS_JSON_DIR}[/red]")
        return

    # Process filings
    total_chunks = 0
    total_filings = 0
    section_summary = {"business": 0, "risk_factors": 0, "management_discussion": 0}

    with Progress() as progress:
        task_id = progress.add_task(
            f"Processing {len(json_files)} filings...",
            total=len(json_files)
        )

        for json_file in json_files:
            chunk_count, section_counts = process_filing(
                json_file, text_splitter, collection, model, progress, task_id
            )

            if chunk_count > 0:
                total_filings += 1
                total_chunks += chunk_count

                for section, count in section_counts.items():
                    section_summary[section] += count

    # Display results
    console.print(f"\n[bold green]Indexing completed![/bold green]")

    # Summary table
    summary_table = create_summary_table(total_filings, total_chunks, section_summary)
    console.print(summary_table)

    # Test search
    if total_chunks > 0:
        search_table = test_search_query(collection)
        console.print(search_table)
    else:
        console.print("[yellow]No documents indexed, skipping search test[/yellow]")


if __name__ == "__main__":
    main()