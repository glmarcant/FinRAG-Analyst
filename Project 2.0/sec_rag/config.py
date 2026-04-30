"""
Configuration module for SEC RAG system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from rich import print as rich_print

# Load environment variables
load_dotenv()

# SEC User Agent - MUST be real name and email as required by SEC
SEC_USER_AGENT = "Maria Emilia Granda  me.granda25@gmail.com"  # PLACEHOLDER - SEC requires real name+email

# Company tickers to analyze
TICKERS = ["AAPL", "JPM", "TSLA"]  # 3 empresas para empezar
NUM_FILINGS_PER_TICKER = 3  # últimos 3 10-Ks

# Directory structure
DATA_DIR = Path("data")
FILINGS_JSON_DIR = DATA_DIR / "filings"
CHROMA_DIR = DATA_DIR / "chroma_db"
DUCKDB_PATH = DATA_DIR / "financials.duckdb"

# Model configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL = "gemini-1.5-flash"  # tier gratuito
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# OpenAI configuration
OPENAI_MODEL = "gpt-4o-mini"  # Más barato y sin rate limits estrictos
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    FILINGS_JSON_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)


def validate_config():
    """Validate configuration and show warnings for placeholder values."""
    warnings = []

    if not GEMINI_API_KEY or GEMINI_API_KEY == "":
        warnings.append("GEMINI_API_KEY not set in environment")

    if not OPENAI_API_KEY or OPENAI_API_KEY == "":
        warnings.append("OPENAI_API_KEY not set in environment")

    if SEC_USER_AGENT == "Your Name your.email@domain.com":
        warnings.append("SEC_USER_AGENT is still placeholder - SEC requires real name+email")

    if warnings:
        rich_print("[yellow]⚠️  Configuration warnings:[/yellow]")
        for warning in warnings:
            rich_print(f"[yellow]  • {warning}[/yellow]")


# Initialize directories and validate config on import
ensure_dirs()
validate_config()