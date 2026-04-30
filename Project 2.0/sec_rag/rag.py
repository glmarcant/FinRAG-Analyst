"""
RAG (Retrieval-Augmented Generation) module using Gemini API.

This module handles:
- Integrating retrieval and generation components
- Using Gemini API for natural language responses
- Generating answers based on retrieved filing context
"""

import time
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from . import config
from .retrieve import search_filings, query_financials, get_schema_description

console = Console()


def create_function_declarations() -> List[Dict]:
    """Create function declarations for Gemini function calling."""

    # Get the schema description for the financials table
    schema_description = get_schema_description()

    search_filings_decl = {
        "name": "search_filings",
        "description": (
            "Búsqueda semántica en el texto de filings 10-K de SEC EDGAR. "
            "Usalo para preguntas cualitativas sobre estrategia, riesgos, "
            "descripción del negocio, o discusión gerencial. Devuelve "
            "fragmentos de texto con su origen (empresa, fecha, sección)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Texto de búsqueda semántica sobre el contenido de los filings"
                },
                "ticker": {
                    "type": "string",
                    "description": "Símbolo de ticker opcional para filtrar (ej. AAPL, MSFT, JPM, TSLA)"
                },
                "section": {
                    "type": "string",
                    "enum": ["business", "risk_factors", "management_discussion"],
                    "description": "Sección específica del filing a buscar (opcional)"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Número de resultados a devolver (default 5)"
                }
            },
            "required": ["query"]
        }
    }

    query_financials_decl = {
        "name": "query_financials",
        "description": f"""Ejecuta SQL read-only contra una tabla DuckDB con datos financieros XBRL.
        Usalo para preguntas cuantitativas sobre revenue, net income, assets, etc.

        {schema_description}""",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "Query SQL SELECT-only para consultar los datos financieros"
                }
            },
            "required": ["sql"]
        }
    }

    return [search_filings_decl, query_financials_decl]


def setup_gemini_client() -> genai.Client:
    """Initialize Gemini client with API key."""
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    return client


def create_generation_config() -> types.GenerateContentConfig:
    """Create generation configuration with tools and system instruction."""

    function_declarations = create_function_declarations()

    tool = types.Tool(function_declarations=function_declarations)

    config_genai = types.GenerateContentConfig(
        tools=[tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        system_instruction=(
            "Sos un analista financiero experto. Usá las funciones disponibles "
            "para responder preguntas sobre filings de SEC y datos financieros. "
            "SIEMPRE citá la fuente: para texto, indicá ticker, fecha del filing "
            "y sección. Para números, indicá 'según los datos XBRL' y el período. "
            "Si una pregunta requiere ambos tipos de información, llamá ambas funciones. "
            "Sé preciso y profesional en tus respuestas."
        )
    )

    return config_genai


def execute_function_call(fc: Any) -> Dict[str, Any]:
    """Execute a function call and return the result."""
    try:
        if fc.name == "search_filings":
            args = dict(fc.args)
            result = search_filings(**args)
            return {"result": result}

        elif fc.name == "query_financials":
            args = dict(fc.args)
            result = query_financials(args["sql"])
            return {"result": result}

        else:
            return {"error": f"Unknown function: {fc.name}"}

    except Exception as e:
        return {"error": str(e)}


def log_function_call(fc: Any):
    """Log function call with rich panel."""
    args_str = ", ".join([f"{k}={repr(v)}" for k, v in fc.args.items()])
    console.print(
        Panel(
            f"[bold]Function Call:[/bold] {fc.name}\n[dim]Args:[/dim] {args_str}",
            style="yellow",
            title="🔧 Calling Function"
        )
    )


def log_function_result(fc_name: str, result: Dict[str, Any]):
    """Log function result with rich panel."""
    if "error" in result:
        console.print(
            Panel(
                f"[bold red]Error in {fc_name}:[/bold red]\n{result['error']}",
                style="red",
                title="❌ Function Error"
            )
        )
    else:
        # Summarize the result
        data = result.get("result", [])
        if isinstance(data, list):
            count = len(data)
            if count > 0:
                first_item = str(data[0])
                preview = first_item[:100] + "..." if len(first_item) > 100 else first_item
                summary = f"Found {count} results.\nFirst result: {preview}"
            else:
                summary = "No results found."
        else:
            summary = str(data)[:200] + "..." if len(str(data)) > 200 else str(data)

        console.print(
            Panel(
                f"[bold]Function:[/bold] {fc_name}\n{summary}",
                style="blue",
                title="✅ Function Result"
            )
        )


def handle_rate_limit(attempt: int) -> bool:
    """Handle rate limiting with exponential backoff."""
    if attempt >= 3:
        return False

    wait_time = 60 * (2 ** attempt)  # 60s, 120s, 240s
    console.print(
        Panel(
            f"Rate limit hit. Waiting {wait_time} seconds... (Attempt {attempt + 1}/3)",
            style="orange1",
            title="⏱️ Rate Limit"
        )
    )
    time.sleep(wait_time)
    return True


def answer(question: str) -> str:
    """
    Answer a question using RAG with manual function calling.

    Args:
        question: User question about SEC filings or financial data

    Returns:
        Generated answer with citations
    """
    client = setup_gemini_client()
    config_genai = create_generation_config()

    # Initialize conversation history
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)]
        )
    ]

    # Function calling loop (max 5 iterations)
    for iteration in range(5):
        console.print(f"\n[dim]--- Iteration {iteration + 1} ---[/dim]")

        # Try with rate limit handling
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=contents,
                    config=config_genai
                )
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    if not handle_rate_limit(attempt):
                        return f"Rate limit exceeded after 3 attempts: {e}"
                    continue
                else:
                    return f"Error calling Gemini API: {e}"

        # Check if we have function calls
        if not response.function_calls:
            # Final response
            final_answer = response.text if response.text else "No response generated."
            console.print(
                Panel(
                    final_answer,
                    style="green",
                    title="🎯 Final Answer"
                )
            )
            return final_answer

        # Add model response to conversation history
        if response.candidates and len(response.candidates) > 0:
            contents.append(response.candidates[0].content)

        # Process function calls
        function_response_parts = []

        for fc in response.function_calls:
            # Log the function call
            log_function_call(fc)

            # Execute the function
            result = execute_function_call(fc)

            # Log the result
            log_function_result(fc.name, result)

            # Create function response part
            function_response_part = types.Part.from_function_response(
                name=fc.name,
                response=result
            )
            function_response_parts.append(function_response_part)

        # Add function responses to conversation history
        contents.append(
            types.Content(
                role="user",
                parts=function_response_parts
            )
        )

    return "Maximum function calling iterations reached. Please try a simpler question."


def main():
    """Interactive RAG system."""
    console.print(
        Panel(
            "[bold green]SEC RAG System[/bold green]\n\n"
            "Ask questions about SEC filings and financial data.\n"
            "Available companies: AAPL, JPM, TSLA\n\n"
            "Examples:\n"
            "• 'What are Apple's main risks?'\n"
            "• 'What was Tesla's revenue in 2025?'\n"
            "• 'Compare JPM and Tesla profitability'\n\n"
            "Type 'exit' to quit.",
            style="blue"
        )
    )

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]Question")

            if question.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not question.strip():
                console.print("[red]Please enter a question.[/red]")
                continue

            console.print(f"\n[bold]Processing question:[/bold] {question}")

            # Get answer
            answer_text = answer(question)

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()