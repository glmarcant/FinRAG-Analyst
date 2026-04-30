"""
RAG (Retrieval-Augmented Generation) module using OpenAI API.

This module handles:
- Integrating retrieval and generation components
- Using OpenAI API for natural language responses with function calling
- Generating answers based on retrieved filing context
"""

import json
import datetime
from typing import List, Dict, Any, Optional

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from . import config
from .retrieve import search_filings, query_financials, get_schema_description

console = Console()


def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_function_definitions() -> List[Dict]:
    """Create function definitions for OpenAI function calling."""

    # Get the schema description for the financials table
    schema_description = get_schema_description()

    search_filings_func = {
        "type": "function",
        "function": {
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
    }

    query_financials_func = {
        "type": "function",
        "function": {
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
    }

    return [search_filings_func, query_financials_func]


def setup_openai_client() -> OpenAI:
    """Initialize OpenAI client with API key."""
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    return client


def execute_function_call(function_name: str, function_args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a function call and return the result."""
    try:
        if function_name == "search_filings":
            result = search_filings(**function_args)
            return {"result": result}

        elif function_name == "query_financials":
            result = query_financials(function_args["sql"])
            return {"result": result}

        else:
            return {"error": f"Unknown function: {function_name}"}

    except Exception as e:
        return {"error": str(e)}


def log_function_call(function_name: str, function_args: Dict[str, Any]):
    """Log function call with rich panel."""
    args_str = ", ".join([f"{k}={repr(v)}" for k, v in function_args.items()])
    console.print(
        Panel(
            f"[bold]Function Call:[/bold] {function_name}\n[dim]Args:[/dim] {args_str}",
            style="yellow",
            title="🔧 Calling Function"
        )
    )


def log_function_result(function_name: str, result: Dict[str, Any]):
    """Log function result with rich panel."""
    if "error" in result:
        console.print(
            Panel(
                f"[bold red]Error in {function_name}:[/bold red]\n{result['error']}",
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
                f"[bold]Function:[/bold] {function_name}\n{summary}",
                style="blue",
                title="✅ Function Result"
            )
        )


def answer(question: str) -> str:
    """
    Answer a question using RAG with OpenAI function calling.

    Args:
        question: User question about SEC filings or financial data

    Returns:
        Generated answer with citations
    """
    client = setup_openai_client()
    tools = create_function_definitions()

    # Initialize conversation history
    messages = [
        {
            "role": "system",
            "content": (
                "Sos un analista financiero experto. Usá las funciones disponibles "
                "para responder preguntas sobre filings de SEC y datos financieros. "
                "SIEMPRE citá la fuente: para texto, indicá ticker, fecha del filing "
                "y sección. Para números, indicá 'según los datos XBRL' y el período. "
                "Si una pregunta requiere ambos tipos de información, llamá ambas funciones. "
                "Sé preciso y profesional en tus respuestas."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # Function calling loop (max 5 iterations)
    for iteration in range(5):
        console.print(f"\n[dim]--- Iteration {iteration + 1} ---[/dim]")

        try:
            # Call OpenAI with function calling
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            # Add assistant's message to conversation
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls
            })

            # Check if there are tool calls to execute
            if message.tool_calls:
                # Execute each tool call
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # Log the function call
                    log_function_call(function_name, function_args)

                    # Execute the function
                    function_result = execute_function_call(function_name, function_args)

                    # Log the result
                    log_function_result(function_name, function_result)

                    # Add function result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_result, default=json_serializer)
                    })

                # Continue the conversation loop
                continue

            else:
                # No more function calls - return final answer
                final_answer = message.content if message.content else "No response generated."
                console.print(
                    Panel(
                        final_answer,
                        style="green",
                        title="🎯 Final Answer"
                    )
                )
                return final_answer

        except Exception as e:
            return f"Error calling OpenAI API: {e}"

    return "Maximum function calling iterations reached. Please try a simpler question."


def main():
    """Interactive RAG system using OpenAI."""
    console.print(
        Panel(
            "[bold green]SEC RAG System (OpenAI)[/bold green]\n\n"
            "Ask questions about SEC filings and financial data.\n"
            "Available companies: AAPL, JPM, TSLA\n\n"
            "Examples:\n"
            "• 'What are Apple's main risks?'\n"
            "• 'What was Tesla's revenue in 2024?'\n"
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