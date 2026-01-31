from typing import Any

import anthropic
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")


def calculate(operation: str, a: float, b: float) -> float:
    if operation == "divide" and b == 0:
        raise ValueError("Dělení nulou není povoleno")

    operations = {
        "add": lambda: a + b,
        "subtract": lambda: a - b,
        "multiply": lambda: a * b,
        "divide": lambda: a / b,
    }

    if operation not in operations:
        raise ValueError(f"Neznámá operace: {operation}")

    return operations[operation]()


def create_tools():
    return [
        {
            "name": "calculator",
            "description": "Provádí základní matematické operace",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        }
    ]


def call_api(client: anthropic.Anthropic, tools: list, messages: list):
    try:
        return client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )
    except anthropic.APIError as e:
        logger.error(f"API chyba: {e}")
        raise


def process_tool_calls(tool_uses: list) -> list:
    tool_results = []

    for tool_use in tool_uses:
        logger.info(f"Nástroj: {tool_use.name}")
        logger.info(f"  Vstup: {json.dumps(tool_use.input, ensure_ascii=False)}")

        try:
            result = calculate(
                tool_use.input["operation"],
                tool_use.input["a"],
                tool_use.input["b"],
            )
            logger.info(f"  Výsledek: {result}")
            content = str(result)
        except ValueError as e:
            content = f"Chyba: {e}"
            logger.error(f"  {content}")

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": content,
        })

    return tool_results


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY není nastavený")
        return

    client = anthropic.Anthropic()
    tools = create_tools()
    user_message = "Kolik je 25 * 17 + 42?"
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

    logger.info(f"Uživatel: {user_message}")
    logger.info("-" * 40)

    response = call_api(client, tools, messages)

    while response.stop_reason == "tool_use":
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})
        tool_results = process_tool_calls(tool_uses)
        messages.append({"role": "user", "content": tool_results})
        response = call_api(client, tools, messages)

    logger.info("-" * 40)
    final_text = next(
        (block.text for block in response.content if hasattr(block, "text")), ""
    )
    logger.info(f"Odpověď: {final_text}")


if __name__ == "__main__":
    main()
