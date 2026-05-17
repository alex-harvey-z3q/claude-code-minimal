from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Mapping

from .config import (
    AWS_REGION,
    BEDROCK_CHAT_MODEL_ID,
    BEDROCK_CONNECT_TIMEOUT_SECONDS,
    BEDROCK_READ_TIMEOUT_SECONDS,
    LLM_PROVIDER,
    MAX_TOKENS,
    TEMPERATURE,
)


@lru_cache(maxsize=1)
def get_bedrock_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        config=Config(
            connect_timeout=BEDROCK_CONNECT_TIMEOUT_SECONDS,
            read_timeout=BEDROCK_READ_TIMEOUT_SECONDS,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def invoke_claude(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    if LLM_PROVIDER == "fake":
        return _invoke_fake(system_prompt, user_prompt)
    if LLM_PROVIDER != "bedrock":
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

    try:
        from botocore.exceptions import ReadTimeoutError

        response = get_bedrock_client().converse(
            modelId=BEDROCK_CHAT_MODEL_ID,
            system=[{"text": system_prompt}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_prompt}],
                }
            ],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        )
    except ReadTimeoutError as exc:
        raise RuntimeError("Bedrock request timed out before the model returned a response.") from exc

    content = response["output"]["message"]["content"]
    text_parts = [part.get("text", "") for part in content if "text" in part]
    return "\n".join(part for part in text_parts if part).strip()


def invoke_claude_with_tools(
    system_prompt: str,
    user_prompt: str,
    *,
    tools: list[dict[str, Any]],
    handlers: Mapping[str, Callable[..., str]],
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    max_tool_rounds: int = 40,
) -> tuple[str, dict[str, Any]]:
    """Run a Bedrock Converse tool-use loop.

    Tool calls are executed by host-side handlers and returned as toolResult
    messages. The model never receives generated file contents unless it asks
    for them with a read tool, which is the core difference from the previous
    "emit every changed file in the prompt" protocol.
    """
    if LLM_PROVIDER == "fake":
        return _invoke_fake_with_tools(system_prompt, user_prompt, tools=tools, handlers=handlers)
    if LLM_PROVIDER != "bedrock":
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"text": user_prompt}],
        }
    ]
    tool_calls: list[dict[str, Any]] = []
    final_text_parts: list[str] = []

    for _ in range(max_tool_rounds):
        response = get_bedrock_client().converse(
            modelId=BEDROCK_CHAT_MODEL_ID,
            system=[{"text": system_prompt}],
            messages=messages,
            toolConfig={"tools": tools},
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        )

        output_message = response["output"]["message"]
        messages.append(output_message)

        content = output_message.get("content", [])
        tool_uses = [part["toolUse"] for part in content if "toolUse" in part]
        final_text_parts.extend(part.get("text", "") for part in content if "text" in part)

        if not tool_uses:
            return "\n".join(part for part in final_text_parts if part).strip(), {
                "messages": messages,
                "tool_calls": tool_calls,
            }

        result_content = []
        for tool_use in tool_uses:
            name = tool_use["name"]
            tool_use_id = tool_use["toolUseId"]
            arguments = tool_use.get("input") or {}

            try:
                if name not in handlers:
                    raise ValueError(f"Unknown tool: {name}")
                result_text = handlers[name](**arguments)
                status = "success"
            except Exception as exc:
                result_text = str(exc)
                status = "error"

            tool_calls.append(
                {
                    "name": name,
                    "input": arguments,
                    "status": status,
                    "result": result_text,
                }
            )
            result_content.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "status": status,
                        "content": [{"text": result_text}],
                    }
                }
            )

        messages.append({"role": "user", "content": result_content})

    raise RuntimeError(f"Claude did not finish after {max_tool_rounds} tool rounds.")


def _invoke_fake(system_prompt: str, user_prompt: str) -> str:
    del user_prompt
    if "Planner" in system_prompt:
        return (
            "1. Files\n"
            "- main.py\n"
            "- test_main.py\n\n"
            "2. Data structures / modules\n"
            "- A minimal Python module\n\n"
            "3. Conventions to follow\n"
            "- Use unittest\n\n"
            "4. Behaviour and flow\n"
            "- Create a small testable implementation\n\n"
            "5. Test strategy\n"
            "- Run unittest discovery"
        )
    return "Fake provider response."


def _invoke_fake_with_tools(
    system_prompt: str,
    user_prompt: str,
    *,
    tools: list[dict[str, Any]],
    handlers: Mapping[str, Callable[..., str]],
) -> tuple[str, dict[str, Any]]:
    del system_prompt, user_prompt, tools
    tool_calls = []

    if "write_file" in handlers:
        for path, content in {
            "main.py": "def hello() -> str:\n    return 'hello from fake provider'\n",
            "test_main.py": (
                "import unittest\n\n"
                "from main import hello\n\n\n"
                "class HelloTest(unittest.TestCase):\n"
                "    def test_hello(self):\n"
                "        self.assertEqual(hello(), 'hello from fake provider')\n"
            ),
        }.items():
            result = handlers["write_file"](path=path, content=content)
            tool_calls.append({"name": "write_file", "input": {"path": path}, "status": "success", "result": result})

        result = handlers["run_tests"]()
        tool_calls.append({"name": "run_tests", "input": {}, "status": "success", "result": result})
        return "Created a fake-provider smoke implementation and ran tests.", {"tool_calls": tool_calls}

    return "PASS: Fake provider review.", {"tool_calls": tool_calls}


def answer_with_evidence(
    question: str,
    evidence_items: list[Mapping[str, Any]],
) -> str:
    evidence_block = "\n\n".join(
        f"[{i+1}] {item['page']} — {item['section']}\n"
        f"URL: {item['url']}\n"
        f"Excerpt: {item['excerpt']}"
        for i, item in enumerate(evidence_items)
    )

    system_prompt = (
        "You are a careful assistant answering questions using ONLY the provided "
        "evidence excerpts from Wikipedia. If the evidence is insufficient, say "
        "you do not know. Always cite evidence items like [1], [2]."
    )

    user_prompt = f"Question: {question}\n\nEvidence:\n{evidence_block}"
    return invoke_claude(system_prompt, user_prompt)
