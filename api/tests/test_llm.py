from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_USER", "user")
os.environ.setdefault("DB_PASSWORD", "password")
os.environ.setdefault("EMBED_DIM", "1536")

from api import llm


class LlmTest(unittest.TestCase):
    def test_fake_planner_response_is_structured(self) -> None:
        with patch.object(llm, "LLM_PROVIDER", "fake"):
            response = llm.invoke_claude("You are Planner", "task")

        self.assertIn("1. Files", response)
        self.assertIn("5. Test strategy", response)

    def test_fake_tool_loop_writes_files_and_runs_tests(self) -> None:
        written: dict[str, str] = {}
        ran_tests = False

        def write_file(path: str, content: str) -> str:
            written[path] = content
            return f"Wrote {path}"

        def run_tests() -> str:
            nonlocal ran_tests
            ran_tests = True
            return "Tests passed."

        response, trace = llm._invoke_fake_with_tools(
            "system",
            "user",
            tools=[],
            handlers={"write_file": write_file, "run_tests": run_tests},
        )

        self.assertIn("fake-provider smoke implementation", response)
        self.assertIn("main.py", written)
        self.assertIn("test_main.py", written)
        self.assertTrue(ran_tests)
        self.assertEqual([call["name"] for call in trace["tool_calls"]], ["write_file", "write_file", "run_tests"])

    def test_fake_tool_loop_without_write_file_returns_review_pass(self) -> None:
        response, trace = llm._invoke_fake_with_tools(
            "system",
            "user",
            tools=[],
            handlers={"run_tests": lambda: "Tests passed."},
        )

        self.assertEqual(response, "PASS: Fake provider review.")
        self.assertEqual(trace, {"tool_calls": []})

    def test_bedrock_tool_loop_executes_tool_results_then_returns_text(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0
                self.messages = []

            def converse(self, **kwargs):
                self.calls += 1
                self.messages.append(kwargs["messages"])
                if self.calls == 1:
                    return {
                        "output": {
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {
                                        "toolUse": {
                                            "name": "read_file",
                                            "toolUseId": "tool-1",
                                            "input": {"path": "main.py"},
                                        }
                                    }
                                ],
                            }
                        }
                    }
                return {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [{"text": "done"}],
                        }
                    }
                }

        client = FakeClient()

        with (
            patch.object(llm, "LLM_PROVIDER", "bedrock"),
            patch.object(llm, "BEDROCK_CHAT_MODEL_ID", "model"),
            patch.object(llm, "get_bedrock_client", return_value=client),
        ):
            response, trace = llm.invoke_claude_with_tools(
                "system",
                "user",
                tools=[],
                handlers={"read_file": lambda path: f"read {path}"},
                max_tool_rounds=2,
            )

        self.assertEqual(response, "done")
        self.assertEqual(trace["tool_calls"][0]["status"], "success")
        self.assertEqual(trace["tool_calls"][0]["result"], "read main.py")
        self.assertEqual(client.calls, 2)

    def test_bedrock_tool_loop_records_handler_errors(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def converse(self, **kwargs):
                del kwargs
                self.calls += 1
                if self.calls == 1:
                    return {
                        "output": {
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {
                                        "toolUse": {
                                            "name": "missing_tool",
                                            "toolUseId": "tool-1",
                                            "input": {},
                                        }
                                    }
                                ],
                            }
                        }
                    }
                return {"output": {"message": {"role": "assistant", "content": [{"text": "done"}]}}}

        with (
            patch.object(llm, "LLM_PROVIDER", "bedrock"),
            patch.object(llm, "get_bedrock_client", return_value=FakeClient()),
        ):
            _, trace = llm.invoke_claude_with_tools("system", "user", tools=[], handlers={}, max_tool_rounds=2)

        self.assertEqual(trace["tool_calls"][0]["status"], "error")
        self.assertIn("Unknown tool", trace["tool_calls"][0]["result"])

    def test_bedrock_tool_loop_rejects_repeated_malformed_tool_calls(self) -> None:
        class MalformedClient:
            def converse(self, **kwargs):
                del kwargs
                return {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"text": "trying"},
                                {
                                    "toolUse": {
                                        "name": "write_file",
                                        "toolUseId": "tool-1",
                                        "input": {"path": "tests/test_minesweeper.py"},
                                    }
                                },
                            ],
                        }
                    }
                }

        tools = [
            {
                "toolSpec": {
                    "name": "write_file",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        }
                    },
                }
            }
        ]

        with (
            patch.object(llm, "LLM_PROVIDER", "bedrock"),
            patch.object(llm, "BEDROCK_CHAT_MODEL_ID", "model"),
            patch.object(llm, "MALFORMED_TOOL_CALL_LIMIT", 2),
            patch.object(llm, "get_bedrock_client", return_value=MalformedClient()),
        ):
            with self.assertRaises(llm.ToolLoopError) as raised:
                llm.invoke_claude_with_tools(
                    "system",
                    "user",
                    tools=tools,
                    handlers={"write_file": lambda path, content: f"{path}:{content}"},
                    max_tool_rounds=5,
                )

        self.assertIn("repeated malformed write_file tool calls", str(raised.exception))
        self.assertEqual(len(raised.exception.tool_calls), 2)
        self.assertIn("Missing: content", raised.exception.tool_calls[-1]["result"])

    def test_bedrock_tool_loop_raises_after_max_rounds(self) -> None:
        class LoopingClient:
            def converse(self, **kwargs):
                del kwargs
                return {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"text": "thinking"},
                                {
                                    "toolUse": {
                                        "name": "read_file",
                                        "toolUseId": "tool-1",
                                        "input": {"path": "main.py"},
                                    }
                                },
                            ],
                        }
                    }
                }

        with (
            patch.object(llm, "LLM_PROVIDER", "bedrock"),
            patch.object(llm, "BEDROCK_CHAT_MODEL_ID", "model"),
            patch.object(llm, "get_bedrock_client", return_value=LoopingClient()),
        ):
            with self.assertRaises(llm.ToolLoopError) as raised:
                llm.invoke_claude_with_tools(
                    "system",
                    "user",
                    tools=[],
                    handlers={"read_file": lambda path: path},
                    max_tool_rounds=1,
                )

        self.assertEqual(raised.exception.partial_text, "thinking")
        self.assertEqual(raised.exception.tool_calls[0]["name"], "read_file")

    def test_tool_loop_error_summary_counts_tools(self) -> None:
        error = llm.ToolLoopError(
            "stuck",
            provider="bedrock",
            model_id="model",
            max_tool_rounds=1,
            tool_calls=[{"name": "a"}, {"name": "b"}, {"name": "a"}],
            partial_text="hello",
        )

        summary = error.summary()

        self.assertEqual(summary["tool_call_count"], 3)
        self.assertEqual(summary["tool_call_counts"], {"a": 2, "b": 1})
        self.assertEqual(summary["partial_text"], "hello")

    def test_invoke_claude_rejects_unknown_provider(self) -> None:
        with patch.object(llm, "LLM_PROVIDER", "other"):
            with self.assertRaisesRegex(ValueError, "Unsupported LLM_PROVIDER"):
                llm.invoke_claude("system", "user")

    def test_invoke_claude_with_tools_rejects_unknown_provider(self) -> None:
        with patch.object(llm, "LLM_PROVIDER", "other"):
            with self.assertRaisesRegex(ValueError, "Unsupported LLM_PROVIDER"):
                llm.invoke_claude_with_tools("system", "user", tools=[], handlers={})

    def test_answer_with_evidence_builds_cited_prompt(self) -> None:
        captured: dict[str, str] = {}

        def fake_invoke(system_prompt: str, user_prompt: str) -> str:
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            return "answer"

        with patch.object(llm, "invoke_claude", side_effect=fake_invoke):
            answer = llm.answer_with_evidence(
                "What?",
                [{"page": "Page", "section": "Section", "url": "https://example.test", "excerpt": "Excerpt"}],
            )

        self.assertEqual(answer, "answer")
        self.assertIn("ONLY the provided", captured["system_prompt"])
        self.assertIn("[1] Page", captured["user_prompt"])
        self.assertIn("Excerpt", captured["user_prompt"])


if __name__ == "__main__":
    unittest.main()
