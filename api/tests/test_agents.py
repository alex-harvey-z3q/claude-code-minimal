from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_USER", "user")
os.environ.setdefault("DB_PASSWORD", "password")
os.environ.setdefault("EMBED_DIM", "1536")

from api import agents
from api.llm import ToolLoopError
from api.sandbox import SandboxSession


class AgentHelperTest(unittest.TestCase):
    def test_format_evidence_renders_items_and_empty_state(self) -> None:
        self.assertEqual(agents._format_evidence([]), "No retrieved evidence.")

        rendered = agents._format_evidence(
            [
                {
                    "source_type": "wiki",
                    "page": "Page",
                    "section": "Section",
                    "url": "https://example.test",
                    "excerpt": "Excerpt",
                }
            ]
        )

        self.assertIn("[1] (wiki) Page", rendered)
        self.assertIn("URL: https://example.test", rendered)
        self.assertIn("Excerpt: Excerpt", rendered)

    def test_test_output_helpers_parse_runtime_facts(self) -> None:
        self.assertEqual(agents._parse_test_run_count("Ran 2 tests in 0.01s"), 2)
        self.assertEqual(agents._parse_test_run_count("no count"), 0)
        self.assertTrue(agents._tests_actually_ran("Ran 1 test"))
        self.assertFalse(agents._tests_actually_ran("Ran 0 tests"))

        status = agents._build_test_status("Ran 2 tests", tests_passed=True)
        self.assertIn("tests_ran: true", status)
        self.assertIn("tests_passed: true", status)
        self.assertIn("tests_run_count: 2", status)

    def test_summarize_test_output_prioritizes_failures(self) -> None:
        output = "noise\nERROR: test_a\nTraceback line\nFAILED (errors=1)\n"

        self.assertEqual(
            agents._summarize_test_output(output),
            "ERROR: test_a\nTraceback line\nFAILED (errors=1)",
        )
        self.assertEqual(agents._summarize_test_output("plain output", limit=5), "plain")

    def test_review_items_support_single_line_and_section_styles(self) -> None:
        review = """
        PASS: Looks fine
        MAJOR:
        - Broken import
        - none
        MINOR: Rename variable
        MAJOR: USER_REQUIREMENT - Missing CLI
        """

        major, minor = agents._extract_review_items(review)

        self.assertEqual(
            major,
            [
                "MAJOR: Broken import",
                "MAJOR: USER_REQUIREMENT - Missing CLI",
            ],
        )
        self.assertEqual(minor, ["MINOR: Rename variable"])

    def test_major_issues_ignores_none_like_major_lines(self) -> None:
        self.assertFalse(agents.major_issues("MAJOR: none\nMINOR: polish"))
        self.assertFalse(agents.major_issues("MAJOR: no blocking issues"))
        self.assertTrue(agents.major_issues("MAJOR: FAILING_TEST - test failed"))

    def test_build_blocking_checklist_deduplicates_test_and_review_items(self) -> None:
        checklist = agents._build_blocking_checklist(
            "FAIL: test_math\nFAIL: test_math\nok",
            "MAJOR: FAILING_TEST - test_math\nMINOR: Add docs",
        )

        self.assertEqual(
            checklist,
            ["FAIL: test_math", "MAJOR: FAILING_TEST - test_math"],
        )

    def test_select_retry_files_uses_workspace_and_related_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "tests").mkdir()
            (workspace / "tests" / "test_game.py").write_text("", encoding="utf-8")
            (workspace / "game.py").write_text("", encoding="utf-8")
            (workspace / "agent_trace.json").write_text("{}", encoding="utf-8")

            selected = agents._select_retry_files(
                "ERROR: tests/test_game.py",
                "MAJOR: game.py fails",
                workspace,
            )

            self.assertEqual(selected, ["tests/test_game.py", "game.py"])

    def test_issue_summary_contains_retry_guidance_and_condensed_failures(self) -> None:
        summary = agents._build_issue_summary(
            ["tests/test_game.py", "game.py"],
            "noise\nFAIL: test_game\nTraceback line\nmore noise",
            "MAJOR: Broken flow\nMINOR: Cosmetic",
        )

        self.assertIn("Files to revise:", summary)
        self.assertIn("Use the read_file tool", summary)
        self.assertIn("FAIL: test_game", summary)
        self.assertIn("MAJOR: Broken flow", summary)
        self.assertIn("MINOR: Cosmetic", summary)

    def test_workflow_error_payload_includes_tool_loop_summary(self) -> None:
        exc = ToolLoopError(
            "stuck",
            provider="bedrock",
            model_id="model",
            max_tool_rounds=2,
            tool_calls=[{"name": "read_file"}, {"name": "read_file"}],
            partial_text="partial",
        )

        payload = agents._workflow_error_payload(exc)

        self.assertEqual(payload["type"], "ToolLoopError")
        self.assertEqual(payload["tool_loop"]["tool_call_counts"], {"read_file": 2})
        self.assertEqual(payload["tool_loop"]["partial_text"], "partial")

    def test_workflow_trace_contains_workspace_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = SandboxSession(Path(tmpdir), run_id="run")

            trace = agents._build_workflow_trace(
                question="q",
                use_retrieval=False,
                sandbox=sandbox,
                evidence=[{"page": "p"}],
                plan="plan",
                stop_reason="done",
            )

            self.assertEqual(trace["workspace_id"], "run")
            self.assertEqual(trace["evidence_count"], 1)
            self.assertEqual(trace["stop_reason"], "done")
            self.assertEqual(trace["observability"]["outer_iteration_count"], 0)

    def test_tool_observability_summarizes_inner_loops(self) -> None:
        observability = agents._build_tool_observability(
            [
                {
                    "iteration": 1,
                    "retry_mode": False,
                    "trace": {
                        "implement": {
                            "tool_loop": {
                                "round_count": 3,
                                "tool_call_count": 4,
                                "tool_call_counts": {"write_file": 2, "run_tests": 2},
                                "run_tests_count": 2,
                                "error_count": 1,
                                "malformed_tool_call_count": 0,
                                "final_text_preview": "done",
                            }
                        },
                        "review": {
                            "tool_loop": {
                                "round_count": 1,
                                "tool_call_count": 1,
                                "tool_call_counts": {"read_file": 1},
                                "run_tests_count": 0,
                                "error_count": 0,
                                "malformed_tool_call_count": 0,
                                "final_text_preview": "PASS",
                            }
                        },
                    },
                }
            ],
            stop_reason="tests_passed_and_review_clean",
            error=None,
        )

        self.assertTrue(observability["completed"])
        self.assertEqual(observability["total_tool_call_count"], 5)
        self.assertEqual(observability["tool_call_counts"], {"write_file": 2, "run_tests": 2, "read_file": 1})
        self.assertEqual([phase["phase"] for phase in observability["phases"]], ["implement", "review"])


class WorkflowTest(unittest.TestCase):
    def test_run_workflow_completes_with_mocked_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            def fake_implement(
                question: str,
                evidence: list[dict],
                plan: str,
                sandbox: SandboxSession,
                *,
                issue_summary: str | None = None,
                retry_mode: bool = False,
            ) -> tuple[str, dict[str, str]]:
                del question, evidence, plan, issue_summary, retry_mode
                sandbox.write_file("main.py", "VALUE = 1\n")
                sandbox.write_file(
                    "test_main.py",
                    "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
                )
                return "implemented", {"system_prompt": "s", "user_prompt": "u", "response": "implemented"}

            with (
                patch.object(agents, "WORKSPACE_DIR", tmpdir),
                patch.object(agents, "plan_task", return_value=("plan", {"system_prompt": "s", "user_prompt": "u", "response": "plan"})),
                patch.object(agents, "implement_task", side_effect=fake_implement),
                patch.object(agents, "review_code", return_value=("PASS: ok", {"system_prompt": "s", "user_prompt": "u", "response": "PASS: ok"})),
            ):
                result = agents.run_workflow("build it", use_retrieval=False)

            self.assertEqual(result["stop_reason"], "tests_passed_and_review_clean")
            self.assertEqual(result["completed_iteration"], 1)
            self.assertTrue(result["observability"]["completed"])
            self.assertIn("=== main.py ===", result["code"])
            self.assertNotIn("agent_trace.json", result["code"])


if __name__ == "__main__":
    unittest.main()
