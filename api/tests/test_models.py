from __future__ import annotations

import os
import unittest

from pydantic import ValidationError

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_USER", "user")
os.environ.setdefault("DB_PASSWORD", "password")
os.environ.setdefault("EMBED_DIM", "1536")

from api.models import AskRequest, EvidenceItem, WorkflowResponse


class ModelTest(unittest.TestCase):
    def test_ask_request_validates_question_length(self) -> None:
        with self.assertRaises(ValidationError):
            AskRequest(question="no")

        request = AskRequest(question="valid question")
        self.assertEqual(request.question, "valid question")

    def test_workflow_response_accepts_nested_trace_payload(self) -> None:
        response = WorkflowResponse(
            evidence=[
                EvidenceItem(
                    page="Page",
                    section="Section",
                    url="https://example.test",
                    excerpt="Excerpt",
                )
            ],
            plan="plan",
            code="code",
            workspace_id="workspace",
            review="PASS: ok",
            iterations=[
                {
                    "iteration": 1,
                    "retry_mode": False,
                    "tests_passed": True,
                    "major_issues": False,
                    "test_output": "Ran 1 test",
                    "review": "PASS: ok",
                    "retry_files": [],
                    "issue_summary": None,
                    "blocking_checklist": [],
                    "workspace_snapshot": "code",
                    "implement_output": "done",
                    "trace": {
                        "implement": {"system_prompt": "s", "user_prompt": "u", "response": "r"},
                        "review": {"system_prompt": "s", "user_prompt": "u", "response": "r"},
                    },
                }
            ],
            completed_iteration=1,
            stop_reason="tests_passed_and_review_clean",
            trace={"plan": {"system_prompt": "s", "user_prompt": "u", "response": "r"}},
        )

        self.assertEqual(response.evidence[0].page, "Page")
        self.assertEqual(response.iterations[0].trace.implement.response, "r")


if __name__ == "__main__":
    unittest.main()
