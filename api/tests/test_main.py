from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_USER", "user")
os.environ.setdefault("DB_PASSWORD", "password")
os.environ.setdefault("EMBED_DIM", "1536")

from api import main
from api.agents import WorkflowExecutionError


class MainRouteTest(unittest.TestCase):
    def test_health_returns_ok(self) -> None:
        self.assertEqual(main.health(), {"status": "ok"})

    def test_query_delegates_to_workflow(self) -> None:
        expected = {
            "evidence": [],
            "plan": "plan",
            "code": "code",
            "workspace_id": "workspace",
            "review": "PASS: ok",
            "iterations": [],
            "completed_iteration": 0,
            "stop_reason": "done",
            "trace": {"plan": {"system_prompt": "s", "user_prompt": "u", "response": "r"}},
        }

        with patch.object(main, "run_workflow", return_value=expected) as run_workflow:
            result = main.query("task", use_retrieval=False)

        self.assertEqual(result, expected)
        run_workflow.assert_called_once_with("task", use_retrieval=False)

    def test_query_maps_workflow_errors_to_502(self) -> None:
        error = WorkflowExecutionError(
            "failed",
            workspace_id="workspace",
            trace_file="/tmp/trace.json",
            debug={"type": "ToolLoopError"},
        )

        with patch.object(main, "run_workflow", side_effect=error):
            with self.assertRaises(HTTPException) as raised:
                main.query("task", use_retrieval=False)

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail["workspace_id"], "workspace")
        self.assertEqual(raised.exception.detail["debug"], {"type": "ToolLoopError"})

    def test_query_maps_runtime_errors_to_502(self) -> None:
        with patch.object(main, "run_workflow", side_effect=RuntimeError("boom")):
            with self.assertRaises(HTTPException) as raised:
                main.query("task", use_retrieval=False)

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail, "boom")


if __name__ == "__main__":
    unittest.main()
