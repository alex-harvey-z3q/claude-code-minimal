from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_USER", "user")
os.environ.setdefault("DB_PASSWORD", "password")
os.environ.setdefault("EMBED_DIM", "1536")

from api.sandbox import SandboxSession


class SandboxSessionTest(unittest.TestCase):
    def test_write_and_read_file_stays_inside_session_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = SandboxSession(Path(tmpdir), run_id="run")

            result = sandbox.write_file("pkg/module.py", "VALUE = 1\n")

            self.assertIn("Wrote pkg/module.py", result)
            self.assertEqual(sandbox.read_file("pkg/module.py"), "VALUE = 1\n")
            self.assertEqual(sandbox.list_files(), "pkg/module.py")

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = SandboxSession(Path(tmpdir), run_id="run")

            with self.assertRaises(ValueError):
                sandbox.write_file("../escape.py", "VALUE = 1\n")

    def test_run_tests_uses_session_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = SandboxSession(Path(tmpdir), run_id="run")
            sandbox.write_file("test_example.py", "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n")

            passed, output = sandbox.run_tests()

            self.assertTrue(passed)
            self.assertIn("Ran 1 test", output)

    def test_write_trace_writes_json_trace_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = SandboxSession(Path(tmpdir), run_id="run")

            sandbox.write_trace({"stop_reason": "debug", "tool_calls": [{"name": "run_tests"}]})

            trace = sandbox.trace_path().read_text(encoding="utf-8")
            self.assertIn('"stop_reason": "debug"', trace)
            self.assertIn('"name": "run_tests"', trace)

    def test_trace_file_is_hidden_from_workspace_tools_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = SandboxSession(Path(tmpdir), run_id="run")
            sandbox.write_file("hello.py", "VALUE = 1\n")
            sandbox.write_trace({"stop_reason": "debug"})

            self.assertEqual(sandbox.list_files(), "hello.py")
            self.assertNotIn("agent_trace.json", sandbox.relative_files())
            self.assertNotIn("agent_trace.json", sandbox.snapshot())

            with self.assertRaises(ValueError):
                sandbox.read_file("agent_trace.json")
            with self.assertRaises(ValueError):
                sandbox.list_files("agent_trace.json")


if __name__ == "__main__":
    unittest.main()
