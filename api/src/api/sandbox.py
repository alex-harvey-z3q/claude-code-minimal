from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import TEST_TIMEOUT_SECONDS, WORKSPACE_DIR

TEXT_SUFFIXES = {".py", ".txt", ".md", ".json", ".yaml", ".yml"}
MAX_FILE_CHARS = 80_000
MAX_TOOL_RESULT_CHARS = 12_000


@dataclass(frozen=True)
class ToolExecution:
    ok: bool
    text: str


class SandboxSession:
    """Host-controlled workspace used by coding agents.

    This is a filesystem sandbox rather than a security boundary. It prevents
    path traversal and keeps each workflow in an isolated directory. Test
    execution still runs as a subprocess on the host, with a scrubbed
    environment, so a container remains the next hardening step before running
    truly untrusted code.
    """

    def __init__(self, root: Path | None = None, *, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        base_root = Path(root or WORKSPACE_DIR)
        self.root = (base_root / self.run_id).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str = ".") -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"Refusing to access path outside sandbox: {path}")
        return candidate

    def relative_files(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            str(path.relative_to(self.root))
            for path in self.root.rglob("*")
            if path.is_file()
        )

    def list_files(self, path: str = ".") -> str:
        base = self.resolve(path)
        if not base.exists():
            return f"No such path: {path}"
        if base.is_file():
            return str(base.relative_to(self.root))

        files = [
            str(child.relative_to(self.root))
            for child in base.rglob("*")
            if child.is_file()
        ]
        return "\n".join(sorted(files)) or "No files."

    def read_file(self, path: str) -> str:
        file_path = self.resolve(path)
        if not file_path.exists():
            raise ValueError(f"No such file: {path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")
        if file_path.suffix not in TEXT_SUFFIXES:
            raise ValueError(f"Refusing to read unsupported file type: {path}")

        text = file_path.read_text(encoding="utf-8")
        if len(text) > MAX_FILE_CHARS:
            return text[:MAX_FILE_CHARS] + "\n...[truncated]..."
        return text

    def write_file(self, path: str, content: str) -> str:
        file_path = self.resolve(path)
        if file_path.suffix not in TEXT_SUFFIXES:
            raise ValueError(f"Refusing to write unsupported file type: {path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {path} ({len(content)} characters)."

    def delete_file(self, path: str) -> str:
        file_path = self.resolve(path)
        if not file_path.exists():
            return f"No such file: {path}"
        if not file_path.is_file():
            raise ValueError(f"Refusing to delete non-file path: {path}")
        file_path.unlink()
        return f"Deleted {path}."

    def run_tests(self) -> tuple[bool, str]:
        tests_dir = self.root / "tests"
        if tests_dir.exists() and tests_dir.is_dir():
            cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test*.py"]
        else:
            cmd = [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test*.py"]

        env = {
            "HOME": os.getenv("HOME", ""),
            "LANG": os.getenv("LANG", "C.UTF-8"),
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": str(self.root),
            "TERM": os.getenv("TERM", "dumb"),
        }

        try:
            result = subprocess.run(
                cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
                timeout=TEST_TIMEOUT_SECONDS,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _decode_output(exc.stdout)
            stderr = _decode_output(exc.stderr)
            output = "\n\n".join(part for part in [stdout.strip(), stderr.strip()] if part)
            return False, f"Test run timed out after {TEST_TIMEOUT_SECONDS} seconds.\n\n{output or 'Tests timed out.'}"
        except Exception as exc:
            return False, f"Test runner failed to start:\n{exc}"

        output_parts = []
        if result.stdout.strip():
            output_parts.append(result.stdout.strip())
        if result.stderr.strip():
            output_parts.append(result.stderr.strip())
        output = "\n\n".join(output_parts).strip() or "No test output captured."

        if "Ran 0 tests" in output:
            return False, output
        return result.returncode == 0, output

    def run_tests_tool(self) -> str:
        passed, output = self.run_tests()
        status = "passed" if passed else "failed"
        return f"Tests {status}.\n\n{output}"

    def snapshot(self) -> str:
        blocks = []
        for filename in self.relative_files():
            path = self.root / filename
            if path.suffix not in TEXT_SUFFIXES:
                continue
            content = path.read_text(encoding="utf-8")
            blocks.append(f"=== {filename} ===\n{content.rstrip()}\n")
        return "\n".join(blocks).strip()

    def tools(self, *, read_only: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tool_specs = [
            {
                "toolSpec": {
                    "name": "list_files",
                    "description": "List files in the sandbox workspace.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"path": {"type": "string", "default": "."}},
                        }
                    },
                }
            },
            {
                "toolSpec": {
                    "name": "read_file",
                    "description": "Read a UTF-8 text file from the sandbox workspace.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        }
                    },
                }
            },
            {
                "toolSpec": {
                    "name": "run_tests",
                    "description": "Run unittest discovery in the sandbox workspace and return the output.",
                    "inputSchema": {"json": {"type": "object", "properties": {}}},
                }
            },
        ]
        handlers: dict[str, Any] = {
            "list_files": lambda path=".": self.list_files(path),
            "read_file": self.read_file,
            "run_tests": lambda: self.run_tests_tool(),
        }

        if not read_only:
            tool_specs.extend(
                [
                    {
                        "toolSpec": {
                            "name": "write_file",
                            "description": "Create or replace a UTF-8 text file in the sandbox workspace.",
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
                    },
                    {
                        "toolSpec": {
                            "name": "delete_file",
                            "description": "Delete one file from the sandbox workspace.",
                            "inputSchema": {
                                "json": {
                                    "type": "object",
                                    "properties": {"path": {"type": "string"}},
                                    "required": ["path"],
                                }
                            },
                        }
                    },
                ]
            )
            handlers["write_file"] = self.write_file
            handlers["delete_file"] = self.delete_file

        return tool_specs, handlers

    def trace_path(self) -> Path:
        return self.root / "agent_trace.json"

    def write_trace(self, trace: dict[str, Any]) -> None:
        self.trace_path().write_text(
            json.dumps(trace, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )


def _decode_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def truncate_tool_result(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    return text[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]..."
