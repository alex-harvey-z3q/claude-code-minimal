from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import config
from .llm import invoke_claude
from .retrieval import retrieve

MAX_ITERS = 3
WORKSPACE_DIR = Path("/tmp/workspace")

_FILE_HEADER_RE = re.compile(r"^===\s*(?P<filename>.+?)\s*===\s*$", re.MULTILINE)


def _format_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "No retrieved evidence."
    parts = []
    for i, item in enumerate(evidence, start=1):
        source_type = item.get("source_type", "unknown")
        parts.append(
            f"[{i}] ({source_type}) {item['page']} — {item['section']}\n"
            f"URL: {item['url']}\n"
            f"Excerpt: {item['excerpt']}"
        )
    return "\n\n".join(parts)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_files_from_response(code: str) -> list[tuple[str, str]]:
    content = _strip_code_fences(code)
    matches = list(_FILE_HEADER_RE.finditer(content))
    if not matches:
        raise ValueError("Implementer output did not contain any === filename === blocks.")

    files: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        filename = match.group("filename").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        body = content[start:end].lstrip("\n")
        if not body.endswith("\n"):
            body += "\n"
        files.append((filename, body))
    return files


def write_files_from_response(code: str, workspace: Path) -> None:
    files = _parse_files_from_response(code)

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    workspace_root = workspace.resolve()
    for filename, body in files:
        path = (workspace / filename).resolve()
        if path != workspace_root and workspace_root not in path.parents:
            raise ValueError(f"Refusing to write outside workspace: {filename}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


def run_tests(workspace: Path) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(workspace)},
        )
    except Exception as exc:
        return False, f"Test runner failed to start:\n{exc}"

    output_parts = []
    if result.stdout.strip():
        output_parts.append(result.stdout.strip())
    if result.stderr.strip():
        output_parts.append(result.stderr.strip())
    output = "\n\n".join(output_parts).strip() or "No test output captured."
    return result.returncode == 0, output


def no_major_issues(review: str) -> bool:
    lines = [line.strip() for line in review.splitlines() if line.strip()]
    if not lines:
        return True

    lowered = review.lower()
    if "no major issues" in lowered:
        return True

    return not any(re.search(r"\bMAJOR\b", line) for line in lines)


def plan_task(question: str, evidence: list[dict]) -> str:
    system_prompt = (
        "You are Planner, a software planning agent. Produce a concise, structured "
        "implementation plan for a small Python application. When retrieved evidence "
        "contains implementation conventions, style guidance, file layout guidance, "
        "testing guidance, or CLI conventions, treat that guidance as authoritative "
        "unless it conflicts with the user's explicit requirements. Use retrieved "
        "domain evidence for gameplay rules and behaviour. Be deterministic. "
        "Do not write code."
    )
    user_prompt = (
        f"Task: {question}\n\n"
        f"Retrieved evidence:\n{_format_evidence(evidence)}\n\n"
        "Return these sections only:\n"
        "1. Files\n"
        "2. Data structures\n"
        "3. Conventions to follow\n"
        "4. Game rules and flow\n"
        "5. Test strategy\n\n"
        "Requirements:\n"
        "- Extract concrete conventions from the retrieved evidence\n"
        "- Make the plan explicitly reflect retrieved file layout, style, CLI, and test conventions when present\n"
        "- Do not invent conventions that are not supported by the evidence\n"
        "- Keep it compact and implementation-ready"
    )
    return invoke_claude(system_prompt, user_prompt, max_tokens=900, temperature=0.0)


def implement_task(
    question: str,
    evidence: list[dict],
    plan: str,
    *,
    previous_code: str | None = None,
    test_output: str | None = None,
    review_feedback: str | None = None,
) -> str:
    system_prompt = (
        "You are Implementer, a Python coding agent. Generate a complete, runnable, "
        "terminal-based Python application. When retrieved evidence contains "
        "implementation conventions, style guidance, file layout guidance, testing "
        "guidance, naming guidance, or CLI conventions, you must follow that guidance "
        "unless it conflicts with the user's explicit requirements. Do not silently "
        "replace retrieved conventions with your own defaults. Use retrieved domain "
        "evidence for gameplay rules and behaviour. Output only code files using the "
        "exact separator format === filename ===. When fixing an existing revision, "
        "return the full updated file set, not a diff."
    )

    feedback_sections = []
    if previous_code:
        feedback_sections.append(f"Previous code:\n{previous_code}")
    if test_output:
        feedback_sections.append(f"Test failures:\n{test_output}")
    if review_feedback:
        feedback_sections.append(f"Review feedback:\n{review_feedback}")

    feedback_block = ""
    if feedback_sections:
        feedback_block = (
            "Fix the following issues in the next revision. "
            "Keep what already works and correct only the problems described below.\n\n"
            + "\n\n".join(feedback_sections)
            + "\n\n"
        )

    user_prompt = (
        f"Task: {question}\n\n"
        f"Retrieved evidence:\n{_format_evidence(evidence)}\n\n"
        f"Plan:\n{plan}\n\n"
        f"{feedback_block}"
        "Generate the full application now.\n\n"
        "Hard requirements:\n"
        "- Python only\n"
        "- standard library preferred\n"
        "- playable in the terminal\n"
        "- configurable board width, height, and mine count\n"
        "- reveal cells\n"
        "- flag and unflag cells\n"
        "- recursive reveal for empty areas\n"
        "- win/loss detection\n"
        "- text board rendering\n"
        "- restartable game loop\n"
        "- modular, readable, compact\n"
        "- include basic tests for core game logic\n\n"
        "Retrieved evidence handling:\n"
        "- Treat retrieved style and conventions as binding implementation guidance when present\n"
        "- Prefer retrieved file layout, naming, rendering, CLI, and test conventions over generic defaults\n"
        "- Only depart from retrieved conventions if following them would violate the task requirements\n"
        "- Do not explain the conventions; just implement them\n\n"
        "Output multiple files in one plain-text response using separators like:\n"
        "=== main.py ===\n"
        "...\n"
        "=== game.py ===\n"
        "...\n"
        "=== test_game.py ==="
    )
    return invoke_claude(system_prompt, user_prompt, max_tokens=4500, temperature=0.0)


def review_code(
    question: str,
    evidence: list[dict],
    code: str,
    *,
    test_output: str = "",
) -> str:
    system_prompt = (
        "You are Reviewer, a software review agent. Review generated Python code for "
        "correctness, completeness, and adherence to retrieved evidence. When "
        "retrieved evidence contains implementation conventions, style guidance, file "
        "layout guidance, testing guidance, naming guidance, or CLI conventions, "
        "treat those conventions as the review baseline unless they conflict with the "
        "user's explicit requirements. Be specific, concise, and deterministic."
    )
    user_prompt = (
        f"Task: {question}\n\n"
        f"Retrieved evidence:\n{_format_evidence(evidence)}\n\n"
        "Review the generated application. Focus on:\n"
        "- correctness\n"
        "- game logic\n"
        "- adherence to retrieved style and conventions\n"
        "- mismatches with retrieved gameplay evidence\n"
        "- edge cases\n"
        "- structure and readability\n"
        "- test coverage\n"
        "- whether the test output indicates unresolved defects\n\n"
        "Return short bullet points only. Prefix each bullet with one of:\n"
        "- MAJOR: for blocking issues that must be fixed before stopping\n"
        "- MINOR: for non-blocking improvements\n"
        "- PASS: when something is good and should remain unchanged\n\n"
        f"Code:\n{code}\n\n"
        f"Test output:\n{test_output or 'No test output.'}"
    )
    return invoke_claude(system_prompt, user_prompt, max_tokens=1200, temperature=0.0)


def run_workflow(question: str, use_retrieval: bool = True) -> dict:
    evidence = retrieve(question) if use_retrieval else []
    plan = plan_task(question, evidence)

    code = ""
    review = ""
    previous_code = None
    test_output = None
    review_feedback = None

    for _ in range(MAX_ITERS):
        code = implement_task(
            question,
            evidence,
            plan,
            previous_code=previous_code,
            test_output=test_output,
            review_feedback=review_feedback,
        )

        write_files_from_response(code, WORKSPACE_DIR)
        tests_passed, test_output = run_tests(WORKSPACE_DIR)
        review = review_code(question, evidence, code, test_output=test_output)

        if tests_passed and no_major_issues(review):
            break

        previous_code = code
        review_feedback = review

    return {
        "evidence": evidence,
        "plan": plan,
        "code": code,
        "review": review,
    }
