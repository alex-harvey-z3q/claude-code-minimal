from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .config import MAX_WORKFLOW_ITERS, TEST_TIMEOUT_SECONDS, WORKSPACE_DIR
from .llm import invoke_claude
from .retrieval import retrieve

MAX_ITERS = MAX_WORKFLOW_ITERS
WORKSPACE_ROOT = Path(WORKSPACE_DIR)
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


def _normalize_file_body(filename: str, body: str) -> str:
    content = body.lstrip("\n").rstrip()

    if content.startswith("```"):
        lines = content.splitlines()
        lines = lines[1:]
        if not lines or lines[-1].strip() != "```":
            raise ValueError(f"Incomplete Markdown code fence in {filename}")
        content = "\n".join(lines[:-1]).rstrip()

    if not content.strip():
        raise ValueError(f"Empty file body for {filename}")

    return content + "\n"


def _parse_files_from_response(code: str) -> list[tuple[str, str]]:
    matches = list(_FILE_HEADER_RE.finditer(code))
    if not matches:
        raise ValueError("Implementer output did not contain any === filename === blocks.")

    files: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        filename = match.group("filename").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(code)
        body = code[start:end]
        files.append((filename, _normalize_file_body(filename, body)))

    if len(files) < 2:
        raise ValueError("Implementer output must contain multiple files")

    if not any(
        Path(filename).name.startswith("test_") or "tests" in Path(filename).parts
        for filename, _ in files
    ):
        raise ValueError("Implementer output must include at least one unittest test file")

    return files


def write_files_from_response(code: str, workspace: Path) -> None:
    files = _parse_files_from_response(code)

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    workspace_root = workspace.resolve()
    for filename, body in files:
        destination = (workspace / filename).resolve()
        if workspace_root != destination and workspace_root not in destination.parents:
            raise ValueError(f"Refusing to write outside workspace: {filename}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(body, encoding="utf-8")


def run_tests(workspace: Path) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
            timeout=TEST_TIMEOUT_SECONDS,
            env={**os.environ, "PYTHONPATH": str(workspace)},
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        output = "\n\n".join(
            part for part in [stdout.strip(), stderr.strip()] if part
        ) or "Tests timed out."
        return False, f"Test run timed out after {TEST_TIMEOUT_SECONDS} seconds.\n\n{output}"
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


def no_major_issues(review: str) -> bool:
    found_major = False
    for raw_line in review.splitlines():
        line = raw_line.strip()
        if not line.startswith("MAJOR:"):
            continue

        found_major = True
        remainder = line[len("MAJOR:"):].strip().lower()
        if remainder.startswith("none") or remainder.startswith("no ") or remainder == "n/a":
            continue
        return False

    return found_major or "no major issues" in review.lower()


def _summarize_test_output(test_output: str, limit: int = 1200) -> str:
    important_lines = []
    for line in test_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (
            stripped.startswith(("FAIL:", "ERROR:", "Traceback", "SyntaxError:", "ImportError:", "AssertionError:"))
            or "FAILED" in stripped
            or "Ran 0 tests" in stripped
            or "timed out" in stripped.lower()
            or "invalid syntax" in stripped.lower()
        ):
            important_lines.append(stripped)

    summary = "\n".join(important_lines).strip()
    if not summary:
        summary = test_output.strip()

    return summary[:limit]


def _extract_major_review_items(review: str) -> str:
    major_lines = [line.strip() for line in review.splitlines() if line.strip().startswith("MAJOR:")]
    if major_lines:
        return "\n".join(major_lines)
    return review.strip()[:1200]


def _build_issue_summary(test_output: str, review: str) -> str:
    parts = []
    if test_output.strip():
        parts.append(f"Test failures:\n{_summarize_test_output(test_output)}")
    if review.strip():
        parts.append(f"Review feedback:\n{_extract_major_review_items(review)}")
    return "\n\n".join(parts).strip()


def plan_task(question: str, evidence: list[dict]) -> str:
    system_prompt = (
        "You are Planner, a software planning agent. Produce a concise, structured "
        "implementation plan for a Python application or code change. When retrieved "
        "evidence contains implementation conventions, style guidance, file layout "
        "guidance, testing guidance, or CLI conventions, treat that guidance as "
        "authoritative unless it conflicts with the user's explicit requirements. "
        "Use retrieved domain evidence for task requirements and expected behaviour. "
        "Be deterministic. Do not write code."
    )
    user_prompt = (
        f"Task: {question}\n\n"
        f"Retrieved evidence:\n{_format_evidence(evidence)}\n\n"
        "Return these sections only:\n"
        "1. Files\n"
        "2. Data structures\n"
        "3. Conventions to follow\n"
        "4. Behaviour and flow\n"
        "5. Test strategy\n\n"
        "Requirements:\n"
        "- Extract concrete conventions from the retrieved evidence\n"
        "- Make the plan explicitly reflect retrieved file layout, style, CLI, and test conventions when present\n"
        "- Do not invent conventions that are not supported by the evidence\n"
        "- Use unittest from the Python standard library\n"
        "- Make the tests discoverable by python -m unittest discover\n"
        "- Keep it compact and implementation-ready"
    )
    return invoke_claude(system_prompt, user_prompt, max_tokens=900, temperature=0.0)


def implement_task(
    question: str,
    evidence: list[dict],
    plan: str,
    *,
    issue_summary: str | None = None,
) -> str:
    system_prompt = (
        "You are Implementer, a Python coding agent. Generate a complete, runnable "
        "Python implementation for the requested task. When retrieved evidence "
        "contains implementation conventions, style guidance, file layout guidance, "
        "testing guidance, naming guidance, or CLI conventions, you must follow that "
        "guidance unless it conflicts with the user's explicit requirements. Do not "
        "silently replace retrieved conventions with your own defaults. Use "
        "retrieved domain evidence for task requirements and expected behaviour. "
        "Output only code files using the exact separator format === filename ===. "
        "Do not include any prose before, between, or after files. Do not wrap files "
        "in Markdown code fences."
    )

    feedback_block = ""
    if issue_summary:
        feedback_block = (
            "Fix the following issues and regenerate the full updated file set.\n\n"
            f"{issue_summary}\n\n"
        )

    user_prompt = (
        f"Task: {question}\n\n"
        f"Retrieved evidence:\n{_format_evidence(evidence)}\n\n"
        f"Plan:\n{plan}\n\n"
        f"{feedback_block}"
        "Generate the full implementation now.\n\n"
        "Hard requirements:\n"
        "- Python only\n"
        "- include or update unittest tests needed to validate the requested behaviour\n"
        "- tests must be discoverable by python -m unittest discover from the project root\n"
        "- output the full updated file set using === filename === separators\n"
        "- do not include Markdown fences\n\n"
        "Retrieved evidence handling:\n"
        "- Treat retrieved style and conventions as binding implementation guidance when present\n"
        "- Prefer retrieved file layout, naming, rendering, CLI, and test conventions over generic defaults\n"
        "- Only depart from retrieved conventions if following them would violate the task requirements\n"
        "- Do not explain the conventions; just implement them\n\n"
        "Output multiple files in one plain-text response using separators like:\n"
        "=== main.py ===\n"
        "...\n"
        "=== module.py ===\n"
        "...\n"
        "=== test_module.py ==="
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
        "Review the generated implementation. Focus on:\n"
        "- correctness\n"
        "- adherence to retrieved style and conventions\n"
        "- mismatches with retrieved requirements or behaviour\n"
        "- edge cases\n"
        "- structure and readability\n"
        "- test coverage\n"
        "- whether tests are discoverable and actually ran\n\n"
        "Return short bullet points only. Prefix each bullet with one of:\n"
        "- PASS:\n"
        "- MINOR:\n"
        "- MAJOR:\n\n"
        "Treat 'Ran 0 tests' as a MAJOR issue.\n\n"
        f"Test output:\n{test_output or 'No test output.'}\n\n"
        f"Code:\n{code}"
    )
    return invoke_claude(system_prompt, user_prompt, max_tokens=1000, temperature=0.0)


def run_workflow(question: str, use_retrieval: bool = True) -> dict[str, object]:
    evidence = retrieve(question) if use_retrieval else []
    plan = plan_task(question, evidence)

    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    code = ""
    review = ""
    issue_summary: str | None = None
    iterations: list[dict[str, object]] = []
    stop_reason = "max_iterations_reached"

    for iteration in range(1, MAX_ITERS + 1):
        code = implement_task(
            question,
            evidence,
            plan,
            issue_summary=issue_summary,
        )

        try:
            write_files_from_response(code, WORKSPACE_ROOT)
            tests_passed, test_output = run_tests(WORKSPACE_ROOT)
            review = review_code(question, evidence, code, test_output=test_output)
        except ValueError as exc:
            tests_passed = False
            test_output = f"File emission validation failed:\n{exc}"
            review = f"MAJOR: {exc}"

        major_issues = not no_major_issues(review)

        iterations.append(
            {
                "iteration": iteration,
                "tests_passed": tests_passed,
                "major_issues": major_issues,
                "test_output": test_output,
                "review": review,
            }
        )

        if tests_passed and not major_issues:
            stop_reason = "tests_passed_and_review_clean"
            break

        issue_summary = _build_issue_summary(test_output, review)

    return {
        "evidence": evidence,
        "plan": plan,
        "code": code,
        "review": review,
        "iterations": iterations,
        "completed_iteration": len(iterations),
        "stop_reason": stop_reason,
    }
