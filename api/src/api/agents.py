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
_PATH_RE = re.compile(r"(?P<path>[\w./-]+\.(?:py|txt|md|json|yaml|yml))")


def _format_evidence(evidence: list[dict]) -> str:
    """Render retrieved evidence items into a compact prompt-friendly string.

    The planner, implementer, and reviewer all consume the same evidence block.
    This helper keeps that formatting consistent across all three stages.
    """
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
    """Normalize one emitted file body before it is written to disk.

    The implementer is instructed to return plain file contents, but models may
    still wrap files in Markdown fences. This helper strips those fences, allows
    intentionally empty __init__.py files, and rejects obviously broken outputs
    such as incomplete fences or empty non-package files.
    """
    content = body.lstrip("\n").rstrip()

    if content.startswith("```"):
        lines = content.splitlines()
        lines = lines[1:]
        if not lines or lines[-1].strip() != "```":
            raise ValueError(f"Incomplete Markdown code fence in {filename}")
        content = "\n".join(lines[:-1]).rstrip()

    if not content.strip():
        if Path(filename).name == "__init__.py":
            return ""
        raise ValueError(f"Empty file body for {filename}")

    return content + "\n"


def _parse_files_from_response(code: str) -> list[tuple[str, str]]:
    """Parse a multi-file model response into ``(filename, body)`` pairs.

    The workflow expects the implementer to emit files using the format
    ``=== filename ===``. This function validates that structure, normalizes
    each body, and enforces a minimal sanity check that the response contains
    more than one file and at least one test file.
    """
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

    if len(files) < 1:
        raise ValueError("Implementer output must contain at least one file")

    return files


def write_files_from_response(code: str, workspace: Path) -> None:
    """Replace the workspace contents with files emitted by the implementer.

    Each iteration starts from a clean workspace so the test run reflects only
    the current candidate solution. Paths are validated to prevent the model
    from writing outside the sandbox directory.
    """
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


def apply_file_updates_from_response(code: str, workspace: Path) -> None:
    """Apply only the emitted files onto the existing workspace."""
    files = _parse_files_from_response(code)
    workspace.mkdir(parents=True, exist_ok=True)
    workspace_root = workspace.resolve()

    for filename, body in files:
        destination = (workspace / filename).resolve()
        if workspace_root != destination and workspace_root not in destination.parents:
            raise ValueError(f"Refusing to write outside workspace: {filename}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(body, encoding="utf-8")


def run_tests(workspace: Path) -> tuple[bool, str]:
    """Run unittest discovery inside the generated workspace."""
    tests_dir = workspace / "tests"
    if tests_dir.exists() and tests_dir.is_dir():
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test*.py"]
    else:
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test*.py"]

    try:
        result = subprocess.run(
            cmd,
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

        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

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


def _parse_test_run_count(test_output: str) -> int:
    """Extract the unittest run count from raw test output.

    If the standard ``Ran N tests`` line is missing, this returns zero.
    """
    match = re.search(r"Ran\s+(\d+)\s+tests?", test_output)
    if not match:
        return 0
    return int(match.group(1))


def _tests_actually_ran(test_output: str) -> bool:
    """Return True only when the runtime output confirms real test execution."""
    return _parse_test_run_count(test_output) > 0 and "Ran 0 tests" not in test_output


def _build_test_status(test_output: str, tests_passed: bool) -> str:
    """Build a compact block of authoritative runtime test facts for the reviewer.

    The reviewer should not guess whether tests ran or passed. This block tells
    it exactly what the executor observed so review comments stay grounded.
    """
    tests_ran = _tests_actually_ran(test_output)
    test_count = _parse_test_run_count(test_output)
    return (
        "Authoritative test execution facts from the runtime:\n"
        f"- tests_ran: {'true' if tests_ran else 'false'}\n"
        f"- tests_passed: {'true' if tests_passed else 'false'}\n"
        f"- tests_run_count: {test_count}\n"
        "Do not contradict these facts in your review."
    )


def major_issues(review: str) -> bool:
    """Return True when the review contains any real blocking issues."""
    for raw_line in review.splitlines():
        line = raw_line.strip()
        if not line.startswith("MAJOR:"):
            continue

        remainder = line[len("MAJOR:"):].strip().lower()
        if remainder.startswith("none") or remainder.startswith("no ") or remainder == "n/a":
            continue

        return True

    return False


def _summarize_test_output(test_output: str, limit: int = 1200) -> str:
    """Condense raw test output into the most useful lines for a retry prompt.

    This keeps the feedback loop small by prioritizing failures, errors,
    tracebacks, and other high-signal lines instead of replaying the entire
    test transcript into the next implementer prompt.
    """
    important_lines = []
    for line in test_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (
            stripped.startswith(
                ("FAIL:", "ERROR:", "Traceback", "SyntaxError:", "ImportError:", "AssertionError:")
            )
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


def _extract_review_items(review: str) -> tuple[list[str], list[str]]:
    """Split reviewer output into MAJOR and MINOR items.

    Supports both of these reviewer styles:

    1. Single-line items:
       MAJOR: Missing import
       MINOR: Add docstrings

    2. Section style:
       MAJOR:
       - Missing import
       - Another blocker
       MINOR:
       - Add docstrings
    """
    major_lines: list[str] = []
    minor_lines: list[str] = []

    current_section: str | None = None

    for raw_line in review.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line == "MAJOR:":
            current_section = "MAJOR"
            continue
        if line == "MINOR:":
            current_section = "MINOR"
            continue
        if line == "PASS:":
            current_section = "PASS"
            continue

        if line.startswith("MAJOR:"):
            remainder = line[len("MAJOR:"):].strip()
            lowered = remainder.lower()
            if remainder and not (
                lowered.startswith("none") or lowered.startswith("no ") or lowered == "n/a"
            ):
                major_lines.append(f"MAJOR: {remainder}")
            current_section = None
            continue

        if line.startswith("MINOR:"):
            remainder = line[len("MINOR:"):].strip()
            if remainder:
                minor_lines.append(f"MINOR: {remainder}")
            current_section = None
            continue

        if line.startswith("- "):
            item = line[2:].strip()
            if not item:
                continue
            if current_section == "MAJOR":
                lowered = item.lower()
                if not (
                    lowered.startswith("none") or lowered.startswith("no ") or lowered == "n/a"
                ):
                    major_lines.append(f"MAJOR: {item}")
            elif current_section == "MINOR":
                minor_lines.append(f"MINOR: {item}")

    return major_lines, minor_lines


def _build_blocking_checklist(test_output: str, review: str) -> list[str]:
    """Turn concrete test failures and MAJOR review items into a blocking checklist."""
    checklist: list[str] = []

    for line in test_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("FAIL:", "ERROR:")):
            checklist.append(stripped)

    major_lines, _ = _extract_review_items(review)
    checklist.extend(major_lines)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in checklist:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def _infer_related_source_file(test_path: str, workspace: Path) -> list[str]:
    """Map a failing test file to likely source files in the workspace."""
    path = Path(test_path)
    stem = path.stem
    if stem.startswith("test_"):
        candidate_stem = stem[len("test_"):]
        candidates = [
            workspace / f"{candidate_stem}.py",
            workspace / path.parent.parent / f"{candidate_stem}.py",
            workspace / "minesweeper" / f"{candidate_stem}.py",
        ]
        found = []
        for candidate in candidates:
            rel = candidate.resolve()
            try:
                rel.relative_to(workspace.resolve())
            except ValueError:
                continue
            if candidate.exists():
                found.append(str(candidate.relative_to(workspace)))
        return found
    return []


def _select_retry_files(code: str, test_output: str, review: str, workspace: Path) -> list[str]:
    """Choose a small set of files to rewrite on retry iterations.

    Selection should be based on the actual workspace, not only on the most recent
    implementer response, because retries often emit only a subset of files.
    """
    if workspace.exists():
        available = {
            str(path.relative_to(workspace))
            for path in workspace.rglob("*")
            if path.is_file() and path.suffix in {".py", ".txt", ".md", ".json", ".yaml", ".yml"}
        }
    else:
        available = {filename for filename, _ in _parse_files_from_response(code)}

    selected: list[str] = []

    for text in (test_output, review):
        for match in _PATH_RE.finditer(text):
            path = match.group("path")
            if path in available and path not in selected:
                selected.append(path)

    for path in list(selected):
        if Path(path).name.startswith("test_") or "tests" in Path(path).parts:
            for related in _infer_related_source_file(path, workspace):
                if related in available and related not in selected:
                    selected.append(related)

    if not selected and workspace.exists():
        py_files = sorted(
            str(path.relative_to(workspace))
            for path in workspace.rglob("*.py")
            if path.is_file()
        )
        preferred = [name for name in py_files if Path(name).name.startswith("test_") or "tests" in Path(name).parts]
        non_tests = [name for name in py_files if name not in preferred]
        selected = preferred[:3] + non_tests[:3]

    return selected[:6]


def _read_workspace_files(workspace: Path, filenames: list[str]) -> str:
    """Read a subset of workspace files back into prompt format."""
    blocks = []
    workspace_root = workspace.resolve()

    for filename in filenames:
        path = (workspace / filename).resolve()
        if workspace_root != path and workspace_root not in path.parents:
            continue
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        blocks.append(f"=== {filename} ===\n{content.rstrip()}\n")

    return "\n".join(blocks).strip()


def _build_issue_summary(
    workspace: Path,
    retry_files: list[str],
    test_output: str,
    review: str,
) -> str:
    """Build the retry payload for the next implementation attempt."""
    parts = ["Files to revise:\n" + "\n".join(f"- {name}" for name in retry_files)]

    blocking_checklist = _build_blocking_checklist(test_output, review)
    if blocking_checklist:
        parts.append(
            "Blocking checklist for this iteration:\n"
            + "\n".join(f"{i}. {item}" for i, item in enumerate(blocking_checklist, start=1))
            + "\n\nClear every item in this checklist before making any optional improvements."
        )

    file_block = _read_workspace_files(workspace, retry_files)
    if file_block:
        parts.append(f"Current file contents:\n{file_block}")

    if test_output.strip():
        parts.append(f"Test failures:\n{_summarize_test_output(test_output)}")

    major_lines, minor_lines = _extract_review_items(review)

    if major_lines:
        parts.append("Blocking review feedback:\n" + "\n".join(major_lines))
    if minor_lines:
        parts.append("Non-blocking review feedback:\n" + "\n".join(minor_lines))

    return "\n\n".join(parts).strip()


def plan_task(question: str, evidence: list[dict]) -> tuple[str, dict[str, str]]:
    """Ask the planner model for a compact implementation plan.

    The planner turns the user request plus retrieved evidence into a small
    structured plan covering files, modules, conventions, behaviour, and tests.
    """
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
        "2. Data structures / modules\n"
        "3. Conventions to follow\n"
        "4. Behaviour and flow\n"
        "5. Test strategy\n\n"
        "Requirements:\n"
        "- Extract concrete conventions from the retrieved evidence\n"
        "- Make the plan explicitly reflect retrieved file layout, style, CLI, and test conventions when present\n"
        "- Do not invent conventions that are not supported by the evidence\n"
        "- Assume Python for implementation\n"
        "- Use unittest from the Python standard library for generated tests\n"
        "- Make the tests discoverable by python -m unittest discover\n"
        "- Keep it compact and implementation-ready"
    )
    response = invoke_claude(system_prompt, user_prompt, max_tokens=900, temperature=0.0)
    return response, {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": response,
    }


def implement_task(
    question: str,
    evidence: list[dict],
    plan: str,
    *,
    issue_summary: str | None = None,
    retry_mode: bool = False,
) -> tuple[str, dict[str, str]]:
    """Ask the implementer model to generate the full multi-file code response.

    On the first iteration this uses only the task, evidence, and plan. On later
    iterations it includes the previous code together with compact test and
    review feedback so the model can revise the concrete artifact instead of
    regenerating from scratch.
    """
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

    if retry_mode:
        feedback_block = (
            "Revise only the files listed below. Preserve all other files in the "
            "workspace unchanged. Return only the updated files, using the same "
            "=== filename === format.\n\n"
            f"{issue_summary}\n\n"
        )
        generation_instruction = "Generate the updated files now.\n\n"
    else:
        feedback_block = ""
        generation_instruction = "Generate the full implementation now.\n\n"

    user_prompt = (
        f"Task: {question}\n\n"
        f"Retrieved evidence:\n{_format_evidence(evidence)}\n\n"
        f"Plan:\n{plan}\n\n"
        f"{feedback_block}"
        f"{generation_instruction}"
        "Hard requirements:\n"
        "- Python only\n"
        "- include or update unittest tests needed to validate the requested behaviour\n"
        "- use unittest only\n"
        "- do not use pytest, pytest.ini, setup.cfg, or pyproject-based test configuration\n"
        "- place tests either as top-level files named test_*.py or under tests/\n"
        "- tests must be discoverable by python -m unittest discover\n"
        "- output files using === filename === separators\n"
        "- do not include Markdown fences\n\n"
        "Retrieved evidence handling:\n"
        "- Treat retrieved style and conventions as binding implementation guidance when present\n"
        "- Prefer retrieved file layout, naming, rendering, CLI, and test conventions over generic defaults\n"
        "- Only depart from retrieved conventions if following them would violate the task requirements\n"
        "- Do not explain the conventions; just implement them\n"
        "- On retry iterations, prioritize clearing the blocking checklist before addressing any non-blocking improvements\n\n"
        "Output multiple files in one plain-text response using separators like:\n"
        "=== main.py ===\n"
        "...\n"
        "=== module.py ===\n"
        "...\n"
        "=== test_module.py ==="
    )
    response = invoke_claude(system_prompt, user_prompt, max_tokens=4500, temperature=0.0)
    return response, {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": response,
    }


def review_code(
    question: str,
    evidence: list[dict],
    code: str,
    *,
    test_output: str = "",
    tests_passed: bool = False,
) -> tuple[str, dict[str, str]]:
    """Ask the reviewer model for blocking vs non-blocking feedback.

    The reviewer is grounded with explicit runtime test facts so it does not
    invent contradictory claims about whether tests ran or passed.
    """
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
        "This workflow uses unittest only.\n"
        "Do not recommend pytest, pytest.ini, setup.cfg, or third-party test runners.\n"
        "Do not infer or guess test execution facts. Use the authoritative runtime test facts below as ground truth.\n"
        "Only mark something as MAJOR if it is a blocking defect relative to the explicit task requirements or retrieved evidence.\n"
        "Use MINOR for smaller defects, polish issues, non-blocking gaps, or quality improvements that should still influence a later revision.\n"
        "Do not mark optional enhancements, nice-to-haves, future improvements, or speculative features as MAJOR.\n\n"
        f"{_build_test_status(test_output, tests_passed)}\n\n"
        "Review the generated implementation. Focus on:\n"
        "- correctness\n"
        "- adherence to retrieved style and conventions\n"
        "- mismatches with retrieved requirements or behaviour\n"
        "- edge cases\n"
        "- structure and readability\n"
        "- test coverage for required behaviour\n\n"
        "Return short bullet points only. Prefix each bullet with one of:\n"
        "- PASS:\n"
        "- MINOR:\n"
        "- MAJOR:\n\n"
        f"Raw test output:\n{test_output or 'No test output.'}\n\n"
        f"Code:\n{code}"
    )
    response = invoke_claude(system_prompt, user_prompt, max_tokens=1000, temperature=0.0)
    return response, {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": response,
    }


def run_workflow(
    question: str,
    use_retrieval: bool = True,
) -> dict[str, object]:
    """Execute the full iterative coding workflow."""
    evidence = retrieve(question) if use_retrieval else []
    plan, plan_trace = plan_task(question, evidence)

    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    code = ""
    review = ""
    issue_summary: str | None = None
    retry_files: list[str] = []
    iterations: list[dict[str, object]] = []
    stop_reason = "max_iterations_reached"

    for iteration in range(1, MAX_ITERS + 1):
        retry_mode = iteration > 1
        code, implement_trace = implement_task(
            question,
            evidence,
            plan,
            issue_summary=issue_summary,
            retry_mode=retry_mode,
        )

        workspace_snapshot = ""
        blocking_checklist: list[str] = []
        if issue_summary is not None:
            blocking_checklist = _build_blocking_checklist("", review)

        try:
            if retry_mode:
                apply_file_updates_from_response(code, WORKSPACE_ROOT)
            else:
                write_files_from_response(code, WORKSPACE_ROOT)

            tests_passed, test_output = run_tests(WORKSPACE_ROOT)
            workspace_snapshot = _read_workspace_files(
                WORKSPACE_ROOT,
                sorted(
                    str(path.relative_to(WORKSPACE_ROOT))
                    for path in WORKSPACE_ROOT.rglob("*.py")
                    if path.is_file()
                ),
            )
            review, review_trace = review_code(
                question,
                evidence,
                workspace_snapshot,
                test_output=test_output,
                tests_passed=tests_passed,
            )
            blocking_checklist = _build_blocking_checklist(test_output, review)
        except ValueError as exc:
            tests_passed = False
            test_output = f"File emission validation failed:\n{exc}"
            review = f"MAJOR: {exc}"
            review_trace = {
                "system_prompt": "",
                "user_prompt": "",
                "response": review,
            }
            blocking_checklist = _build_blocking_checklist(test_output, review)

        has_major_issues = major_issues(review)

        iteration_record: dict[str, object] = {
            "iteration": iteration,
            "retry_mode": retry_mode,
            "tests_passed": tests_passed,
            "major_issues": has_major_issues,
            "test_output": test_output,
            "review": review,
            "retry_files": retry_files,
            "issue_summary": issue_summary,
            "blocking_checklist": blocking_checklist,
            "workspace_snapshot": workspace_snapshot,
            "implement_output": code,
            "trace": {
                "implement": implement_trace,
                "review": review_trace,
            },
        }

        iterations.append(iteration_record)

        if tests_passed and not has_major_issues:
            stop_reason = "tests_passed_and_review_clean"
            break

        retry_files = _select_retry_files(
            _read_workspace_files(
                WORKSPACE_ROOT,
                sorted(
                    str(path.relative_to(WORKSPACE_ROOT))
                    for path in WORKSPACE_ROOT.rglob("*.py")
                    if path.is_file()
                ),
            ),
            test_output,
            review,
            WORKSPACE_ROOT,
        )
        issue_summary = _build_issue_summary(WORKSPACE_ROOT, retry_files, test_output, review)

    final_code = _read_workspace_files(
        WORKSPACE_ROOT,
        sorted(
            str(path.relative_to(WORKSPACE_ROOT))
            for path in WORKSPACE_ROOT.rglob("*.py")
            if path.is_file()
        ),
    )

    response: dict[str, object] = {
        "evidence": evidence,
        "plan": plan,
        "code": final_code,
        "review": review,
        "iterations": iterations,
        "completed_iteration": len(iterations),
        "stop_reason": stop_reason,
        "trace": {
            "plan": plan_trace,
        },
    }

    return response
