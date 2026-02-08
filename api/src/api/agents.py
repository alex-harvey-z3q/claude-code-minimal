from __future__ import annotations

from .llm import invoke_claude
from .retrieval import retrieve


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


def implement_task(question: str, evidence: list[dict], plan: str) -> str:
    system_prompt = (
        "You are Implementer, a Python coding agent. Generate a complete, runnable, "
        "terminal-based Python application. When retrieved evidence contains "
        "implementation conventions, style guidance, file layout guidance, testing "
        "guidance, naming guidance, or CLI conventions, you must follow that guidance "
        "unless it conflicts with the user's explicit requirements. Do not silently "
        "replace retrieved conventions with your own defaults. Use retrieved domain "
        "evidence for gameplay rules and behaviour. Output only code files using the "
        "exact separator format === filename ===."
    )
    user_prompt = (
        f"Task: {question}\n\n"
        f"Retrieved evidence:\n{_format_evidence(evidence)}\n\n"
        f"Plan:\n{plan}\n\n"
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


def review_code(question: str, evidence: list[dict], code: str) -> str:
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
        "- test coverage\n\n"
        "Return short bullet points only. Call out specific convention violations when present.\n\n"
        f"Code:\n{code}"
    )
    return invoke_claude(system_prompt, user_prompt, max_tokens=1000, temperature=0.0)


def run_workflow(question: str, use_retrieval: bool = True) -> dict[str, object]:
    evidence = retrieve(question) if use_retrieval else []
    plan = plan_task(question, evidence)
    code = implement_task(question, evidence, plan)
    review = review_code(question, evidence, code)
    return {
        "evidence": evidence,
        "plan": plan,
        "code": code,
        "review": review,
    }
