from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class EvidenceItem(BaseModel):
    page: str
    section: str
    url: str
    revision_id: int | None = None
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    evidence: list[EvidenceItem]


class AgentTrace(BaseModel):
    system_prompt: str
    user_prompt: str
    response: str
    tool_calls: str | None = None
    tool_loop: dict[str, Any] | None = None
    tool_rounds: list[dict[str, Any]] | None = None


class IterationTrace(BaseModel):
    implement: AgentTrace
    review: AgentTrace


class WorkflowTrace(BaseModel):
    plan: AgentTrace


class IterationInfo(BaseModel):
    iteration: int
    retry_mode: bool
    tests_passed: bool
    major_issues: bool
    test_output: str
    review: str
    retry_files: list[str]
    issue_summary: str | None
    blocking_checklist: list[str]
    workspace_snapshot: str
    implement_output: str
    trace: IterationTrace


class WorkflowResponse(BaseModel):
    evidence: list[EvidenceItem]
    plan: str
    code: str
    workspace_id: str | None = None
    review: str
    iterations: list[IterationInfo]
    completed_iteration: int
    stop_reason: str
    observability: dict[str, Any] | None = None
    trace: WorkflowTrace
