from __future__ import annotations

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


class IterationInfo(BaseModel):
    iteration: int
    tests_passed: bool
    major_issues: bool
    test_output: str
    review: str


class WorkflowResponse(BaseModel):
    evidence: list[EvidenceItem]
    plan: str
    code: str
    review: str
    iterations: list[IterationInfo]
    completed_iteration: int
    stop_reason: str
