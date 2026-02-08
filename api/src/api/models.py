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


class WorkflowResponse(BaseModel):
    evidence: list[EvidenceItem]
    plan: str
    code: str
    review: str
