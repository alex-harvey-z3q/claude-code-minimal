from fastapi import FastAPI, Query

from .config import DEFAULT_USE_RETRIEVAL
from .agents import run_workflow
from .models import WorkflowResponse

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/query", response_model=WorkflowResponse)
def query(
    q: str = Query(..., min_length=1, max_length=2000),
    use_retrieval: bool = Query(DEFAULT_USE_RETRIEVAL),
):
    return run_workflow(q, use_retrieval=use_retrieval)
