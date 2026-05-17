from fastapi import FastAPI, HTTPException, Query

from .config import DEFAULT_USE_RETRIEVAL
from .agents import WorkflowExecutionError, run_workflow
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
    try:
        return run_workflow(q, use_retrieval=use_retrieval)
    except WorkflowExecutionError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": str(exc),
                "workspace_id": exc.workspace_id,
                "trace_file": exc.trace_file,
                "debug": exc.debug,
            },
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
