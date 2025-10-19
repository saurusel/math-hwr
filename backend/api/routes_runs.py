from fastapi import APIRouter
from .routes_train import RUNS

router = APIRouter()

@router.get("/runs")
def list_runs():
    out = []
    for rid, r in RUNS.items():
        out.append({
            "run_id": rid,
            "model": r["config"]["model"],
            "status": r["status"],
            "epochs_done": r.get("epoch", 0),
            "steps_done": r.get("step", 0),
        })
    return {"runs": out}

@router.get("/checkpoints")
def list_checkpoints():
    out = []
    for rid, r in RUNS.items():
        out.extend([
            {"run_id": rid, "path": f"runs/{rid}/checkpoint_epoch_{e}.pt"}
            for e in range(1, r.get("epoch", 0)+1)
        ])
    return {"checkpoints": out}
