from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from .routes_train import RUNS
import os
import json

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

@router.get("/runs/{run_id}")
def run_detail(run_id: str):
    r = RUNS.get(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    # return minimal safe info
    return {
        "run_id": run_id,
        "status": r["status"],
        "epoch": r.get("epoch", 0),
        "step": r.get("step", 0),
        "last_error": r.get("last_error")
    }

@router.get("/{run_id}/history")
def get_training_history(run_id: str):
    """
    Load complete training history from events.jsonl
    Works for FINISHED, STOPPED, or FAILED runs
    """
    events_file = f"runs/{run_id}/events.jsonl"

    if not os.path.exists(events_file):
        raise HTTPException(404, detail=f"No training history found for {run_id}")

    metrics = []
    logs = []
    samples = []

    try:
        with open(events_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    event = json.loads(line)

                    # Metric events (epoch_end)
                    if event.get("event") == "epoch_end":
                        metrics.append({
                            "epoch": event["epoch"],
                            **event["metrics"]
                        })

                    # Log events
                    elif event.get("type") == "log":
                        logs.append({
                            "ts": event.get("timestamp", ""),
                            "line": event.get("message", "")
                        })

                    # Sample prediction events (check both "event" and "type")
                    elif event.get("event") == "sample_pred" or event.get("type") == "sample_pred":
                        samples.append({
                            "epoch": event.get("epoch", 0),
                            "items": event.get("items", [])
                        })

                except json.JSONDecodeError:
                    continue

        return JSONResponse(
            {
                "metrics": metrics,
                "logs": logs,
                "samples": samples
            },
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

    except Exception as e:
        raise HTTPException(500, detail=f"Failed to load history: {str(e)}")

@router.get("/checkpoints")
def list_checkpoints():
    out = []
    for rid, r in RUNS.items():
        out.extend([
            {"run_id": rid, "path": f"runs/{rid}/checkpoint_epoch_{e}.pt"}
            for e in range(1, r.get("epoch", 0)+1)
        ])
    return {"checkpoints": out}
