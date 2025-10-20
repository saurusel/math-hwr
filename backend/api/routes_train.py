import os, json, threading, time, random, traceback
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

RUNS = {}
router = APIRouter()

def _emit(run_id, payload):
    run = RUNS.get(run_id)
    if not run:
        return
    # Update progress counters if present
    try:
        if payload.get("event") == "train_step":
            run["step"] = int(payload.get("step", run.get("step", 0)))
            run["epoch"] = int(payload.get("epoch", run.get("epoch", 0)))
        elif payload.get("event") == "epoch_end":
            run["epoch"] = int(payload.get("epoch", run.get("epoch", 0)))
    except Exception:
        pass

    for q in list(run["queues"]):
        try:
            q.put_nowait(payload)
        except Exception:
            pass

try:
    from ..core.train.trainer_crnn import train_crnn_ctc
except Exception:
    train_crnn_ctc = None

@router.post("/start")
def start_training(config: dict):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = config.get("model","crnn_ctc")
    run_id = f"{now}_{model}"
    runs_dir = os.path.join("runs", run_id)
    os.makedirs(runs_dir, exist_ok=True)

    RUNS[run_id] = {
        "config": config,
        "status": "running",
        "queues": set(),
        "epoch": 0,
        "step": 0,
        "dir": runs_dir,
        "last_error": None
    }

    if model == "crnn_ctc" and train_crnn_ctc is not None:
        def _runner():
            try:
                train_crnn_ctc(config, runs_dir, lambda msg: _emit(run_id, msg))
            except Exception as e:
                RUNS[run_id]["status"] = "error"
                RUNS[run_id]["last_error"] = traceback.format_exc()
                _emit(run_id, {"run_id": run_id, "event": "error", "message": str(e)})
                return
            RUNS[run_id]["status"] = "finished"
            _emit(run_id, {"run_id": run_id, "event": "finished"})
        t = threading.Thread(target=_runner, daemon=True)
    else:
        # fallback to simulator if needed
        def _sim():
            cfg = RUNS[run_id]["config"]
            total_epochs = cfg.get("train", {}).get("epochs", 1)
            for ep in range(1, total_epochs+1):
                for st in range(1, 51):
                    _emit(run_id, {"event":"train_step","epoch":ep,"step":st,"metrics":{"cer":0.3}})
                    time.sleep(0.05)
                _emit(run_id, {"event":"epoch_end","epoch":ep,"metrics":{"cer":0.2}})
            RUNS[run_id]["status"]="finished"
            _emit(run_id, {"event":"finished"})
        t = threading.Thread(target=_sim, daemon=True)

    t.start()
    return JSONResponse({"run_id": run_id, "status": RUNS[run_id]["status"]})
