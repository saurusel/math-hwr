import os, json, threading, time, random, asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

RUNS = {}
router = APIRouter()

def _emit(run_id, payload):
    run = RUNS.get(run_id)
    if not run: return
    for q in list(run["queues"]):
        try:
            q.put_nowait(payload)
        except Exception:
            pass

def _training_simulator(run_id: str):
    run = RUNS[run_id]
    cfg = run["config"]
    total_epochs = cfg.get("train", {}).get("epochs", 5)
    steps_per_epoch = 50
    cer = 0.35
    wer = 0.45
    exact = 0.1
    tree_f1 = 0.3

    for ep in range(1, total_epochs+1):
        run["epoch"] = ep
        for st in range(1, steps_per_epoch+1):
            if run["status"] == "stopped":
                _emit(run_id, {"run_id": run_id, "event":"finished", "epoch": ep, "step": st})
                return
            while run["status"] == "paused":
                time.sleep(0.3)

            run["step"] = st
            # fake metric drift
            cer = max(0.05, cer - random.uniform(0.001, 0.01))
            wer = max(0.08, wer - random.uniform(0.001, 0.01))
            exact = min(0.95, exact + random.uniform(0.002, 0.015))
            tree_f1 = min(0.95, tree_f1 + random.uniform(0.002, 0.01))

            msg = {
                "run_id": run_id,
                "event": "train_step",
                "epoch": ep,
                "step": st,
                "lr": 0.001,
                "metrics": {
                    "loss": round(1.5*cer + 0.2*wer, 4),
                    "cer": round(cer, 4),
                    "wer": round(wer, 4),
                    "exact": round(exact, 4),
                    "tree_f1": round(tree_f1, 4),
                    "valid": round(min(1.0, 0.6 + tree_f1*0.4), 4)
                },
                "gpu": {"mem_gb": 4.2 + random.random(), "util": int(50 + 40*random.random()), "temp_c": 60 + int(10*random.random())},
                "samples": [
                    {
                      "png_path": f"runs/{run_id}/samples/e{ep}_s{st}.png",
                      "target": "frac ( { x + 1 } , { y - 2 } )",
                      "pred":   "frac ( { x + 1 } , { y - 2 } )",
                      "ok": True
                    }
                ]
            }
            _emit(run_id, msg)
            time.sleep(0.15)  # simulate work

        # epoch end
        _emit(run_id, {"run_id": run_id, "event": "epoch_end", "epoch": ep, "metrics": {"cer": round(cer,4), "wer": round(wer,4), "exact": round(exact,4), "tree_f1": round(tree_f1,4)}})

    RUNS[run_id]["status"] = "finished"
    _emit(run_id, {"run_id": run_id, "event":"finished", "epoch": total_epochs})

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
        "dir": runs_dir
    }
    t = threading.Thread(target=_training_simulator, args=(run_id,), daemon=True)
    t.start()
    return JSONResponse({"run_id": run_id, "status": "running"})

@router.post("/pause")
def pause(run_id: str):
    if run_id not in RUNS: raise HTTPException(404, "run not found")
    RUNS[run_id]["status"] = "paused"
    return {"ok": True}

@router.post("/resume")
def resume(run_id: str):
    if run_id not in RUNS: raise HTTPException(404, "run not found")
    RUNS[run_id]["status"] = "running"
    return {"ok": True}

@router.post("/stop")
def stop(run_id: str):
    if run_id not in RUNS: raise HTTPException(404, "run not found")
    RUNS[run_id]["status"] = "stopped"
    return {"ok": True}
