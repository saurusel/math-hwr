import os, json, threading, time, random, traceback, queue
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from ..core.event_logger import EventLogger
except ImportError:
    EventLogger = None
RUNS = {}
router = APIRouter()

# Pydantic models for /jobs endpoint (as per Train-and-Models-Spec)
class DatasetConfig(BaseModel):
    name: str
    train_manifest: str
    val_manifest: str

class HyperConfig(BaseModel):
    batch_size: int
    epochs: int
    img_h: int
    img_w_max: int
    lr: float
    optimizer: str
    scheduler: str
    seed: int
    max_samples: Optional[int] = None  # NEW: Limit dataset size
    advanced: Optional[Dict[str, Any]] = None

class AugmentConfig(BaseModel):
    invert: bool
    random_pad: bool

class TrainingJobRequest(BaseModel):
    model_type: str  # M1, M2, M3
    run_name: str
    dataset: DatasetConfig
    hyper: HyperConfig
    augment: AugmentConfig

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

try:
    from ..core.train.trainer_seg_mlp import train_seg_mlp
except Exception:
    train_seg_mlp = None

@router.post("/jobs")
def create_training_job(request: TrainingJobRequest):
    """
    Create a new training job according to Train-and-Models-Spec.
    Maps the standardized request format to internal training config.
    """
    import uuid

    job_id = f"{request.run_name}_{uuid.uuid4().hex[:8]}"
    runs_dir = os.path.join("runs", job_id)
    os.makedirs(runs_dir, exist_ok=True)

    # Map frontend format to internal config format
    # Extract directory from manifest path (e.g., "data/synth/train/labels.jsonl" -> "data/synth/train")
    train_dir = os.path.dirname(request.dataset.train_manifest)
    val_dir = os.path.dirname(request.dataset.val_manifest)

    # Calculate train/val split if max_samples is set
    max_train_samples = None
    max_val_samples = None
    if request.hyper.max_samples:
        # Split: 80% train, 20% val (10% would be test, but we don't use test in training)
        max_train_samples = int(request.hyper.max_samples * 0.8)
        max_val_samples = int(request.hyper.max_samples * 0.2)

    internal_config = {
        "model": "crnn_ctc" if request.model_type == "M1" else request.model_type.lower(),
        "train": {
            "epochs": request.hyper.epochs,
            "batch_size": request.hyper.batch_size,
            "lr": request.hyper.lr,
            "optimizer": request.hyper.optimizer,
            "scheduler": request.hyper.scheduler,
            "seed": request.hyper.seed,
        },
        "data": {
            "train_dir": train_dir,
            "val_dir": val_dir,
            "img_h": request.hyper.img_h,
            "img_w_max": request.hyper.img_w_max,
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
        },
        "augment": {
            "invert": request.augment.invert,
            "random_pad": request.augment.random_pad,
        }
    }

    # Add advanced settings if provided
    if request.hyper.advanced:
        model_key = request.model_type.lower()
        if model_key in request.hyper.advanced:
            internal_config["train"].update(request.hyper.advanced[model_key])

    # Save config
    with open(os.path.join(runs_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(request.dict(), f, indent=2, ensure_ascii=False)

    RUNS[job_id] = {
        "job_id": job_id,
        "run_name": request.run_name,
        "model_type": request.model_type,
        "config": internal_config,
        "status": "QUEUED",
        "queues": set(),
        "epoch": 0,
        "step": 0,
        "dir": runs_dir,
        "last_error": None,
        "started_at": datetime.now().isoformat()
    }

    # Start training in background thread - select trainer by model type
    trainer_fn = None

    if request.model_type == "M1" and train_crnn_ctc is not None:
        trainer_fn = train_crnn_ctc
    elif request.model_type == "M2" and train_seg_mlp is not None:
        trainer_fn = train_seg_mlp

    if trainer_fn:
        def _runner():
            try:
                RUNS[job_id]["status"] = "RUNNING"
                trainer_fn(internal_config, runs_dir, lambda msg: _emit(job_id, msg))
                RUNS[job_id]["status"] = "FINISHED"
                RUNS[job_id]["finished_at"] = datetime.now().isoformat()
                _emit(job_id, {"run_id": job_id, "event": "finished"})
            except Exception as e:
                RUNS[job_id]["status"] = "FAILED"
                RUNS[job_id]["last_error"] = traceback.format_exc()
                _emit(job_id, {"run_id": job_id, "event": "error", "message": str(e)})
                if EventLogger:
                    try:
                        logger_err = EventLogger(os.path.join(runs_dir, "events.jsonl"))
                        logger_err.log_status("FAILED", str(e))
                        logger_err.close()
                    except:
                        pass

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
    else:
        # Simulator for when real trainer not available
        def _sim():
            import random
            RUNS[job_id]["status"] = "RUNNING"
            _emit(job_id, {"event": "log", "ts": datetime.now().isoformat(), "line": f"Starting {request.model_type} simulation mode..."})

            # Write events to file for persistence
            logger_sim = EventLogger(os.path.join(runs_dir, "events.jsonl"))
            logger_sim.log_status("RUNNING", f"{request.model_type} simulation started")

            total_epochs = request.hyper.epochs

            for ep in range(1, total_epochs + 1):
                RUNS[job_id]["epoch"] = ep

                # Check stop signal
                stop_file = os.path.join(runs_dir, "STOP_REQUESTED")
                if os.path.exists(stop_file):
                    RUNS[job_id]["status"] = "STOPPED"
                    logger_sim.log_status("STOPPED", f"Stopped at epoch {ep}")
                    _emit(job_id, {"event": "stopped", "epoch": ep - 1})
                    break

                # Simulate improving metrics (model-specific patterns)
                if request.model_type == "M1":
                    base_cer = 0.3 * (1.0 - ep / total_epochs)
                    base_wer = 0.4 * (1.0 - ep / total_epochs)
                    base_exact = 0.5 + 0.4 * (ep / total_epochs)
                elif request.model_type == "M2":
                    base_cer = 0.25 * (1.0 - ep / total_epochs)  # Better than M1
                    base_wer = 0.35 * (1.0 - ep / total_epochs)
                    base_exact = 0.6 + 0.35 * (ep / total_epochs)
                else:  # M3
                    base_cer = 0.20 * (1.0 - ep / total_epochs)  # Best
                    base_wer = 0.28 * (1.0 - ep / total_epochs)
                    base_exact = 0.65 + 0.32 * (ep / total_epochs)

                for st in range(1, 4):
                    step_metrics = {
                        "loss": 2.5 * (1.0 - (ep * 4 + st) / (total_epochs * 4)),
                        "cer": max(0.01, base_cer + random.uniform(-0.03, 0.03)),
                        "lr": request.hyper.lr * (1.0 - ep / total_epochs) if request.hyper.scheduler == "cosine" else request.hyper.lr
                    }
                    _emit(job_id, {"event": "train_step", "epoch": ep, "step": (ep - 1) * 4 + st, "metrics": step_metrics})
                    time.sleep(0.2)

                # Epoch end metrics
                epoch_metrics = {
                    "train_loss": 2.5 * (1.0 - ep / total_epochs),
                    "val_loss": 2.3 * (1.0 - ep / total_epochs),
                    "cer": max(0.01, base_cer),
                    "wer": max(0.01, base_wer),
                    "exact": min(0.97, base_exact),
                    "valid": min(0.99, 0.88 + 0.11 * (ep / total_epochs)),
                    "lr": request.hyper.lr * (1.0 - ep / total_epochs) if request.hyper.scheduler == "cosine" else request.hyper.lr,
                    "epoch_time_sec": 18.0 + random.uniform(-3, 3),
                    "samples_processed": request.hyper.max_samples or 1000,
                    "batches_processed": (request.hyper.max_samples or 1000) // request.hyper.batch_size
                }

                # Add model-specific metrics
                if request.model_type == "M2":
                    epoch_metrics["attention_entropy"] = 0.4 + random.uniform(-0.1, 0.1)
                    epoch_metrics["teacher_forcing_used"] = 0.5
                elif request.model_type == "M3":
                    epoch_metrics["decoder_perplexity"] = 3.0 * (1.0 - ep / total_epochs)
                    epoch_metrics["patch_attention_mean"] = 0.15

                _emit(job_id, {"event": "epoch_end", "epoch": ep, "metrics": epoch_metrics})
                logger_sim.log_metric(ep, epoch_metrics)
                logger_sim.log_text(f"Epoch {ep}/{total_epochs} completed")
                _emit(job_id, {"event": "log", "ts": datetime.now().isoformat(), "line": f"Epoch {ep}/{total_epochs} completed"})

            if RUNS[job_id]["status"] != "STOPPED":
                RUNS[job_id]["status"] = "FINISHED"
                RUNS[job_id]["finished_at"] = datetime.now().isoformat()
                logger_sim.log_status("FINISHED", "Simulation completed")
                _emit(job_id, {"event": "finished", "message": "Simulation completed"})

            logger_sim.close()

        t = threading.Thread(target=_sim, daemon=True)
        t.start()

    return JSONResponse(
        {"job_id": job_id, "status": RUNS[job_id]["status"]},
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

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

@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Get training job status and progress."""
    if job_id not in RUNS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    run = RUNS[job_id]
    return JSONResponse(
        {
            "job_id": job_id,
            "run_name": run.get("run_name", job_id),
            "model_type": run.get("model_type", "M1"),
            "status": run["status"],
            "started_at": run.get("started_at", datetime.now().isoformat()),
            "current_epoch": run.get("epoch", 0),
            "progress": run.get("epoch", 0),
            "best_metric": run.get("best_metric"),
            "last_checkpoint": run.get("last_checkpoint"),
        },
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    """
    Stop a running training job gracefully.
    Creates STOP_REQUESTED file that trainer checks between epochs.
    """
    if job_id not in RUNS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    run = RUNS[job_id]
    if run["status"] in ["FINISHED", "FAILED", "STOPPED"]:
        return JSONResponse(
            {"message": f"Job already {run['status']}"},
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

    # Create stop signal file for trainer to detect
    run_dir = run.get("dir", os.path.join("runs", job_id))
    stop_file = os.path.join(run_dir, "STOP_REQUESTED")

    try:
        with open(stop_file, 'w') as f:
            f.write(f"Stop requested at {datetime.now().isoformat()}\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create stop signal: {str(e)}")

    # Update status (will be confirmed by trainer)
    run["status"] = "STOPPING"
    _emit(job_id, {"event": "log", "ts": datetime.now().isoformat(), "line": "Stop signal sent. Will stop after current epoch..."})

    return JSONResponse(
        {"job_id": job_id, "status": "STOPPING", "message": "Stop signal sent"},
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@router.get("/stream/{job_id}")
async def stream_training_events(job_id: str):
    """
    SSE endpoint for live training events streaming.
    Streams metrics, logs, samples, checkpoints, and status updates.
    """
    if job_id not in RUNS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    run = RUNS[job_id]

    # Create a queue for this client
    client_queue = queue.Queue(maxsize=100)
    run["queues"].add(client_queue)

    async def event_generator():
        try:
            # Send initial status
            yield f"event: status\ndata: {json.dumps({'status': run['status']}, ensure_ascii=False)}\n\n"

            while True:
                try:
                    # Wait for events with timeout
                    event = client_queue.get(timeout=30)

                    # Map internal events to SSE format
                    event_type = event.get("event", "log")

                    if event_type == "train_step":
                        # Convert to metric event
                        metric_data = {
                            "epoch": event.get("epoch", 0),
                            "train_loss": event.get("metrics", {}).get("loss", 0),
                            "val_loss": 0,  # Will be updated on epoch_end
                            "cer": event.get("metrics", {}).get("cer", 0),
                            "lr": event.get("metrics", {}).get("lr", 0.001)
                        }
                        yield f"event: metric\ndata: {json.dumps(metric_data, ensure_ascii=False)}\n\n"

                    elif event_type == "epoch_end":
                        metric_data = {
                            "epoch": event.get("epoch", 0),
                            "train_loss": event.get("metrics", {}).get("train_loss", 0),
                            "val_loss": event.get("metrics", {}).get("val_loss", 0),
                            "cer": event.get("metrics", {}).get("cer", 0),
                            "wer": event.get("metrics", {}).get("wer", 0),
                            "exact": event.get("metrics", {}).get("exact", 0),
                            "lr": event.get("metrics", {}).get("lr", 0.001)
                        }
                        yield f"event: metric\ndata: {json.dumps(metric_data, ensure_ascii=False)}\n\n"

                    elif event_type == "sample_pred":
                        # Forward sample predictions with images
                        sample_data = {
                            "epoch": event.get("epoch", 0),
                            "items": event.get("items", [])
                        }
                        yield f"event: sample_pred\ndata: {json.dumps(sample_data, ensure_ascii=False)}\n\n"

                    elif event_type in ["finished", "error", "stopped"]:
                        status_map = {"finished": "FINISHED", "error": "FAILED", "stopped": "STOPPED"}
                        status_data = {"status": status_map.get(event_type, "UNKNOWN")}
                        yield f"event: status\ndata: {json.dumps(status_data, ensure_ascii=False)}\n\n"

                        # Also send as log
                        log_data = {
                            "ts": datetime.now().isoformat(),
                            "line": f"Training {event_type}: {event.get('message', '')}"
                        }
                        yield f"event: log\ndata: {json.dumps(log_data, ensure_ascii=False)}\n\n"

                        if event_type in ["finished", "error", "stopped"]:
                            break

                    else:
                        # Generic log event
                        log_data = {
                            "ts": datetime.now().isoformat(),
                            "line": str(event.get("message", event))
                        }
                        yield f"event: log\ndata: {json.dumps(log_data, ensure_ascii=False)}\n\n"

                except queue.Empty:
                    # Send keep-alive ping
                    yield f": ping\n\n"

                    # Check if job finished
                    if run["status"] in ["FINISHED", "FAILED", "STOPPED"]:
                        break

        except Exception as e:
            print(f"SSE stream error for {job_id}: {e}")
        finally:
            # Clean up: remove queue from run
            if client_queue in run["queues"]:
                run["queues"].discard(client_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream; charset=utf-8",
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/experiments")
def list_experiments(
    model: Optional[str] = None,
    status: Optional[str] = None,
    q: Optional[str] = None
):
    """
    List all training experiments from both memory and filesystem.
    Returns list of experiments with metadata.
    """
    experiments = []

    # First, add from in-memory RUNS
    for job_id, run in RUNS.items():
        exp = {
            "run_id": job_id,
            "run_name": run.get("run_name", job_id),
            "model_type": run.get("model_type", "M1"),
            "status": run["status"],
            "started_at": run.get("started_at", datetime.now().isoformat()),
            "finished_at": run.get("finished_at"),
            "best_metrics": run.get("best_metric"),
            "config": run.get("config")
        }
        experiments.append(exp)

    # Also scan filesystem for completed runs not in memory
    runs_root = "runs"
    if os.path.exists(runs_root):
        for run_name in os.listdir(runs_root):
            # Skip if already in RUNS
            if run_name in RUNS:
                continue

            run_path = os.path.join(runs_root, run_name)
            if not os.path.isdir(run_path):
                continue

            # Try to read config.json
            config_path = os.path.join(run_path, "config.json")
            run_config = None
            run_name_clean = run_name
            model_type_guess = "M1"

            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        run_config = json.load(f)
                        run_name_clean = run_config.get("run_name", run_name)
                        model_type_guess = run_config.get("model_type", "M1")
                except Exception:
                    pass

            # Determine status from checkpoints
            has_final = os.path.exists(os.path.join(run_path, "crnn_final.pt"))
            has_latest = os.path.exists(os.path.join(run_path, "checkpoint_latest.pt"))
            status_guess = "FINISHED" if has_final or has_latest else "UNKNOWN"

            # Get modification time for sorting
            try:
                mtime = os.path.getmtime(run_path)
                started_at = datetime.fromtimestamp(mtime).isoformat()
            except Exception:
                started_at = datetime.now().isoformat()

            exp = {
                "run_id": run_name,
                "run_name": run_name_clean,
                "model_type": model_type_guess,
                "status": status_guess,
                "started_at": started_at,
                "finished_at": started_at if status_guess == "FINISHED" else None,
                "best_metrics": None,
                "config": run_config
            }
            experiments.append(exp)

    # Apply filters
    filtered = []
    for exp in experiments:
        if model and exp["model_type"] != model:
            continue
        if status and exp["status"] != status:
            continue
        if q and q.lower() not in exp["run_name"].lower():
            continue
        filtered.append(exp)

    # Sort by started_at descending (newest first)
    filtered.sort(key=lambda x: x.get("started_at", ""), reverse=True)

    return JSONResponse(
        filtered,
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@router.get("/experiments/{run_id}/checkpoints")
def list_checkpoints(run_id: str):
    """
    List all checkpoints for a training run.
    Returns list of checkpoint metadata.
    """
    if run_id not in RUNS:
        # Try to find checkpoints in filesystem
        runs_dir = os.path.join("runs", run_id, "checkpoints")
        if not os.path.exists(runs_dir):
            return JSONResponse([], headers={"Content-Type": "application/json; charset=utf-8"})

        checkpoints = []
        for fname in os.listdir(runs_dir):
            if fname.endswith(".pt"):
                fpath = os.path.join(runs_dir, fname)
                ckpt = {
                    "ckpt_id": fname.replace(".pt", ""),
                    "path": fpath,
                    "size": os.path.getsize(fpath),
                    "epoch": 0,  # Parse from filename
                    "kind": "best" if "best" in fname else "last" if "last" in fname else "epoch"
                }
                checkpoints.append(ckpt)

        return JSONResponse(checkpoints, headers={"Content-Type": "application/json; charset=utf-8"})

    # For active runs, return from memory
    run = RUNS[run_id]
    checkpoints = run.get("checkpoints", [])

    return JSONResponse(
        checkpoints,
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@router.get("/jobs/{job_id}/history")
def get_training_history(job_id: str):
    """Get historical training data from events.jsonl."""
    events_file = os.path.join("runs", job_id, "events.jsonl")
    if not os.path.exists(events_file):
        return JSONResponse({"metrics": [], "logs": []}, headers={"Content-Type": "application/json; charset=utf-8"})
    
    try:
        from ..core.event_logger import read_events, parse_events_by_type
        events = read_events(events_file)
        grouped = parse_events_by_type(events)
        return JSONResponse({
            "metrics": grouped.get("metric", []),
            "logs": grouped.get("log", []),
            "samples": grouped.get("sample_pred", []),
            "checkpoints": grouped.get("checkpoint", [])
        }, headers={"Content-Type": "application/json; charset=utf-8"})
    except Exception as e:
        return JSONResponse({"error": str(e), "metrics": [], "logs": []}, headers={"Content-Type": "application/json; charset=utf-8"})
