
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os, json, time
import torch

from ..core.eval.eval_run import evaluate_crnn_ctc

router = APIRouter(prefix="/api/eval", tags=["eval"])

class EvalDataCfg(BaseModel):
    val_dir: str
    img_h: int = 64
    img_w_max: int = 512

class EvalCfg(BaseModel):
    batch_size: int = 64
    max_batches: Optional[int] = None  # for sanity runs

class EvalRequest(BaseModel):
    run_id: str
    model: str = "crnn_ctc"
    data: EvalDataCfg
    eval: EvalCfg = EvalCfg()
    checkpoint: Optional[str] = None  # path to .pt or one of ['crnn_final.pt','checkpoint_latest.pt']

@router.post("/run")
def eval_run(req: EvalRequest):
    run_dir = os.path.join("runs", req.run_id)
    if not os.path.isdir(run_dir):
        raise HTTPException(404, f"run_dir not found: {run_dir}")

    # resolve checkpoint
    ckpt_path = req.checkpoint
    if ckpt_path is None:
        # prefer final, then latest
        c1 = os.path.join(run_dir, "crnn_final.pt")
        c2 = os.path.join(run_dir, "checkpoint_latest.pt")
        ckpt_path = c1 if os.path.exists(c1) else (c2 if os.path.exists(c2) else None)
    elif ckpt_path in ("crnn_final.pt","checkpoint_latest.pt"):
        ckpt_path = os.path.join(run_dir, ckpt_path)

    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise HTTPException(400, f"checkpoint not found (looked at {ckpt_path})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    report = evaluate_crnn_ctc(
        ckpt_path=ckpt_path,
        val_dir=req.data.val_dir,
        img_h=req.data.img_h,
        img_w_max=req.data.img_w_max,
        batch_size=req.eval.batch_size,
        max_batches=req.eval.max_batches,
        device=device,
    )
    dt = time.time() - t0

    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    report_path = os.path.join(eval_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return {
        "run_id": req.run_id,
        "checkpoint": ckpt_path,
        "device": device,
        "t_seconds": dt,
        "metrics": report.get("metrics", {}),
        "report_path": report_path.replace("\\","/"),
    }

@router.get("/report/{run_id}")
def get_report(run_id: str):
    run_dir = os.path.join("runs", run_id)
    report_path = os.path.join(run_dir, "eval", "report.json")
    if not os.path.exists(report_path):
        raise HTTPException(404, f"report not found: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)
