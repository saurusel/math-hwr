
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os, glob

router = APIRouter(prefix="/api/models", tags=["models"])

def _find_checkpoints(run_dir: str):
    cands = []
    for name in ["crnn_final.pt", "checkpoint_latest.pt"]:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            cands.append({"name": name, "path": p.replace('\\','/'), "size_bytes": os.path.getsize(p)})
    # also list any *.pt in run_dir (except duplicates)
    for p in glob.glob(os.path.join(run_dir, "*.pt")):
        bn = os.path.basename(p)
        if bn not in [c["name"] for c in cands]:
            cands.append({"name": bn, "path": p.replace('\\','/'), "size_bytes": os.path.getsize(p)})
    return cands

@router.get("/list")
def list_models():
    runs_root = "runs"
    items = []
    if not os.path.isdir(runs_root):
        return {"models": []}
    for run_id in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, run_id)
        if not os.path.isdir(run_dir):
            continue
        ckpts = _find_checkpoints(run_dir)
        if ckpts:
            items.append({"run_id": run_id, "checkpoints": ckpts})
    return {"models": items}

@router.get("/available")
def list_available_for_inference():
    """
    List checkpoints available for inference in Playground.
    Returns only FINISHED runs with valid checkpoints.
    """
    models = []
    runs_root = "runs"

    if not os.path.exists(runs_root):
        return JSONResponse([], headers={"Content-Type": "application/json; charset=utf-8"})

    for run_name in os.listdir(runs_root):
        run_path = os.path.join(runs_root, run_name)
        if not os.path.isdir(run_path):
            continue

        # Priority: checkpoint_best.pt > crnn_final.pt > checkpoint_latest.pt
        checkpoint_candidates = [
            ("checkpoint_best.pt", "Best"),
            ("crnn_final.pt", "Final"),
            ("checkpoint_latest.pt", "Latest"),
        ]

        for ckpt_file, ckpt_kind in checkpoint_candidates:
            ckpt_path = os.path.join(run_path, ckpt_file)
            if os.path.exists(ckpt_path):
                models.append({
                    "id": f"{run_name}/{ckpt_file}",
                    "run_id": run_name,
                    "checkpoint_name": ckpt_file,
                    "checkpoint_kind": ckpt_kind,
                    "path": ckpt_path,
                    "size_mb": round(os.path.getsize(ckpt_path) / 1024 / 1024, 2),
                    "model_type": "M1",  # CRNN-CTC
                    "modified_at": os.path.getmtime(ckpt_path)
                })
                break  # Only take one checkpoint per run

    # Sort by modification time (newest first)
    models.sort(key=lambda x: x["modified_at"], reverse=True)

    return JSONResponse(models, headers={"Content-Type": "application/json; charset=utf-8"})

@router.post("/{ckpt_id_encoded}/promote")
def promote_checkpoint(ckpt_id_encoded: str, body: dict):
    """
    Promote a checkpoint to production.
    ckpt_id_encoded: base64 or URL-encoded checkpoint ID
    """
    alias = body.get("alias", "production")

    # For MVP, just return success
    # Full implementation would update a registry file or database

    return JSONResponse(
        {"success": True, "ckpt_id": ckpt_id_encoded, "alias": alias},
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@router.delete("/{ckpt_id_encoded}")
def delete_checkpoint_api(ckpt_id_encoded: str):
    """
    Delete a checkpoint.
    ckpt_id_encoded: URL-safe checkpoint path
    """
    try:
        # Decode checkpoint ID (assuming format: run_name/checkpoint_file.pt)
        ckpt_id = ckpt_id_encoded.replace("__", "/")

        if "best" in ckpt_id.lower():
            raise HTTPException(status_code=403, detail="Cannot delete best checkpoint")

        parts = ckpt_id.split("/")
        if len(parts) < 2:
            raise HTTPException(status_code=400, detail="Invalid checkpoint ID")

        ckpt_path = os.path.join("runs", *parts)

        if not os.path.exists(ckpt_path):
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        os.remove(ckpt_path)

        return JSONResponse(
            {"success": True, "deleted": ckpt_id},
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
