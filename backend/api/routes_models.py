
from fastapi import APIRouter
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
