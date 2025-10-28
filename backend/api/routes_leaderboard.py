
from fastapi import APIRouter
import os, json, glob, time

router = APIRouter(prefix="/api/leaderboard", tags=["leaderboard"])

def _read_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@router.get("")
def leaderboard():
    runs_root = "runs"
    rows = []
    if not os.path.isdir(runs_root):
        return {"rows": []}

    for run_id in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, run_id)
        if not os.path.isdir(run_dir):
            continue

        eval_report = _read_json(os.path.join(run_dir, "eval", "report.json"))
        metrics = eval_report.get("metrics") if eval_report else None
        # optional extra: read size of latest checkpoint
        ckpt_path = None
        for name in ["crnn_final.pt","checkpoint_latest.pt"]:
            p = os.path.join(run_dir, name)
            if os.path.exists(p):
                ckpt_path = p
                break
        size_bytes = os.path.getsize(ckpt_path) if ckpt_path and os.path.exists(ckpt_path) else None

        rows.append({
            "run_id": run_id,
            "metrics": metrics,
            "has_eval": eval_report is not None,
            "checkpoint": ckpt_path.replace('\\','/') if ckpt_path else None,
            "checkpoint_size_bytes": size_bytes,
        })
    # sort: best exact desc, then cer asc
    def keyfun(r):
        m = r.get("metrics") or {}
        return (-(m.get("exact") or 0.0), (m.get("cer") or 1.0))
    rows.sort(key=keyfun)
    return {"rows": rows}
