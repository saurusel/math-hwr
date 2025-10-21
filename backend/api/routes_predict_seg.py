"""
M2 Prediction API: Segmentation + MLP Classifier
Handles inference for the classical OCR approach.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os, io, base64
from PIL import Image
import numpy as np
import torch

from ..core.models.seg_mlp import SegmentationOCR
from ..core.tokenizer import TOKEN_LIST
from ..core.segmentation import segment_and_prepare

router = APIRouter(prefix="/api/predict_seg", tags=["predict_seg"])


class PredictSegRequest(BaseModel):
    """Request for M2 segmentation-based prediction."""
    run_id: Optional[str] = None
    checkpoint: Optional[str] = None
    ckpt_path: Optional[str] = None
    image_b64: Optional[str] = None
    image_path: Optional[str] = None
    min_area: int = 5
    max_area: int = 4000


def _resolve_ckpt_path(run_id: Optional[str], checkpoint: Optional[str], ckpt_path: Optional[str]) -> str:
    """Resolve checkpoint path from various input formats."""
    if ckpt_path:
        p = ckpt_path
    elif run_id and checkpoint:
        p = os.path.join("runs", run_id, checkpoint)
    elif run_id and not checkpoint:
        c1 = os.path.join("runs", run_id, "seg_mlp_final.pt")
        c2 = os.path.join("runs", run_id, "seg_mlp_best.pt")
        c3 = os.path.join("runs", run_id, "checkpoint_latest.pt")
        for candidate in [c1, c2, c3]:
            if os.path.exists(candidate):
                p = candidate
                break
        else:
            p = c1  # Will raise 404 below
    else:
        raise HTTPException(400, "Provide either ckpt_path, or run_id")

    if not p or not os.path.exists(p):
        raise HTTPException(404, f"Checkpoint not found: {p}")

    return p


def _load_image_from_request(req: PredictSegRequest) -> Image.Image:
    """Load image from base64 or file path."""
    if req.image_b64:
        b64s = req.image_b64
        if b64s.startswith("data:"):
            b64s = b64s.split(",", 1)[1]
        raw = base64.b64decode(b64s)
        return Image.open(io.BytesIO(raw)).convert("L")

    if req.image_path and os.path.exists(req.image_path):
        return Image.open(req.image_path).convert("L")

    raise HTTPException(400, "No image provided")


@router.post("/run")
def predict_seg(req: PredictSegRequest):
    """
    Predict using M2 (Segmentation + MLP) model.

    Process:
    1. Load trained MLP model
    2. Segment input image into characters
    3. Classify each character
    4. Combine into sequence

    Returns:
        prediction: Predicted token sequence
        num_segments: Number of characters found
        seg_details: Details about each segment
    """
    ckpt = _resolve_ckpt_path(req.run_id, req.checkpoint, req.ckpt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    # Load model
    num_classes = len(TOKEN_LIST)
    model = SegmentationOCR(num_classes=num_classes, input_size=32*32).to(device_t).eval()

    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")

    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)

    # Load and prepare image
    img_pil = _load_image_from_request(req)
    img_arr = np.array(img_pil)

    # Segment characters
    segments = segment_and_prepare(img_arr, target_size=(32, 32), min_area=req.min_area, max_area=req.max_area)

    if not segments:
        return {
            "device": device,
            "prediction": "",
            "tokens": [],
            "num_segments": 0,
            "message": "No characters detected. Try adjusting min_area/max_area or check if image has visible content."
        }

    # Prepare batch of character tensors
    seg_tensors = [torch.from_numpy(seg[0]).unsqueeze(0) for seg in segments]
    batch = torch.stack(seg_tensors).to(device_t)  # (num_chars, 1, H, W)

    # Classify
    with torch.no_grad():
        logits = model(batch)
        pred_ids = torch.argmax(logits, dim=1).cpu().tolist()

    # Convert to tokens
    pred_tokens = [TOKEN_LIST[pid] if pid < len(TOKEN_LIST) else "?" for pid in pred_ids]
    text = " ".join(pred_tokens)

    # Segment details
    seg_details = []
    for idx, (seg_img, bbox) in enumerate(segments):
        seg_details.append({
            "index": idx,
            "bbox": {"x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]), "h": int(bbox[3])},
            "predicted_token": pred_tokens[idx] if idx < len(pred_tokens) else "?"
        })

    return {
        "device": device,
        "prediction": text,
        "tokens": pred_tokens,
        "num_segments": len(segments),
        "seg_details": seg_details,
        "checkpoint": os.path.basename(ckpt)
    }
