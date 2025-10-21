"""M3 Inference Endpoint - Vision Transformer"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os, io, base64
from PIL import Image
import numpy as np
import torch

from ..core.models.vision_transformer import VisionTransformerSeq2Seq
from ..core.tokenizer import TOKEN_LIST

router = APIRouter(prefix="/api/predict2/vit", tags=["predict_vit"])


class VitPredictRequest(BaseModel):
    run_id: Optional[str] = None
    checkpoint: Optional[str] = None
    ckpt_path: Optional[str] = None
    image: dict  # {b64: str} or {path: str}
    data: Optional[dict] = None


def _resolve_checkpoint(run_id: Optional[str], checkpoint: Optional[str], ckpt_path: Optional[str]) -> str:
    """Resolve checkpoint path."""
    if ckpt_path:
        p = ckpt_path
    elif run_id and checkpoint:
        p = os.path.join("runs", run_id, checkpoint)
    elif run_id:
        # Try vit_final.pt first, then checkpoint_latest.pt
        p1 = os.path.join("runs", run_id, "vit_final.pt")
        p2 = os.path.join("runs", run_id, "checkpoint_latest.pt")
        p = p1 if os.path.exists(p1) else p2
    else:
        raise HTTPException(400, "Provide run_id or ckpt_path")

    if not os.path.exists(p):
        raise HTTPException(404, f"Checkpoint not found: {p}")

    return p


def _load_image(img_dict: dict) -> Image.Image:
    """Load image from base64 or path."""
    if "b64" in img_dict:
        b64_str = img_dict["b64"]
        if b64_str.startswith("data:"):
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(raw)).convert("L")
    elif "path" in img_dict:
        return Image.open(img_dict["path"]).convert("L")
    raise HTTPException(400, "Image not provided")


def _preprocess_image(img: Image.Image, img_h: int, img_w: int) -> torch.Tensor:
    """Preprocess image for M3."""
    w, h = img.size
    scale = img_h / h
    new_w = min(int(w * scale), img_w)
    img = img.resize((new_w, img_h), Image.BILINEAR)

    # Pad to img_w
    if new_w < img_w:
        from PIL import ImageOps
        img = ImageOps.expand(img, border=(0, 0, img_w - new_w, 0), fill=255)

    # To tensor and normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = 1.0 - arr  # Invert (black ink on white)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    return tensor


@router.post("/run")
def predict_vit(req: VitPredictRequest):
    """M3 (Vision Transformer) inference."""
    ckpt_path = _resolve_checkpoint(req.run_id, req.checkpoint, req.ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except:
        state = torch.load(ckpt_path, map_location="cpu")

    vocab_tokens = state.get("vocab_tokens", TOKEN_LIST)
    img_h = state.get("img_h", 64)

    # Build model
    vocab_size = len(vocab_tokens) + 3  # PAD, BOS, EOS
    model = VisionTransformerSeq2Seq(
        vocab_size=vocab_size,
        img_h=img_h,
        img_w=512,
        patch_size=16,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4
    ).to(device)

    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()

    # Load and preprocess image
    img = _load_image(req.image)
    x = _preprocess_image(img, img_h, 512).to(device)

    # Inference
    with torch.no_grad():
        pred_ids_batch = model.greedy_decode(x, max_len=64)

    # Decode tokens
    pred_ids = pred_ids_batch[0]
    pred_ids = [i - 3 for i in pred_ids if i >= 3]  # Remove BOS/EOS/PAD
    pred_tokens = [vocab_tokens[i] for i in pred_ids if 0 <= i < len(vocab_tokens)]
    pred_text = " ".join(pred_tokens)

    return {
        "device": str(device),
        "ckpt": ckpt_path.replace("\\", "/"),
        "model_type": "M3",
        "tokens": pred_tokens,
        "text": pred_text,
        "valid": len(pred_tokens) > 0
    }
