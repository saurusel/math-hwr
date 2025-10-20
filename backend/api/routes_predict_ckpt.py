
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Literal, Tuple
import os, io, base64
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn.functional as F

from ..core.models.crnn_ctc import CRNN_CTC
from ..core.tokenizer import TOKEN_LIST
from ..core.decoders.ctc_decode import ctc_greedy_decode
from ..core.parser import is_valid_token_stream

router = APIRouter(prefix="/api/predict2", tags=["predict2"])

# --------------------------
# Request/Response schemas
# --------------------------
class PredictDataCfg(BaseModel):
    img_h: int = 64
    img_w_max: int = 512
    invert: bool = True                             # <— дефолт: инвертируем
    pad_mode: Literal["right", "center"] = "right"  # <— паддинг справа
    return_b64_preprocessed: bool = False
    binarize: Literal["none", "otsu"] = "none"      # можно включить "otsu" при желании

class PredictDecodeCfg(BaseModel):
    type: Literal["greedy","beam"] = "greedy"
    beam_width: int = 5

class PredictImage(BaseModel):
    # Either provide b64 (optionally data URL), or a local path (debug/testing)
    b64: Optional[str] = None
    path: Optional[str] = None

class PredictRequest(BaseModel):
    run_id: Optional[str] = None
    checkpoint: Optional[str] = None  # if run_id given, can be "checkpoint_latest.pt" or "crnn_final.pt"
    ckpt_path: Optional[str] = None   # full path alternative
    data: PredictDataCfg = PredictDataCfg()
    decode: PredictDecodeCfg = PredictDecodeCfg()
    image: PredictImage

# --------------------------
# Utils
# --------------------------
def _resolve_ckpt_path(run_id: Optional[str], checkpoint: Optional[str], ckpt_path: Optional[str]) -> str:
    if ckpt_path:
        p = ckpt_path
    elif run_id and checkpoint:
        p = os.path.join("runs", run_id, checkpoint)
    elif run_id and not checkpoint:
        # prefer final, then latest
        c1 = os.path.join("runs", run_id, "crnn_final.pt")
        c2 = os.path.join("runs", run_id, "checkpoint_latest.pt")
        p = c1 if os.path.exists(c1) else c2
    else:
        raise HTTPException(400, "Provide either ckpt_path, or (run_id [+ checkpoint]).")
    if not p or not os.path.exists(p):
        raise HTTPException(404, f"Checkpoint not found: {p}")
    return p

def _auto_invert_needed(img_gray: Image.Image) -> bool:
    # Heuristic: if background is dark (mean<128), invert so background becomes white
    arr = np.asarray(img_gray, dtype=np.uint8)
    mean = float(arr.mean())
    return mean < 128.0

def _to_tensor_from_pil(pil: Image.Image, cfg: PredictDataCfg) -> Tuple[torch.Tensor, Optional[str]]:
    """
    Превращает PIL в тензор [1,1,H,W] строго в том формате, как при трейне:
      - градации серого
      - aspect-aware resize по высоте img_h
      - ограничение ширины img_w_max
      - паддинг справа (или центр)
      - invert (по умолчанию True — как в твоём «правильном» пресете)
      - опционально binarize="otsu"
      - нормализация в [0,1]
    Возвращает: (тензор, b64 PNG после препроц если запрошено)
    """
    # 1) Grayscale
    if pil.mode != "L":
        pil = pil.convert("L")

    # 2) Aspect-aware resize
    target_h = int(cfg.img_h)
    max_w = int(cfg.img_w_max)
    scale = target_h / float(pil.height)
    new_w = max(1, int(round(pil.width * scale)))

    # если ширина превышает лимит — уменьшаем масштаб (высота тоже изменится)
    if new_w > max_w:
        scale = max_w / float(pil.width)
        new_w = max_w
        target_h = max(1, int(round(pil.height * scale)))

    img_resized = pil.resize((new_w, target_h), Image.BILINEAR)

    # 3) Паддинг на белый фон
    canvas = Image.new("L", (max_w, target_h), color=255)  # белый фон (как бумага)
    if cfg.pad_mode == "center":
        x0 = (max_w - new_w) // 2
    else:  # "right"
        x0 = 0
    canvas.paste(img_resized, (x0, 0))

    # 4) В numpy и инверсия (чернила -> светлое на тёмном, как в трейне)
    arr_uint8 = np.asarray(canvas, dtype=np.uint8)
    if cfg.invert:
        arr_uint8 = 255 - arr_uint8

    # 5) Опциональная бинаризация (часто помогает тонким штрихам)
    if cfg.binarize == "otsu":
        t = _otsu_threshold(arr_uint8)
        arr_uint8 = (arr_uint8 > t).astype(np.uint8) * 255

    # 6) Нормализация в [0,1] и тензор [1,1,H,W]
    arr = arr_uint8.astype(np.float32) / 255.0
    ten = torch.from_numpy(arr)[None, None, :, :]

    # 7) Отладочная картинка после препроц (как «видит» модель)
    dbg_b64 = None
    if cfg.return_b64_preprocessed:
        dbg_img = Image.fromarray(arr_uint8, mode="L")
        with io.BytesIO() as buf:
            dbg_img.save(buf, format="PNG")
            dbg_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    return ten, dbg_b64

def _otsu_threshold(arr_uint8: np.ndarray) -> int:
    """Простой Otsu (без OpenCV). На входе: uint8 [H,W]. Возвращает порог 0..255."""
    hist, _ = np.histogram(arr_uint8, bins=256, range=(0, 255))
    total = arr_uint8.size
    sum_total = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0.0
    var_max = -1.0
    thresh = 127
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > var_max:
            var_max = var_between
            thresh = t
    return int(thresh)

# --------------------------
# Simple CTC prefix beam search (no LM)
# --------------------------
def ctc_beam_search(logits: torch.Tensor, blank_id: int, beam_width: int = 5) -> List[List[int]]:
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, C]
    B, T, C = log_probs.shape
    hyps: List[List[int]] = []

    for b in range(B):
        lp = log_probs[b]  # [T, C]
        # Beam is dict: prefix(tuple) -> (p_blank, p_nonblank) in log-space
        beam = {(): (0.0, float('-inf'))}  # log(1)=0 for blank-only
        for t in range(T):
            next_beam = {}
            for prefix, (p_b, p_nb) in beam.items():
                # extend with blank
                prob_blank = p_b + lp[t, blank_id].item()
                pb, pnb = next_beam.get(prefix, (float('-inf'), float('-inf')))
                pb = float(torch.logaddexp(torch.tensor(pb), torch.tensor(prob_blank)))
                next_beam[prefix] = (pb, pnb)
                # extend with symbols
                last = prefix[-1] if prefix else None
                for c in range(C):
                    if c == blank_id:
                        continue
                    p = lp[t, c].item()
                    if c == last:
                        # same symbol -> only from blank
                        p_new_nb = p_b + p
                        new_prefix = prefix
                    else:
                        p_new_nb = max(p_b, p_nb) + p
                        new_prefix = prefix + (c,)
                    pb, pnb = next_beam.get(new_prefix, (float('-inf'), float('-inf')))
                    pnb = float(torch.logaddexp(torch.tensor(pnb), torch.tensor(p_new_nb)))
                    next_beam[new_prefix] = (pb, pnb)
            # prune
            def score(item):
                (pb, pnb) = item[1]
                return float(torch.logaddexp(torch.tensor(pb), torch.tensor(pnb)))
            beam = dict(sorted(next_beam.items(), key=score, reverse=True)[:max(1, beam_width)])

        best = max(beam.items(), key=lambda kv: float(torch.logaddexp(torch.tensor(kv[1][0]), torch.tensor(kv[1][1]))))[0]
        hyps.append(list(best))
    return hyps

# --------------------------
# Endpoint
# --------------------------
@router.post("/run")
def predict2(req: PredictRequest):
    ckpt = _resolve_ckpt_path(req.run_id, req.checkpoint, req.ckpt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    blank_id = len(TOKEN_LIST)
    num_classes = blank_id + 1
    model = CRNN_CTC(num_classes=num_classes).to(device_t).eval()
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)  # torch >=2.4
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")  # fallback for older torch
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)

    # Prepare image
    pil = _load_image(req.image)
    x, dbg_b64 = _to_tensor_from_pil(pil, req.data)
    x = x.to(device_t)

    with torch.no_grad():
        logits, T = model(x)  # [1, T, C]
        if req.decode.type == "beam":
            ids_batch = ctc_beam_search(logits, blank_id=blank_id, beam_width=max(1, req.decode.beam_width))
        else:
            ids_batch = ctc_greedy_decode(logits, blank_id=blank_id)
    ids = ids_batch[0]
    tokens = [TOKEN_LIST[i] for i in ids]
    s = " ".join(tokens)

    return {
        "device": device,
        "ckpt": ckpt.replace('\\\\','/'),
        "img_h": req.data.img_h,
        "img_w_max": req.data.img_w_max,
        "invert": req.data.invert,
        "pad_mode": req.data.pad_mode,
        "decoder": req.decode.type,
        "beam_width": req.decode.beam_width if req.decode.type == "beam" else None,
        "tokens": tokens,
        "text": s,
        "valid": bool(tokens and is_valid_token_stream(tokens)),
        "T": int(T),
        "preprocessed_b64": dbg_b64 if req.data.return_b64_preprocessed else None,
    }

def _load_image(req_img: PredictImage) -> Image.Image:
    if req_img.b64:
        b64s = req_img.b64
        if b64s.startswith("data:"):
            b64s = b64s.split(",",1)[1]
        raw = base64.b64decode(b64s)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if req_img.path and os.path.exists(req_img.path):
        return Image.open(req_img.path).convert("RGB")
    raise HTTPException(400, "Image not provided or path not found.")
