# backend/api/routes_predict_attn.py
# -*- coding: utf-8 -*-
"""
Эндпоинт для attention seq2seq (только greedy).
Маршрут: POST /api/predict2/attn/run
Формат запроса совместим с /api/predict2/run.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from PIL import Image, ImageOps, ImageStat
import io
import base64
import torch
import torchvision.transforms.functional as TF

# Модель (инференс) — мини-реализация в backend/core/models/attn_seq2seq.py
from backend.core.models.attn_seq2seq import AttnSeq2Seq

router = APIRouter(tags=["predict2-attn"])


# -------- Pydantic-схемы запроса/ответа --------

class DecodeCfg(BaseModel):
    # beam здесь не поддерживаем; держим ровно greedy
    type: Literal["greedy"] = "greedy"

class DataCfg(BaseModel):
    img_h: int = 64
    img_w_max: int = 512
    invert: Optional[bool] = None  # True/False/None(авто)
    pad_mode: Literal["right", "center"] = "right"
    return_b64_preprocessed: bool = False

class ImageInput(BaseModel):
    path: Optional[str] = None
    b64: Optional[str] = None

class PredictReq(BaseModel):
    run_id: Optional[str] = None
    checkpoint: Optional[str] = None
    data: DataCfg = Field(default_factory=DataCfg)
    decode: DecodeCfg = Field(default_factory=DecodeCfg)
    image: ImageInput

class PredictResp(BaseModel):
    device: str
    ckpt: str
    img_h: int
    img_w_max: int
    decoder: str
    tokens: List[str]
    text: str
    valid: bool
    T: int
    preprocessed_b64: Optional[str] = None


# --------- локальные утилиты (без внешних зависимостей проекта) ---------
def _ensure_attn_checkpoint(ckpt_path: str) -> None:
    try:
        meta = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось открыть checkpoint: {e}")

    arch = str(meta.get("arch", "")).lower()
    sd   = meta.get("state_dict", meta)
    names = " ".join(sd.keys()).lower()

    # явный CTC — сразу 400
    if ("ctc" in arch) or ("ctc_head" in names):
        raise HTTPException(status_code=400, detail="Похоже на CTC-чекпоинт — нужен attention/seq2seq.")

    # явные признаки attn/transformer
    if any(k in arch for k in ("attn","seq2seq","transformer")):
        return
    if any(k in names for k in ("tr.", "transformer", "pos", "emb", "proj")):
        return

    raise HTTPException(status_code=400, detail="Тип чекпоинта не распознан. Укажи attention/seq2seq .pt.")

def _pil_from_data_url(data_url: str) -> Image.Image:
    """Простой парсер data:image/...;base64,..."""
    if not isinstance(data_url, str) or not data_url.startswith("data:image"):
        raise ValueError("unsupported image b64")
    comma = data_url.find(",")
    if comma < 0:
        raise ValueError("bad data URL: no comma")
    b64 = data_url[comma + 1 :]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")

def _to_tensor_from_pil(
    img: Image.Image,
    img_h: int,
    img_w_max: int,
    invert: Optional[bool],
    pad_mode: str,
) -> torch.Tensor:
    """
    1) L (1 канал)
    2) invert: True/False/auto(None -> по средней яркости)
    3) resize по высоте до img_h, ограничить ширину img_w_max
    4) паддинг (right|center)
    5) в тензор [0..1], форма (1, H, W)
    """
    if img.mode != "L":
        img = img.convert("L")

    if invert is True:
        img = ImageOps.invert(img)
    elif invert is None:
        # автоинверт: если темно — инвертнём
        if ImageStat.Stat(img).mean[0] < 127:
            img = ImageOps.invert(img)

    W, H = img.size
    new_w = int(round(W * (img_h / float(H))))
    new_w = max(1, min(new_w, img_w_max))
    img_resized = img.resize((new_w, img_h), Image.BILINEAR)

    if pad_mode == "center":
        pad_left = (img_w_max - new_w) // 2
        pad_right = img_w_max - new_w - pad_left
    else:
        pad_left = 0
        pad_right = img_w_max - new_w

    canvas = Image.new("L", (img_w_max, img_h), color=255)
    canvas.paste(img_resized, (pad_left, 0))

    t = TF.to_tensor(canvas)  # (1,H,W) float32 [0..1]
    return t


# ---------- кеш модели ----------
_model_cache: dict[str, "AttnSeq2Seq"] = {}

def _get_model(ckpt_path: str, device) -> "AttnSeq2Seq":
    """
    Жёсткая проверка: пытаемся именно attention/seq2seq.
    Если передан чужой .pt (например, CTC), вернём 400 с понятной ошибкой.
    """
    m = _model_cache.get(ckpt_path)
    if m is not None:
        return m

    try:
        # Если у твоего класса другой конструктор — подставь свой.
        # Идея проста: попытка инициализировать attn-модель из файла.
        default_tokens = list("0123456789") + ["x","y","z","+","-","*","÷","^","(",")","="]
        model = AttnSeq2Seq.from_checkpoint(
            ckpt_path,
            device=device,
            default_tokens=default_tokens,
            img_h=64,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Checkpoint '{ckpt_path}' не является attention/seq2seq моделью "
                f"или несовместим с ней: {e}"
            ),
        )

    _model_cache[ckpt_path] = model
    return model


# --------------- endpoint ---------------

@router.post("/api/predict2/attn/run", response_model=PredictResp)
def predict2_attn(req: PredictReq) -> PredictResp:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = req.checkpoint
    if not ckpt:
        if not req.run_id:
            raise HTTPException(status_code=400, detail="checkpoint or run_id required")
        ckpt = f"runs/{req.run_id}/checkpoint_latest.pt"

    if getattr(req, "run_id", None):
        rid = (req.run_id or "").lower()
        if not any(tag in rid for tag in ("attn", "seq2seq")):
            raise HTTPException(
                status_code=400,
                detail="Этот эндпоинт только для attention/seq2seq. Укажи run_id с 'attn'/'seq2seq' или верный ckpt."
            )
    _ensure_attn_checkpoint(req.checkpoint)
    # загрузка картинки
    if req.image.path:
        img = Image.open(req.image.path).convert("RGB")
    elif req.image.b64:
        try:
            img = _pil_from_data_url(req.image.b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"bad image b64: {e}")
    else:
        raise HTTPException(status_code=400, detail="image.path or image.b64 required")

    # препроцесс
    t = _to_tensor_from_pil(
        img=img,
        img_h=req.data.img_h,
        img_w_max=req.data.img_w_max,
        invert=req.data.invert,
        pad_mode=req.data.pad_mode,
    )  # (1,H,W)
    t = t.unsqueeze(0).to(device)  # (B=1,1,H,W)
    T = t.size(-1)

    # модель (greedy)
    model = _get_model(ckpt, device=device)
    toks_batch = model.greedy_decode(t, max_len=128)   # List[List[str]]
    toks = toks_batch[0]
    text = " ".join(toks)

    # (опционально) вернуть препроцесс PNG как data URL
    dbg_b64 = None
    if req.data.return_b64_preprocessed:
        arr = (t[0, 0].detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        pil = Image.fromarray(arr, mode="L")
        bio = io.BytesIO()
        pil.save(bio, format="PNG")
        dbg_b64 = "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode("ascii")

    return PredictResp(
        device=str(device),
        ckpt=ckpt,
        img_h=req.data.img_h,
        img_w_max=req.data.img_w_max,
        decoder=req.decode.type,
        tokens=toks,
        text=text,
        valid=True,
        T=T,
        preprocessed_b64=dbg_b64,
    )
