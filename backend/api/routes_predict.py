from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ..core.tokenizer import TOKEN_LIST, normalize_tokens
from PIL import Image
import io

router = APIRouter()

@router.post("/predict")
async def predict(model_id: str = "baseline", file: UploadFile = File(...)):
    # Stub: just returns a fixed demo prediction to test the flow
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(400, "please upload png or jpeg")
    img_bytes = await file.read()
    try:
        Image.open(io.BytesIO(img_bytes)).convert("L")
    except Exception:
        raise HTTPException(400, "invalid image")

    pred_tokens = ["f","r","a","c","(","{","x","+","1","}",",","{","y","-","2","}",")"]
    pred = normalize_tokens(pred_tokens)
    return JSONResponse({
        "model_id": model_id,
        "prediction": pred,
        "tokens": pred_tokens,
    })
