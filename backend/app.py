import os
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .api.routes_train import router as train_router
from .api.routes_predict import router as predict_router
from .api.routes_runs import router as runs_router
from .api.ws_stream import router as ws_router
from .api.routes_eval import router as eval_router
from .api.routes_models import router as models_router
from .api.routes_leaderboard import router as leaderboard_router
from .api.routes_predict_ckpt import router as predict2_router
from .api.routes_predict_seg import router as predict_seg_router
from .core.logger import setup_logger, get_logger, log_request, log_error
import torch

# Setup logging
logger = setup_logger("math-hwr", log_dir="logs", level=10)  # DEBUG level
logger.info("=" * 60)
logger.info("Math-HWR Backend Starting...")
logger.info("=" * 60)

app = FastAPI(title="Math-HWR Backend - M1/M2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173","http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and responses."""
    start_time = datetime.now()

    # Log request
    logger.info(f"→ {request.method} {request.url.path}")

    try:
        response = await call_next(request)

        # Log response
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"← {request.method} {request.url.path} | Status: {response.status_code} | Duration: {duration:.3f}s")

        return response
    except Exception as e:
        log_error(f"Request {request.method} {request.url.path}", e)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )

app.include_router(train_router, prefix="/api/train", tags=["train"])
# Mount experiments endpoints at /api level (as per Train-and-Models-Spec)
app.include_router(train_router, prefix="/api", tags=["experiments"])
app.include_router(predict_router, prefix="/api", tags=["predict"])
app.include_router(runs_router, prefix="/api", tags=["runs"])
app.include_router(ws_router, prefix="/api", tags=["ws"])
app.include_router(eval_router)
app.include_router(models_router)
app.include_router(leaderboard_router)
app.include_router(predict2_router)
app.include_router(predict_seg_router)  # M2: Segmentation + MLP
# Old M3 (ViT) removed

@app.on_event("startup")
async def startup_event():
    """Log system info on startup."""
    logger.info("🚀 Checking system configuration...")

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    logger.info(f"   CUDA Available: {cuda_available}")
    if cuda_available:
        logger.info(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Check directories
    dirs_to_check = ["runs", "data", "logs"]
    for dir_name in dirs_to_check:
        exists = os.path.exists(dir_name)
        logger.info(f"   Directory '{dir_name}': {'✓ exists' if exists else '✗ missing'}")
        if not exists and dir_name == "logs":
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"   Created '{dir_name}' directory")

    logger.info("✓ Backend ready!")
    logger.info("=" * 60)


@app.get("/")
def root():
    """Root endpoint."""
    logger.debug("Root endpoint accessed")
    return {"ok": True, "app": "math-hwr-backend", "version": "2.0.0"}


@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring.
    Returns system status and configuration.
    """
    import psutil

    try:
        # System info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')

        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "cuda": {
                "available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            },
            "directories": {
                "runs": os.path.exists("runs"),
                "data": os.path.exists("data"),
                "logs": os.path.exists("logs")
            }
        }

        logger.debug(f"Health check: CPU {cpu_percent}%, Memory {memory.percent}%")
        return health_data

    except Exception as e:
        log_error("Health check", e)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
