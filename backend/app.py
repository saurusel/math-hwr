from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes_train import router as train_router
from .api.routes_predict import router as predict_router
from .api.routes_runs import router as runs_router
from .api.ws_stream import router as ws_router
from .api.routes_eval import router as eval_router
from .api.routes_models import router as models_router
from .api.routes_leaderboard import router as leaderboard_router
from .api.routes_predict_ckpt import router as predict2_router
from backend.api import routes_predict_attn
from backend.api import routes_predict_vit

app = FastAPI(title="Math-HWR Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173","http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
app.include_router(routes_predict_attn.router)
app.include_router(routes_predict_vit.router)

@app.get("/")
def root():
    return {"ok": True, "app": "math-hwr-backend"}
