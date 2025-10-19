from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes_train import router as train_router
from .api.routes_predict import router as predict_router
from .api.routes_runs import router as runs_router
from .api.ws_stream import router as ws_router

app = FastAPI(title="Math-HWR Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train_router, prefix="/api/train", tags=["train"])
app.include_router(predict_router, prefix="/api", tags=["predict"])
app.include_router(runs_router, prefix="/api", tags=["runs"])
app.include_router(ws_router, prefix="/api", tags=["ws"])

@app.get("/")
def root():
    return {"ok": True, "app": "math-hwr-backend"}
