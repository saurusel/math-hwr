# Math-HWR Backend

FastAPI backend for handwriting mathematical expression recognition.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python -m backend.app

# Server runs at:
# http://127.0.0.1:8000
```

## 📡 API Endpoints

### Training

```
POST /api/train/jobs           # Create training job
GET  /api/train/jobs/{id}      # Job status
POST /api/train/jobs/{id}/stop # Stop training
GET  /api/train/stream/{id}    # SSE live events
GET  /api/train/jobs/{id}/history # Historical data
```

### Experiments

```
GET /api/experiments                    # List all experiments
GET /api/experiments/{id}/checkpoints   # List checkpoints
```

### Inference

```
POST /api/predict2/run       # M1 (CRNN-CTC)
POST /api/predict2/attn/run  # M2 (Attention)
POST /api/predict2/vit/run   # M3 (Vision Transformer)
```

### Models

```
GET  /api/models/available        # List trained models
POST /api/models/{id}/promote     # Promote checkpoint
DELETE /api/models/{id}           # Delete checkpoint
```

## 🏗️ Architecture

```
backend/
├── api/                    # FastAPI routes
│   ├── routes_train.py     # Training endpoints
│   ├── routes_predict_*.py # Inference endpoints
│   └── routes_models.py    # Model management
├── core/
│   ├── models/             # M1/M2/M3 architectures
│   ├── train/              # Trainers
│   ├── data/               # Datasets
│   ├── decoders/           # CTC decoders
│   └── event_logger.py     # events.jsonl persistence
└── app.py                  # FastAPI app
```

## 🎓 Models

- **M1:** CRNN + CTC (fast, 1K samples)
- **M2:** Attention Seq2Seq (balanced, 2K samples)
- **M3:** Vision Transformer (best, 5K samples)

## 📝 Training Request Format

```json
{
  "model_type": "M1",
  "run_name": "my_experiment",
  "dataset": {
    "name": "synth",
    "train_manifest": "data/synth/train/labels.jsonl",
    "val_manifest": "data/synth/val/labels.jsonl"
  },
  "hyper": {
    "batch_size": 64,
    "epochs": 20,
    "img_h": 64,
    "img_w_max": 512,
    "lr": 0.001,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "seed": 42,
    "max_samples": 1000
  },
  "augment": {
    "invert": true,
    "random_pad": false
  }
}
```

## 🔄 SSE Events

```
event: status
data: {"status": "RUNNING"}

event: metric
data: {"epoch": 1, "train_loss": 2.5, "cer": 0.25, ...}

event: log
data: {"ts": "...", "line": "Epoch 1 completed"}

event: sample_pred
data: {"epoch": 1, "items": [...]}

event: checkpoint
data: {"kind": "best", "path": "..."}
```

## 📁 events.jsonl Format

```jsonl
{"type":"status","status":"RUNNING","timestamp":"..."}
{"type":"metric","epoch":1,"train_loss":2.5,"cer":0.25,...}
{"type":"log","line":"Epoch 1 completed"}
{"type":"checkpoint","kind":"best","path":"...","metrics":{...}}
{"type":"status","status":"FINISHED"}
```

## 🛡️ Error Handling

All endpoints return proper HTTP status codes:
- 200: Success
- 400: Bad request (invalid parameters)
- 404: Not found (job/checkpoint)
- 500: Server error

All responses UTF-8 encoded with `charset=utf-8` header.

## 🎯 CORS

Configured for frontend at:
- http://127.0.0.1:5173
- http://localhost:5173

## 📝 Development

**Run with auto-reload:**
```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

**View API docs:**
```
http://127.0.0.1:8000/docs
```
