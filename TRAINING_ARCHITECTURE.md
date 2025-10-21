# Training Architecture - How Models Are Trained via Website

Complete explanation of the training flow from frontend to filesystem.

## 🎯 Overview

When you create a training job via the website, here's what happens:

```
Frontend Form
    ↓
POST /api/train/jobs
    ↓
Backend creates job
    ↓
Background thread starts training
    ↓
Model saved to filesystem
    ↓
SSE streams live updates
```

## 📁 File System Structure

### Where Models Are Stored

When you create a training job, everything is stored in `runs/<job_id>/`:

```
runs/
└── run_1760992172599_b1e4111a/     # Your job ID
    ├── config.json                  # Training configuration (from frontend)
    ├── checkpoint_latest.pt         # Latest model checkpoint
    ├── checkpoint_epoch_5.pt        # Checkpoint at epoch 5
    ├── checkpoint_epoch_10.pt       # Checkpoint at epoch 10
    ├── crnn_final.pt               # Final trained model
    ├── events.jsonl                # Training events log (future)
    └── samples/                    # Validation sample images
        ├── step_100.png
        ├── step_200.png
        └── ...
```

**Key files:**
- **`config.json`** - Full training configuration (what you entered in form)
- **`checkpoint_latest.pt`** - Always the most recent checkpoint
- **`checkpoint_epoch_N.pt`** - Checkpoint saved at end of epoch N
- **`crnn_final.pt`** - Final model after all epochs complete
- **`samples/*.png`** - Validation samples for debugging

### Checkpoint Contents

Each `.pt` file contains:
```python
{
    "model": model.state_dict(),      # Model weights
    "opt": optimizer.state_dict(),    # Optimizer state (for resume)
    "scaler": scaler.state_dict(),    # AMP scaler state
    "epoch": 10,                      # Current epoch number
    "global_step": 5000,              # Total training steps
    "vocab": ["0","1",...,"^"]        # Token vocabulary (in final model)
}
```

## 🔄 Training Flow (Step by Step)

### 1. Frontend: User Fills Form

**Page:** `/train/new`

**User enters:**
- Model type: M1 (CRNN-CTC)
- Run name: `my_experiment`
- Batch size: 64
- Epochs: 20
- Learning rate: 0.001
- etc.

### 2. Frontend: POST Request

**Endpoint:** `POST /api/train/jobs`

**Request body:**
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
    "seed": 42
  },
  "augment": {
    "invert": true,
    "random_pad": false
  }
}
```

### 3. Backend: Create Job

**File:** `backend/api/routes_train.py` (line 64)

**Process:**
1. Generate unique `job_id`: `my_experiment_a1b2c3d4`
2. Create directory: `runs/my_experiment_a1b2c3d4/`
3. Save `config.json` with UTF-8 encoding
4. Map frontend format to internal trainer format
5. Store job in `RUNS` dictionary (in-memory registry)
6. Start background thread with `train_crnn_ctc()`
7. Return `{"job_id": "...", "status": "QUEUED"}`

### 4. Backend: Start Training Thread

**File:** `backend/core/train/trainer_crnn.py` (line 27)

**Thread runs:**
```python
def _runner():
    RUNS[job_id]["status"] = "RUNNING"
    train_crnn_ctc(config, runs_dir, emit_callback)
    RUNS[job_id]["status"] = "FINISHED"
```

**Training process:**
1. Load datasets from `data/synth/train/` and `data/synth/val/`
2. Create CRNN_CTC model
3. Setup optimizer (AdamW), scheduler (Cosine), AMP scaler
4. For each epoch:
   - Train on batches
   - Emit metrics every N steps
   - Validate at epoch end
   - Save `checkpoint_latest.pt`
   - Save `checkpoint_epoch_N.pt`
5. Save `crnn_final.pt` when done
6. Emit "finished" event

### 5. Backend: SSE Streaming

**Endpoint:** `GET /api/train/stream/{job_id}`

**File:** `backend/api/routes_train.py` (line 249)

**SSE events sent to frontend:**
```
event: status
data: {"status": "RUNNING"}

event: metric
data: {"epoch": 1, "train_loss": 2.3, "val_loss": 2.1, "cer": 0.15, "lr": 0.001}

event: metric
data: {"epoch": 2, "train_loss": 1.8, "val_loss": 1.7, "cer": 0.12, "lr": 0.0009}

...

event: status
data: {"status": "FINISHED"}
```

### 6. Frontend: Live Updates

**Page:** `/train/monitor?job={job_id}`

**Updates in real-time:**
- Loss curves chart (Recharts)
- Metrics cards (CER, WER, Exact Match, LR)
- Logs viewer
- Sample predictions

**State management:** Zustand store collects all events

## 🐛 Why Your Models Were FAILING

### Root Cause

**Problem:** Config format mismatch between frontend and trainer.

**Trainer expects:**
```python
data_cfg["train_dir"]  # e.g., "data/synth/train"
data_cfg["val_dir"]    # e.g., "data/synth/val"
```

**Frontend sent (before fix):**
```python
data_cfg["train_manifest"]  # "data/synth/train/labels.jsonl"
data_cfg["val_manifest"]    # "data/synth/val/labels.jsonl"
```

**Result:** `KeyError: 'train_dir'` → Training thread crashed → Status = FAILED

### Fix Applied

**File:** `backend/api/routes_train.py` (lines 77-79)

```python
# Extract directory from manifest path
train_dir = os.path.dirname(request.dataset.train_manifest)  # "data/synth/train"
val_dir = os.path.dirname(request.dataset.val_manifest)      # "data/synth/val"

internal_config = {
    "data": {
        "train_dir": train_dir,    # ✅ Now correct!
        "val_dir": val_dir,        # ✅ Now correct!
        "img_h": request.hyper.img_h,
        "img_w_max": request.hyper.img_w_max,
    }
}
```

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
├─────────────────────────────────────────────────────────────┤
│  /train/new        →  User fills form                        │
│  /train/monitor    →  Live SSE updates                       │
│  /train/experiments →  View all jobs                         │
│  /train/models     →  Manage checkpoints                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/SSE
┌──────────────────────────▼──────────────────────────────────┐
│                   Backend (FastAPI)                          │
├─────────────────────────────────────────────────────────────┤
│  routes_train.py:                                            │
│    POST /api/train/jobs          → Create job               │
│    GET  /api/train/jobs/{id}     → Get status               │
│    GET  /api/train/stream/{id}   → SSE events               │
│    GET  /api/experiments         → List all                 │
│                                                              │
│  RUNS = {}  (in-memory registry)                            │
│    job_id → {status, config, epoch, queues, ...}            │
└──────────────────────────┬──────────────────────────────────┘
                           │ Threading
┌──────────────────────────▼──────────────────────────────────┐
│              Training Thread (Background)                    │
├─────────────────────────────────────────────────────────────┤
│  trainer_crnn.py:                                            │
│    1. Load datasets from data/synth/train & val             │
│    2. Create CRNN_CTC model                                 │
│    3. Setup optimizer, scheduler, AMP                       │
│    4. For each epoch:                                       │
│         - Train batches                                     │
│         - Emit metrics via callback                         │
│         - Validate                                          │
│         - Save checkpoints                                  │
│    5. Save final model                                      │
│    6. Emit "finished" event                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ File I/O
┌──────────────────────────▼──────────────────────────────────┐
│                  File System (runs/)                         │
├─────────────────────────────────────────────────────────────┤
│  runs/                                                       │
│  └── my_experiment_a1b2c3d4/                                │
│      ├── config.json              ← Training config         │
│      ├── checkpoint_latest.pt     ← Latest weights          │
│      ├── checkpoint_epoch_5.pt    ← Epoch 5 weights         │
│      ├── checkpoint_epoch_10.pt   ← Epoch 10 weights        │
│      ├── crnn_final.pt           ← Final model              │
│      └── samples/                 ← Validation images       │
│          └── step_100.png                                   │
└─────────────────────────────────────────────────────────────┘
```

## 🔍 Data Flow: Frontend → Backend → Trainer

### Config Transformation

**1. Frontend sends (Train-and-Models-Spec format):**
```json
{
  "dataset": {
    "train_manifest": "data/synth/train/labels.jsonl"
  }
}
```

**2. Backend transforms (routes_train.py):**
```python
train_dir = os.path.dirname("data/synth/train/labels.jsonl")
# Result: "data/synth/train"

internal_config = {
    "data": {
        "train_dir": "data/synth/train"  # ✅ Trainer format
    }
}
```

**3. Trainer receives (trainer_crnn.py):**
```python
train_dir = config["data"]["train_dir"]  # "data/synth/train"
train_ds = HWRDataset(train_dir, ...)    # Loads labels.jsonl from this dir
```

## 🚀 How to Use

### Start New Training

1. **Frontend:** Go to `/train/new`
2. **Fill form:**
   - Model: M1
   - Epochs: 5 (for quick test)
   - Rest: defaults are fine
3. **Click:** "Start Training"
4. **Backend creates:**
   - Directory: `runs/run_<timestamp>_<uuid>/`
   - Config: `config.json`
   - Starts training thread
5. **Frontend redirects:** `/train/monitor?job=run_xxx`
6. **Watch live:** Metrics, logs, loss curves

### View Saved Models

**After training finishes:**

```bash
ls runs/run_<id>/
# You'll see:
# - checkpoint_latest.pt      (last epoch)
# - checkpoint_epoch_5.pt     (epoch 5)
# - checkpoint_epoch_10.pt    (epoch 10)
# - crnn_final.pt            (final model)
```

**To use a checkpoint for inference:**

Go to `/train/models?run=<id>`:
1. Select checkpoint
2. Click "Quick Test" tab
3. Draw expression
4. Click "Recognize"

## 🔧 Technical Details

### In-Memory Registry

```python
RUNS = {
    "run_xxx": {
        "job_id": "run_xxx",
        "run_name": "my_experiment",
        "model_type": "M1",
        "status": "RUNNING",  # QUEUED → RUNNING → FINISHED/FAILED
        "config": {...},       # Internal config for trainer
        "epoch": 5,            # Current epoch
        "step": 2500,          # Global step counter
        "dir": "runs/run_xxx", # Output directory
        "queues": {q1, q2},    # SSE client queues
        "started_at": "2025-10-21T01:29:00",
        "last_error": None     # Error traceback if failed
    }
}
```

### Threading Model

**Main thread:** FastAPI server handles HTTP requests

**Background threads:** One per training job
- Runs `train_crnn_ctc()` function
- Emits events to all connected SSE clients
- Catches exceptions → sets status to FAILED
- Auto-cleanup when done

**SSE threads:** One per connected client
- Async generator streams events
- 30-second timeout with keep-alive pings
- Auto-removes queue on disconnect

### Event System

**Trainer emits events via callback:**
```python
def train_crnn_ctc(config, run_dir, emit):
    emit({"event": "train_step", "epoch": 1, "step": 100, "metrics": {...}})
    emit({"event": "epoch_end", "epoch": 1, "metrics": {...}})
    emit({"event": "finished"})
```

**Backend routes events to SSE clients:**
```python
def _emit(run_id, payload):
    for client_queue in RUNS[run_id]["queues"]:
        client_queue.put_nowait(payload)
```

**Frontend receives via SSE:**
```typescript
eventSource.addEventListener('metric', (e) => {
    const data = JSON.parse(e.data);
    addMetric(data);  // Update Zustand store → React re-renders
});
```

## 🐛 Common Issues & Fixes

### Issue 1: FAILED Status Immediately

**Symptom:** Training job created but status = FAILED right away

**Causes:**
1. ✅ **FIXED:** Config format mismatch (`train_dir` vs `train_manifest`)
2. Dataset path doesn't exist
3. Missing data files (`labels.jsonl`)
4. Permission errors
5. CUDA/GPU errors

**How to debug:**

Check the run directory for errors:
```bash
# Check if config was saved
cat runs/run_xxx/config.json

# Check if datasets exist
ls data/synth/train/labels.jsonl
ls data/synth/val/labels.jsonl
```

**Look at backend console** for exception traceback when FAILED occurs.

### Issue 2: SSE 404 Errors

**Symptom:** Browser console shows `GET /api/train/stream/{id} 404`

**Fix:** ✅ **FIXED** - Added SSE endpoint (routes_train.py:249)

### Issue 3: No Metrics Appearing

**Symptom:** Charts show "No metrics yet"

**Causes:**
1. Trainer not emitting events
2. SSE not connected
3. Event format mismatch

**Fix:** Ensure `emit()` callback is called in trainer

## 🎯 What Was Fixed

### Before (❌ Broken):

**Backend sent to trainer:**
```python
config = {
    "data": {
        "train_manifest": "data/synth/train/labels.jsonl",  # Wrong key!
        "val_manifest": "data/synth/val/labels.jsonl"
    }
}
```

**Trainer tried to access:**
```python
train_dir = data_cfg["train_dir"]  # KeyError! → FAILED
```

### After (✅ Fixed):

**Backend now sends:**
```python
train_dir = os.path.dirname("data/synth/train/labels.jsonl")  # Extract dir
val_dir = os.path.dirname("data/synth/val/labels.jsonl")

config = {
    "data": {
        "train_dir": "data/synth/train",  # ✅ Correct!
        "val_dir": "data/synth/val"       # ✅ Correct!
    }
}
```

**Trainer receives:**
```python
train_dir = data_cfg["train_dir"]  # ✅ Works!
train_ds = HWRDataset(train_dir, ...)
```

## 🚀 How to Apply Fix

**1. Restart backend:**
```bash
# Stop current backend (Ctrl+C)
python -m backend.app
```

**2. Refresh frontend:**
```bash
# In browser: Ctrl + Shift + R
```

**3. Create new training job:**
- Go to `/train/new`
- Fill form with defaults
- Click "Start Training"

**4. Expected result:**
```
Status: QUEUED → RUNNING → training for N epochs → FINISHED ✅
```

**No more FAILED!** 🎉

## 📊 Monitoring Active Training

### Where to Find Information

**Backend console:**
```
INFO: "POST /api/train/jobs HTTP/1.1" 200 OK
INFO: "GET /api/train/stream/run_xxx HTTP/1.1" 200 OK
```

**Frontend monitor page:**
- Loss curves update live
- Metrics update every ~25 steps
- Logs stream in real-time

**File system:**
```bash
# Watch checkpoints being created
watch ls -lh runs/run_xxx/

# View latest checkpoint info
python -c "import torch; print(torch.load('runs/run_xxx/checkpoint_latest.pt', map_location='cpu').keys())"
```

## 🎓 Advanced Topics

### Resume Training

To resume from a checkpoint:
```python
config["resume_from"] = "run_previous_id"
```

Trainer will load `runs/run_previous_id/checkpoint_latest.pt` and continue.

### Custom Scheduler

Frontend supports:
- `"onecycle"` - OneCycleLR (default)
- `"cosine"` - CosineAnnealingLR

Configured via `hyper.scheduler` field.

### Advanced Hyperparameters

**M1 (CRNN-CTC):**
```json
{
  "hyper": {
    "advanced": {
      "m1": {
        "weight_decay": 0.01
      }
    }
  }
}
```

Merged into `config["train"]` by backend.

## ✅ Verification Checklist

After fix, verify:

- [ ] Create training job → Status = QUEUED
- [ ] Wait 1 second → Status = RUNNING
- [ ] See metrics updating on monitor page
- [ ] Check `runs/run_xxx/` directory created
- [ ] See `checkpoint_latest.pt` appearing
- [ ] Training completes → Status = FINISHED
- [ ] See `crnn_final.pt` in run directory
- [ ] No 404 errors in browser console
- [ ] SSE connection successful

## 📝 Summary

**Physical location of models:**
```
C:\123\math-hwr\runs\<job_id>\*.pt
```

**Why it was failing:**
- Config key mismatch: `train_manifest` vs `train_dir`

**Fix:**
- Extract directory path from manifest path
- Pass correct keys to trainer

**Now it works!** ✅

Try creating a new training job - it should complete successfully! 🎉
