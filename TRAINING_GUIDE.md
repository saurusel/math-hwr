# Training Guide - Math-HWR

Complete guide for training handwriting recognition models.

## 🎯 Quick Start

**1. Create Training:**
```
Navigate to: http://127.0.0.1:5173/train/new
Model: M1 (CRNN-CTC)
Max Samples: 1000
Epochs: 20
Start Training
```

**2. Monitor Progress:**
- Real-time metrics updates
- Interactive charts with zoom
- 10 sample predictions with images
- Can stop early if overfitting

**3. Use Trained Model:**
```
Navigate to: http://127.0.0.1:5173/
Select your trained model from dropdown
Draw expression
Recognize!
```

---

## 📊 Model Comparison

| Model | Best For | Min Samples | Training Time | Expected CER |
|-------|----------|-------------|---------------|--------------|
| M1 (CRNN-CTC) | Speed, simple expressions | 1,000 | ~5 min | 2-5% |
| M2 (Attention) | Complex expressions | 2,000 | ~15 min | 1.5-4% |
| M3 (Vision Transformer) | Maximum quality | 5,000 | ~45 min | 0.5-2% |

---

## 🎛️ Recommended Parameters

### M1 (Fast & Reliable)
```yaml
max_samples: 1000
epochs: 20
batch_size: 64
lr: 0.001
scheduler: cosine

advanced (optional):
  weight_decay: 0.01
```

### M2 (Balanced)
```yaml
max_samples: 2000
epochs: 30
batch_size: 32
lr: 0.0005

advanced:
  d_model: 256
  n_heads: 4
  n_layers_enc: 4
  n_layers_dec: 4
  teacher_forcing: 0.5
```

### M3 (Best Quality)
```yaml
max_samples: 5000
epochs: 50
batch_size: 16
lr: 0.0001

advanced:
  dropout: 0.1
```

---

## 📈 Features

**Real-Time Monitoring:**
- Loss curves (train/val)
- Accuracy metrics (CER/WER/Exact)
- Learning rate schedule
- Training speed
- Best metrics tracker

**Early Stopping:**
- Stop button when overfitting detected
- Graceful shutdown (finishes current epoch)
- Preserves best checkpoint

**Historical Data:**
- events.jsonl persistence
- Can review completed trainings
- Metrics and logs saved

**Sample Gallery:**
- 10 random validation samples
- Images with predictions
- ✓/✗ match indicators
- Character-level diff

---

## 🗂️ File Structure

After training completes:
```
runs/<run_id>/
├── config.json              # Training configuration
├── events.jsonl             # All training events
├── checkpoint_best.pt       # Best CER checkpoint
├── checkpoint_latest.pt     # Last epoch
├── checkpoint_epoch_*.pt    # Per-epoch checkpoints
├── crnn_final.pt (M1)       # Final model
├── attn_final.pt (M2)       # Final model
├── vit_final.pt (M3)        # Final model
└── samples/                 # Validation sample images
```

---

## 🎓 Best Practices

**Dataset Size:**
- Too small (<500): Won't converge
- Minimum (1000): Basic functionality
- Recommended (5000+): Good quality
- Full dataset (96K): Best quality

**When to Stop:**
- Validation loss starts increasing
- CER plateaus for 5+ epochs
- Reached target quality
- Best metrics tracker shows degradation

**Hyperparameter Tips:**
- Start with defaults
- Lower LR if loss unstable
- Increase epochs if still improving
- Use max_samples for quick experiments

---

## 🐛 Troubleshooting

**CER = 100%:**
- Insufficient data (increase max_samples)
- Too few epochs (increase)
- Model not suitable for data

**Training crashes:**
- Check data paths exist
- Verify GPU memory sufficient
- Lower batch_size if OOM

**No real-time updates:**
- Check browser console (F12)
- Verify SSE connection
- Restart backend

---

For detailed architecture, see `README.md`.
