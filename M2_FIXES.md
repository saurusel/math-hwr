# M2 Bug Fixes & Improvements

**Date:** 2025-10-21
**Issue:** M2 training failed immediately with status FAILED, epoch 0

---

## 🐛 Найденные проблемы:

### Problem #1: cv2.threshold syntax error
**File:** `backend/core/segmentation.py:28`
**Error:** `cv2.threshold | cv2.THRESH_OTSU`

**Fix:**
```python
# WRONG:
_, binary = cv2.threshold(image, 0, 255, cv2.threshold | cv2.THRESH_OTSU)

# CORRECT:
_, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
```

`cv2.threshold` is a **function**, not a constant! Need `cv2.THRESH_BINARY`.

---

### Problem #2: Wrong dictionary keys in dataset
**File:** `backend/core/data/seg_dataset.py:57-58`
**Error:** `KeyError: 'img'` and `KeyError: 'label'`

**Fix:**
```python
# WRONG:
img_path = os.path.join(self.data_dir, "images", expr_data["img"])
label_tokens = expr_data["label"].split()

# CORRECT:
img_path = os.path.join(self.data_dir, expr_data["image"])
label_tokens = expr_data["target"].split()
```

JSONL format uses `"image"` and `"target"`, not `"img"` and `"label"`.

---

### Problem #3: Same error in trainer validation
**File:** `backend/core/train/trainer_seg_mlp.py:200`
**Error:** `KeyError: 'img'`

**Fix:**
```python
# WRONG:
img_path = os.path.join(val_dir, "images", val_row["img"])

# CORRECT:
img_path = os.path.join(val_dir, val_row["image"])
```

---

### Problem #4: Poor segmentation quality
**Issue:** Synthetic dataset has touching characters, segmentation found only 2-7 segments instead of 5-13 tokens

**Improvements applied:**
1. **Morphological opening** to separate touching characters
2. **Connectivity=4** instead of 8 (less aggressive merging)
3. **min_area=5** instead of 20 (accept smaller characters)
4. **Relaxed matching** in dataset (use partial matches)

**Results:**
- Before: 0 characters extracted (crash on KeyError)
- After: 664 characters from 100 expressions (success!)

---

## ✅ All Fixes Applied:

### Backend files updated:
- ✅ `backend/core/segmentation.py` - Fixed cv2.threshold, added morphology, min_area=5
- ✅ `backend/core/data/seg_dataset.py` - Fixed keys, relaxed matching
- ✅ `backend/core/train/trainer_seg_mlp.py` - Fixed keys, min_area=5
- ✅ `backend/api/routes_predict_seg.py` - Default min_area=5

### Strategy changes:
- ✅ **Best-effort matching**: Train on partial segment-token pairs
- ✅ **Lower thresholds**: Accept smaller components (area ≥ 5 pixels)
- ✅ **Morphological processing**: MORPH_OPEN to separate characters

---

## 🧪 Test Results:

### Dataset Creation:
```
100 expressions → 664 characters extracted
Average: 6.6 characters per expression
Success rate: 100% (no crashes)
```

### Segmentation Examples:
- Expression "0 ^ ( 8 )" (5 tokens) → 2 segments found
- Expression "4 ^ ( 1 ) - 3 4 1 ^ ( z )" (13 tokens) → 7 segments found
- Expression "z * z ^ ( 6 ) + 9" (9 tokens) → 4 segments found

**Note:** Segmentation doesn't find all characters perfectly, but **relaxed matching** allows training on partial data.

---

## 🎯 Expected Training Behavior:

### With fixed code:
- ✅ Training starts successfully
- ✅ Dataset loads 500-800 characters from 100 expressions
- ✅ Model trains on character classification
- ✅ Metrics update each epoch

### Realistic expectations:
- **Character accuracy:** 60-80% (imperfect segmentation)
- **Sequence accuracy:** 20-40% (due to missing/merged characters)
- **Segmentation accuracy:** 50-70% (baseline for touching characters)

### For better results:
- Use more data (1000+ expressions)
- Adjust segmentation parameters in Advanced Settings
- Or use M1 (CRNN-CTC) which doesn't need segmentation

---

## 🚀 Ready to use:

```bash
# Restart backend to load fixes:
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

# Create M2 training:
# http://127.0.0.1:5174/train/new
# - Model: M2 (Segmentation + MLP)
# - Epochs: 5
# - Batch Size: 128
# - Max Samples: 100-1000
# - Start Training!
```

---

## 📋 Files Changed:

1. `backend/core/segmentation.py` - 3 fixes
2. `backend/core/data/seg_dataset.py` - 3 fixes
3. `backend/core/train/trainer_seg_mlp.py` - 1 fix
4. `backend/api/routes_predict_seg.py` - 1 fix
5. `backend/core/models/seg_mlp.py` - 1 fix (import List)

**Total:** 5 files, 9 fixes

---

**M2 готова к обучению!** 🚀
