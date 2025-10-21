# Migration Summary: M2/M3 → M2 (Segmentation)

**Date:** 2025-10-21
**Branch:** `development`
**Base Commit:** `c018348 feat(frontend): Complete training and inference UI`

---

## ✅ Что сделано:

### 1. Удалены старые модели:
- ❌ M2 (Attention Seq2Seq) - `backend/core/models/attn_seq2seq.py`
- ❌ M3 (Vision Transformer) - `backend/core/models/vision_transformer.py`
- ❌ Trainers: `trainer_attn.py`, `trainer_vit.py`
- ❌ API routes: `routes_predict_attn.py`, `routes_predict_vit.py`

### 2. Создана новая M2 (Segmentation + MLP):

**Backend компоненты:**
- ✅ `backend/core/segmentation.py` - Connected components сегментация
- ✅ `backend/core/models/seg_mlp.py` - MLP классификатор (перцептрон)
- ✅ `backend/core/data/seg_dataset.py` - Dataset с сегментированными символами
- ✅ `backend/core/train/trainer_seg_mlp.py` - Trainer для M2
- ✅ `backend/api/routes_predict_seg.py` - API endpoint `/api/predict_seg/run`

**Frontend обновления:**
- ✅ `frontend/src/types/training.ts` - ModelType: 'M1' | 'M2' (убран M3)
- ✅ `frontend/src/pages/Train/NewTraining.tsx` - Обновлён UI для M2
- ✅ `frontend/src/components/Train/AdvancedSettings.tsx` - Новые M2 параметры

**Документация:**
- ✅ `M2_SEGMENTATION_GUIDE.md` - Полное руководство по M2
- ✅ `README.md` - Обновлён для M1/M2

---

## 🏗️ Новая архитектура M2:

```python
# Segmentation (OpenCV)
segments = segment_characters(image)  # Connected components
↓
# Normalization
prepared = prepare_segment_for_classification(segment, target_size=(32,32))
↓
# MLP Classification
class SegmentationMLP(nn.Module):
    Input: 1024 (32x32 flattened)
    Hidden: 1024 → 512 → 256 → 128
    Output: num_classes
    Regularization: BatchNorm + Dropout(0.3)
↓
# Sequence assembly
tokens = [TOKEN_LIST[pred_id] for pred_id in predictions]
```

---

## 📊 Файловая структура:

```
backend/
├── core/
│   ├── models/
│   │   ├── crnn_ctc.py (M1) ✅
│   │   └── seg_mlp.py (M2) ✅ NEW
│   ├── train/
│   │   ├── trainer_crnn.py (M1) ✅
│   │   └── trainer_seg_mlp.py (M2) ✅ NEW
│   ├── data/
│   │   ├── dataset.py (M1 sequences) ✅
│   │   ├── collate.py (M1 CTC) ✅
│   │   └── seg_dataset.py (M2 chars) ✅ NEW
│   └── segmentation.py ✅ NEW
├── api/
│   ├── routes_train.py (updated for M2) ✅
│   ├── routes_predict_ckpt.py (M1) ✅
│   └── routes_predict_seg.py (M2) ✅ NEW
└── app.py (updated - removed M3) ✅

frontend/
├── src/
│   ├── types/training.ts (M1/M2 only) ✅
│   ├── pages/Train/NewTraining.tsx (M2 updated) ✅
│   └── components/Train/AdvancedSettings.tsx (M2 settings) ✅
```

---

## 🚀 Как запустить:

### Backend:
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### Frontend:
```bash
cd frontend
npm run dev
# Откроется на http://127.0.0.1:5173 или 5174
```

### Создать M2 тренировку:
```bash
# 1. Открой http://127.0.0.1:5174/train/new
# 2. Выбери Model Type: M2 (Segmentation + MLP)
# 3. Установи параметры:
#    - Epochs: 5-10
#    - Batch Size: 128
#    - Max Samples: 1000
# 4. Start Training
# 5. Monitor в real-time
```

---

## 📈 Метрики M2:

### Training metrics:
- `train_loss` - CrossEntropyLoss на отдельных символах
- `train_char_acc` - Accuracy классификации символов

### Validation metrics:
- `val_char_acc` - Точность на отдельных символах
- `val_seq_acc` - Exact match accuracy на полных выражениях
- `val_seg_acc` - Процент выражений с правильным числом сегментов

### Best metrics tracked:
- `best_char_acc` - Лучшая character accuracy
- `best_seq_acc` - Лучшая sequence accuracy

---

## 🔍 Debugging M2:

### Если сегментация не работает:

```python
# Тестовый скрипт
from backend.core.segmentation import segment_characters
import numpy as np
from PIL import Image

img = Image.open("data/synth/train/images/00000.png").convert("L")
img_arr = np.array(img)

segments = segment_characters(img_arr, min_area=20, max_area=4000)
print(f"Found {len(segments)} segments")

for i, (char_img, bbox) in enumerate(segments):
    print(f"Segment {i}: bbox={bbox}, shape={char_img.shape}")
```

### Если классификация не работает:

```python
# Проверка MLP
from backend.core.models.seg_mlp import SegmentationOCR
import torch

model = SegmentationOCR(num_classes=len(TOKEN_LIST))
dummy_input = torch.randn(1, 1, 32, 32)  # Batch of 1 character
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Should be (1, num_classes)
```

---

## ⚡ Performance Tips:

1. **Batch Size:** Увеличьте до 256-512 для M2 (символы меньше чем sequences)
2. **Learning Rate:** Начните с 0.001, уменьшите при необходимости
3. **Epochs:** 5-10 достаточно для малых данных, 20-30 для полного датасета
4. **Max Samples:** Начните с 1000 для быстрой проверки концепции

---

## 🎓 Следующие шаги:

1. ✅ Протестировать M2 на малых данных (100-1000 samples)
2. ⏳ Оптимизировать параметры сегментации
3. ⏳ Добавить визуализацию сегментов в Monitor UI
4. ⏳ Сравнить M1 vs M2 на benchmark
5. ⏳ Документировать best practices для M2

---

**Система готова к использованию!** 🎉

См. подробности в:
- `M2_SEGMENTATION_GUIDE.md` - Руководство по M2
- `README.md` - Общий обзор
- `TRAINING_GUIDE.md` - Как тренировать модели
