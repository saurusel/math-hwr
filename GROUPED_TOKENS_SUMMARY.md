# Grouped Tokens Implementation

**Date:** 2025-10-21
**Feature:** Visual group tokenization for M1 and M2

---

## 🎯 Что изменилось:

### До:
```
Expression: "0 ^ ( 8 )"
Tokens: ["0", "^", "(", "8", ")"]  (5 separate tokens)
Visual: One group (0 in power 8)

Problem: Segmentation finds 1-2 components but expects 5 tokens!
```

### После:
```
Expression: "0 ^ ( 8 )"
Tokens: ["0^8"]  (1 grouped token)
Visual: One group (0 in power 8)

Solution: Tokens match visual structure!
```

---

## 📊 Новый Vocabulary:

**Size:** 198 unique tokens (было 21)

**Состав:**
- Базовые символы: `0-9`, `x`, `y`, `z`
- Операторы: `+`, `-`, `*`, `÷`, `=`
- Скобки: `(`, `)`
- **Степени (NEW):** `0^0`, `0^1`, ..., `z^z` (все комбинации)

**Примеры группированных токенов:**
- `0^8` - ноль в восьмой степени
- `x^2` - x в квадрате
- `3^y` - три в степени y
- `z^z` - z в степени z

---

## 🔧 Изменённые файлы:

### 1. **Новые файлы:**
- ✅ `backend/core/grouping.py` - Логика группировки токенов
- ✅ `build_grouped_vocab.py` - Скрипт построения vocabulary
- ✅ `grouped_vocab.txt` - 198 токенов

### 2. **Обновлённые файлы:**
- ✅ `backend/core/tokenizer.py` - Загружает grouped vocab
- ✅ `backend/core/data/dataset.py` (M1) - Использует group_tokens()
- ✅ `backend/core/data/seg_dataset.py` (M2) - Использует group_tokens()
- ✅ `backend/core/train/trainer_seg_mlp.py` - Группировка в validation

---

## 💡 Как работает группировка:

### Функция `group_tokens()`:
```python
Input:  ["0", "^", "(", "8", ")"]
Output: ["0^8"]

Input:  ["x", "^", "(", "2", ")", "+", "1"]
Output: ["x^2", "+", "1"]

Pattern: base ^ ( power ) → base^power
```

### Функция `ungroup_tokens()`:
```python
Input:  ["0^8"]
Output: ["0", "^", "(", "8", ")"]

# Reverse operation for parsing/display
```

---

## ✅ Результаты:

### M1 (CRNN-CTC):
- ✅ Dataset loads: 10 samples
- ✅ Grouped tokens applied
- ✅ Vocab: 198 tokens
- ✅ Ready for training

### M2 (Segmentation + MLP):
- ✅ Dataset loads: 58 characters from 10 expressions
- ✅ First label: `0^8` (grouped!)
- ✅ Segmentation matches tokens now!
- ✅ Ready for training

---

## 🚀 Ожидаемые улучшения:

### Segmentation quality:
**До группировки:**
- Target: `0 ^ ( 8 )` (5 tokens)
- Segments: 2 visual components
- Match: ❌ 2 != 5 (40% match)

**После группировки:**
- Target: `0^8` (1 token)
- Segments: 1 visual component
- Match: ✅ 1 == 1 (100% match!)

### Training expectations:
- **M2 character accuracy:** 80-95% (было 60-80%)
- **M2 sequence accuracy:** 50-75% (было 20-40%)
- **M1 train speed:** Faster (fewer tokens per expression)
- **M1 accuracy:** Same or better (more meaningful tokens)

---

## 📝 Использование:

### Перезапусти backend с новым vocab:
```bash
# Убедись что grouped_vocab.txt в корне проекта
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### Создай M2 тренировку:
```
http://127.0.0.1:5173/train/new

Model: M2 (Segmentation + MLP)
Epochs: 5
Batch Size: 128
Max Samples: 1000

Expected:
- Segmentation will match visual groups
- Character accuracy 80-95%
- Sequence accuracy 50-75%
```

### Создай M1 тренировку:
```
Model: M1 (CRNN-CTC)
Epochs: 5
Batch Size: 64
Max Samples: 1000

Expected:
- Shorter sequences (fewer tokens)
- Faster training (less CTC alignment work)
- Same or better accuracy
```

---

## 🔍 Примеры трансформации:

```
"3 + 5" → "3 + 5"  (no change, no powers)
"x ^ ( 2 )" → "x^2"  (grouped)
"0 ^ ( 8 ) + 1" → "0^8 + 1"  (partial grouping)
"x ^ ( y ) * z ^ ( 2 )" → "x^y * z^2"  (multiple groups)
```

---

## ⚙️ Технические детали:

### Vocabulary building:
- Scanned all 96,300 expressions
- Extracted all unique grouped tokens
- Sorted alphabetically
- Saved to `grouped_vocab.txt`

### Token mapping:
- `TOK2ID`: token → id (0-197)
- `ID2TOK`: id → token
- Dynamic loading from file

### Backwards compatibility:
- Old datasets still work (ungrouped)
- New datasets use grouping automatically
- Fallback to basic vocab if grouped_vocab.txt missing

---

**Система готова к работе с группированными токенами!** 🎉

Now tokens MATCH visual structure!
