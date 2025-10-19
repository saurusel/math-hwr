


## Stage/02 — Подготовка символов и генерация синтетики

**Куда положить папки с классами** (из Kaggle):
Создай папку `data\symbols_kaggle\` и помести внутрь **папки-классы** с именами:
```
0 1 2 3 4 5 6 7 8 9
plus minus times divide equals
x y z
```
Внутри каждой — изображения символов (png/jpg/bmp/webp).

**Запуск генератора:**
```powershell
.\.venv\Scripts\activate
python .\backend\scripts\make_synth_data.py --symbols_dir data\symbols_kaggle --out data\synth --n 100000 --seed 20251019
```

Скрипт создаст:
```
data/synth/
  train/images/*.png + labels.jsonl
  val/images/*.png   + labels.jsonl
  test/images/*.png  + labels.jsonl
```

*Примечание:* структурные токены (`frac ( { … } , { … } )` и скобки экспонент) присутствуют **в метках**,
но на изображении визуализируются как 2D-верстка (горизонтальная черта дроби, верхний индекс для степени).
