# Math-HWR Starter (Windows + RTX 4080, Python 3.11)

Стартовый шаблон проекта для распознавания рукописных выражений (цифры, операции, x/y/z, степени `^`, дроби `frac ( { … } , { … } )`),
с FastAPI-бэкендом, WebSocket-стримом метрик и заглушками под обучение.

## Быстрый старт (Windows, PowerShell)
1) Установи Python 3.11 (если ещё не).
2) Открой PowerShell в корне репозитория и запусти:
   ```powershell
   .\backend\setup.ps1
   ```
3) Запусти сервер:
   ```powershell
   .\backend\run_backend.ps1
   ```
   Сервер поднимется на http://localhost:8000 .
4) Проверь WS-стрим: открой http://localhost:8000/docs , сначала вызови `POST /api/train/start`, затем подключись к WS `/api/train/stream?run_id=<id>` (можно через любой клиент, напр. "WebSocket King").

## Структура
- backend/app.py — FastAPI приложение
- backend/api/* — REST и WS маршруты (старт/стоп, предикт, список раннов)
- backend/core/* — утилиты (метрики, токенайзер, парсер-валидатор, устройство/GPU, события)
- backend/configs/*.yaml — дефолтные конфиги
- backend/scripts/* — генерация синтетики/оценка (заглушки)
- runs/ — артефакты и логи (не кладём в git)
- data/ — данные (synth/hand) (не кладём в git)

## Следующие шаги
- Stage 02: генератор синтетики (`scripts/make_synth_data.py`)
- Stage 03–04: фронтенд (React/Vite) и подключение к WS
- Stage 05–07: интеграция моделей (Baseline/CRNN/ViT-Dec), реальный поток метрик
- Stage 08: отчёт/оценка
