# Math HWR Frontend

Handwritten mathematical expression recognition frontend with Training & Models management.

Built with **React 18 + TypeScript + Vite + Tailwind + React Router + Zustand + Recharts**.

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 🌐 Development

App runs at `http://127.0.0.1:5173`

Backend must be at `http://127.0.0.1:8000`

## 📱 Features

### 🎨 Playground (Inference)
- **Canvas Drawing**: HiDPI-aware canvas with brush and eraser tools
- **Pan & Zoom**: Space + drag to pan, mouse wheel to zoom
- **Image Upload**: Drag & drop or file picker for PNG/JPG
- **Model Selection**: M1 (CRNN-CTC), M2 (mock), M3 (mock)
- **Real-time Recognition**: Inference via `/api/predict2/run`
- **Result Display**: Recognized text with copy button, tokens, preprocessed image
- **Toast Notifications**: User-friendly error and success messages

### 🧪 Training & Models (NEW!)
- **New Training**: Create training jobs with configurable hyperparameters
- **Live Monitoring**: Real-time SSE streaming of metrics, logs, and samples
- **Loss Curves**: Interactive charts (train/val loss, CER, WER, Exact Match)
- **Experiments**: List and manage all training runs
- **UTF-8 Support**: Math symbols (÷, ×, etc.) preserved throughout

**See `TRAIN_AND_MODELS_README.md` for full training system documentation.**

## 🗺️ Routes

- `/` - Playground (inference)
- `/train/new` - Create new training run
- `/train/monitor?job=<id>` - Monitor training job live
- `/train/experiments` - View all experiments

## 📁 Project Structure

```
src/
├── api/
│   ├── client.ts           # Base API client
│   ├── predict.ts          # Prediction endpoints
│   └── training.ts         # Training endpoints (NEW)
├── components/
│   ├── Layout/             # Navigation & layout (NEW)
│   ├── CanvasBoard/        # Drawing canvas
│   ├── Controls/           # UI controls
│   ├── ResultPanel/        # Results display
│   └── Toast/              # Notifications
├── pages/
│   ├── Playground.tsx      # Inference page
│   └── Train/              # Training pages (NEW)
│       ├── NewTraining.tsx
│       ├── Monitor.tsx
│       └── Experiments.tsx
├── services/
│   └── sseClient.ts        # SSE client (NEW)
├── stores/
│   └── trainingStore.ts    # Zustand store (NEW)
├── types/
│   └── training.ts         # Training types (NEW)
└── App.tsx                 # Router setup
```

## ⌨️ Keyboard Shortcuts

- **Space + Drag**: Pan canvas
- **Mouse Wheel**: Zoom (centered at cursor)

## ✅ Acceptance Criteria

### Playground
✅ Drawing works with exact cursor alignment at any DPI/zoom
✅ Eraser shows visible circular outline
✅ Zoom/pan work correctly, Reset View restores defaults
✅ Image upload displays without distortion
✅ M1 recognition returns correct results
✅ UTF-8 characters (÷, ×) render correctly
✅ Toast notifications for errors and success
✅ Static build works (`npm run build`)

### Training System (Sprint 1 MVP)
✅ Create training jobs via API
✅ Live monitoring with SSE streaming
✅ Real-time loss curves and metrics
✅ Live logs viewer
✅ Sample predictions display
✅ Experiments list with status

## 🔧 Tech Stack

- **Framework**: React 18
- **Language**: TypeScript
- **Build**: Vite
- **Styling**: Tailwind CSS
- **Routing**: React Router v6
- **State**: Zustand
- **Charts**: Recharts
- **Real-time**: SSE (Server-Sent Events)

## 📚 Documentation

- **Frontend Spec**: `FRONTEND_SPEC.md` (Playground specification)
- **Training Spec**: `Train-and-Models-Spec.md` (Full system specification)
- **Sprint 1 README**: `TRAIN_AND_MODELS_README.md` (Basic training system)
- **Sprint 2 README**: `SPRINT2_README.md` (Models & comparison)
- **Sprint 3 README**: `SPRINT3_README.md` (Advanced features & i18n)
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md` (Complete overview)

## 🗺️ Complete Route Map

- `/` - Playground (inference)
- `/train/new` - Create new training run
- `/train/monitor?job=<id>` - Monitor training job live
- `/train/experiments` - View all experiments with filters
- `/train/models?run=<id>&ckpt=<id>` - Models & checkpoints management

## 🚧 Roadmap

**Sprint 1 (✅ Complete):**
- ✅ Basic training form
- ✅ Live monitoring with SSE
- ✅ Experiments list
- ✅ Loss curves visualization

**Sprint 2 (✅ Complete):**
- ✅ Models management page
- ✅ Checkpoint promotion
- ✅ Quick Test feature (canvas/upload)
- ✅ Experiments comparison modal
- ✅ Filters and search for experiments
- ✅ Multi-select experiments

**Sprint 3 (✅ Complete):**
- ✅ Advanced hyperparameters form (M1/M2/M3)
- ✅ Run duplication
- ✅ Checkpoint deletion with safety
- ✅ Internationalization (EN/RU)
- ✅ Notes and tags system
- ✅ Artifact preview component

**Future (Optional):**
- Dark mode
- Saved filter presets
- RBAC and permissions
- More languages
- PDF/CSV export
