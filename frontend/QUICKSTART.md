# Quick Start Guide

Get Math HWR frontend running in 5 minutes.

## ⚡ Installation

```bash
cd frontend
npm install
```

## 🚀 Run Development Server

```bash
npm run dev
```

Open **http://127.0.0.1:5173**

**Note:** Backend must be running at **http://127.0.0.1:8000**

## 🎯 First Steps

### 1. Test Playground (Inference)

1. Navigate to **/** (Playground)
2. Draw "2 + 3" on canvas
3. Click **"Recognize"**
4. See result: `2 + 3`

### 2. Create Training Job

1. Click **"🧪 Training"** in navigation
2. Click **"+ New Training"**
3. Fill form:
   - Model: M1 (CRNN-CTC)
   - Run name: `my_first_run`
   - Epochs: 5 (for quick test)
4. Expand **"Advanced Settings"** (optional)
5. Click **"Start Training"**

### 3. Monitor Training

1. Redirected to `/train/monitor?job=<id>`
2. Watch live loss curves update
3. View logs streaming
4. See sample predictions
5. Add notes/tags (scroll down)

### 4. View Experiments

1. Navigate to **"/train/experiments"**
2. See all training runs
3. Use filters (search, model, status)
4. Select 2+ runs and click **"📊 Compare"**

### 5. Test Checkpoints

1. Click **"Models"** for any experiment
2. View checkpoints list
3. Select a checkpoint
4. Go to **"Quick Test"** tab
5. Draw expression and test

## 🌍 Switch Language

Click **EN** or **RU** in top-right corner.

## 📦 Build for Production

```bash
npm run build
```

Static files in `dist/` folder.

## 🐛 Troubleshooting

**Page won't load:**
- Check console (F12) for errors
- Verify old `.jsx` files deleted
- Try hard refresh: Ctrl+Shift+R

**Backend connection failed:**
- Ensure backend running at port 8000
- Check CORS settings
- Verify endpoint URLs in `src/api/`

**TypeScript errors:**
- Run `npm install` again
- Delete `node_modules` and reinstall
- Check `tsconfig.json` is correct

**Charts not showing:**
- Verify Recharts installed: `npm list recharts`
- Check browser console for errors
- Ensure metrics data has correct format

## 📚 Next Steps

- Read `IMPLEMENTATION_SUMMARY.md` for full feature list
- Review API contracts in Sprint READMEs
- Check backend integration checklists
- Explore component code in `src/`

## 🎉 You're Ready!

The frontend is fully functional. Backend integration is the next step.
