---
title: TruthLens AI
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Fake News Detection Web App
... (rest of the content)

This is a full-stack web application for detecting fake news using models trained in the `fake_news_pipeline.ipynb`.

## Features
- **Real-time Prediction**: Input news text and get instant results from 4 different models (DeBERTa, RoBERTa, DistilRoBERTa, Bi-LSTM).
- **Analysis Dashboard**: View training history, confusion matrices, ROC curves, and EDA results directly from the UI.
- **SOTA Performance**: Leveraging DeBERTa-v3 for high-accuracy predictions.

## Tech Stack
- **Backend**: FastAPI (Python)
- **Frontend**: React + TypeScript + Vite
- **Styling**: Vanilla CSS
- **Models**: PyTorch + HuggingFace Transformers

## How to Run

### 1. Start the Backend
Navigate to the root directory and run:
```bash
python -m uvicorn web_app.backend.main:app --reload
```
The API will be available at `http://localhost:8000`.

### 2. Start the Frontend
Navigate to `web_app/frontend`:
```bash
cd web_app/frontend
npm install
npm run dev
```
The website will be available at `http://localhost:5173`.

## File Structure
- `web_app/backend/`: FastAPI source code and static analysis plots.
- `web_app/frontend/`: React source code and UI styles.
- `saved_models/`: Pre-trained model weights (used by backend).
