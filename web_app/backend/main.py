import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from utils import Predictor

app = FastAPI(title="Fake News Detection API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor
MODELS_DIR = "../../saved_models"
if not os.path.exists(MODELS_DIR):
    # Fallback for different deployment scenarios
    MODELS_DIR = "saved_models"

predictor = Predictor(MODELS_DIR)

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Serve static files (plots)
if os.path.exists(STATIC_DIR):
    app.mount("/plots", StaticFiles(directory=STATIC_DIR), name="plots")
else:
    print(f"Warning: Static directory not found at {STATIC_DIR}")

# Serve React frontend
if os.path.exists("web_app/backend/static/frontend"):
    app.mount("/", StaticFiles(directory="web_app/backend/static/frontend", html=True), name="frontend")


class NewsRequest(BaseModel):
    text: str
    model: str = "deberta"

@app.get("/")
def read_root():
    return {"message": "Fake News Detection API is running"}

@app.post("/predict")
def predict_news(request: NewsRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if request.model not in predictor.available_models:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not available. Choose from {predictor.available_models}")
    
    try:
        result = predictor.predict(request.text, request.model)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc() # Ini akan memunculkan error detail di terminal
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis")
def get_analysis_plots():
    plots = [f for f in os.listdir("web_app/backend/static") if f.endswith(".png")]
    return {"plots": plots}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
