from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
import numpy as np
from app.core.model_loader import ModelLoader
from app.core.signal_processing import process_ecg

router = APIRouter()

class PredictionRequest(BaseModel):
    model_name: str
    signal_data: List[float] # Simple 1D signal for MVP

class PredictionResponse(BaseModel):
    diagnosis: str
    confidence: float
    model_used: str
    error: Optional[str] = None

@router.get("/models")
async def get_models():
    """Return list of available .pt models"""
    models = ModelLoader.get_available_models()
    if not models:
        return ["ModelChecking_v1.pt", "ResNet_ECG.pt"] # Return dummies if empty for UI dev
    return models

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Analyze ECG signal using the selected model.
    """
    # 1. Process Signal
    try:
        clean_signal = process_ecg(request.signal_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signal processing error: {str(e)}")

    # 2. Inference
    result = ModelLoader.predict(request.model_name, clean_signal)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Helper to parse uploaded CSV/JSON and return raw signal data to frontend.
    """
    try:
        content = await file.read()
        if file.filename.endswith(".json"):
            data = json.loads(content)
            # Assume simple list format -> [0.1, 0.2, ...]
            if isinstance(data, list):
                return {"signal": data}
            elif "lead_1" in data:
                return {"signal": data["lead_1"]}
        elif file.filename.endswith(".csv"):
            # Simple CSV parsing
            text = content.decode("utf-8")
            values = [float(x.strip()) for x in text.split(",") if x.strip()]
            return {"signal": values}
            
        return {"error": "Unsupported file format. Use JSON or CSV."}
    except Exception as e:
        return {"error": str(e)}
