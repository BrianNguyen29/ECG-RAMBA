from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from app.services.dsp import DSPService

router = APIRouter()

class SignalPayload(BaseModel):
    signal: List[float] # Single channel for simplicity first, or list of lists
    fs: float = 250.0

class FilterPayload(BaseModel):
    signal: List[float]
    fs: float = 250.0
    type: str # bandpass, lowpass, highpass, notch
    low: Optional[float] = None
    high: Optional[float] = None

@router.post("/analyze")
async def analyze_signal(payload: SignalPayload):
    """
    Compute Frequency Domain features (PSD) for the signal.
    """
    data = np.array(payload.signal)
    result = DSPService.compute_psd(data, payload.fs)
    return result

@router.post("/process")
async def process_signal(payload: FilterPayload):
    """
    Apply real-time filters to the signal.
    """
    data = np.array(payload.signal)
    
    # Validate filter params
    if payload.type == "bandpass" and (not payload.low or not payload.high):
        raise HTTPException(status_code=400, detail="Bandpass requires low and high")
    
    filtered = DSPService.apply_filter(
        data, 
        payload.fs, 
        payload.type, 
        payload.low, 
        payload.high
    )
    
    # Convert back to list
    return {"filtered_signal": filtered.tolist()}
