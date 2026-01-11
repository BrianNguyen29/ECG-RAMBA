"""
ECG-RAMBA History Endpoints
===========================
Patient history and prediction records API.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from app.db.database import get_db
from app.db.models import Prediction, Patient
from app.core.security import get_current_user, TokenData


router = APIRouter(prefix="/history", tags=["History"])


# =============================================================================
# Response Models
# =============================================================================
class PredictionResponse(BaseModel):
    """Prediction history response model."""
    id: int
    diagnosis: str
    confidence: float
    probability_normal: Optional[float]
    probability_afib: Optional[float]
    probability_other: Optional[float]
    morphology_score: Optional[float]
    rhythm_score: Optional[float]
    inference_time: Optional[float]
    cached: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class PatientCreate(BaseModel):
    """Patient creation request."""
    patient_id: Optional[str] = None
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    notes: Optional[str] = None


class PatientResponse(BaseModel):
    """Patient response model."""
    id: int
    patient_id: Optional[str]
    name: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


# =============================================================================
# History Endpoints
# =============================================================================

@router.get("/predictions", response_model=List[PredictionResponse])
async def get_predictions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[TokenData] = Depends(get_current_user)
):
    """
    Get prediction history.
    
    If authenticated, returns user's predictions.
    If not authenticated, returns recent public predictions.
    
    Args:
        limit: Maximum number of results (1-100)
        offset: Pagination offset
        
    Returns:
        List of predictions
    """
    query = select(Prediction).order_by(desc(Prediction.created_at))
    
    # Filter by user if authenticated
    if current_user and current_user.user_id:
        query = query.where(Prediction.user_id == current_user.user_id)
    
    query = query.limit(limit).offset(offset)
    
    result = await db.execute(query)
    predictions = result.scalars().all()
    
    return predictions


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific prediction by ID.
    
    Args:
        prediction_id: Prediction ID
        
    Returns:
        Prediction details
        
    Raises:
        404: Prediction not found
    """
    result = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id)
    )
    prediction = result.scalar_one_or_none()
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    
    return prediction


@router.delete("/predictions/{prediction_id}")
async def delete_prediction(
    prediction_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
):
    """
    Delete a prediction record.
    
    Requires authentication. Users can only delete their own predictions.
    
    Args:
        prediction_id: Prediction ID to delete
        
    Returns:
        Success message
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    result = await db.execute(
        select(Prediction).where(
            Prediction.id == prediction_id,
            Prediction.user_id == current_user.user_id
        )
    )
    prediction = result.scalar_one_or_none()
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found or access denied"
        )
    
    await db.delete(prediction)
    await db.commit()
    
    return {"message": "Prediction deleted successfully"}


# =============================================================================
# Patient Endpoints
# =============================================================================

@router.post("/patients", response_model=PatientResponse)
async def create_patient(
    patient_data: PatientCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new patient record.
    
    Args:
        patient_data: Patient information
        
    Returns:
        Created patient
    """
    new_patient = Patient(
        patient_id=patient_data.patient_id,
        name=patient_data.name,
        age=patient_data.age,
        gender=patient_data.gender,
        notes=patient_data.notes
    )
    
    db.add(new_patient)
    await db.commit()
    await db.refresh(new_patient)
    
    return new_patient


@router.get("/patients", response_model=List[PatientResponse])
async def get_patients(
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all patients.
    
    Args:
        limit: Maximum number of results
        
    Returns:
        List of patients
    """
    result = await db.execute(
        select(Patient).order_by(desc(Patient.created_at)).limit(limit)
    )
    return result.scalars().all()


@router.get("/patients/{patient_id}/predictions", response_model=List[PredictionResponse])
async def get_patient_predictions(
    patient_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get all predictions for a specific patient.
    
    Args:
        patient_id: Patient ID
        
    Returns:
        List of predictions for the patient
    """
    result = await db.execute(
        select(Prediction)
        .where(Prediction.patient_id == patient_id)
        .order_by(desc(Prediction.created_at))
    )
    return result.scalars().all()


# =============================================================================
# Statistics Endpoint
# =============================================================================

@router.get("/stats")
async def get_stats(
    db: AsyncSession = Depends(get_db),
    current_user: Optional[TokenData] = Depends(get_current_user)
):
    """
    Get prediction statistics.
    
    Returns:
        Total predictions, by diagnosis, average confidence
    """
    from sqlalchemy import func
    
    # Total predictions
    query = select(func.count(Prediction.id))
    if current_user and current_user.user_id:
        query = query.where(Prediction.user_id == current_user.user_id)
    
    result = await db.execute(query)
    total = result.scalar()
    
    # Count by diagnosis
    query = select(Prediction.diagnosis, func.count(Prediction.id))
    if current_user and current_user.user_id:
        query = query.where(Prediction.user_id == current_user.user_id)
    query = query.group_by(Prediction.diagnosis)
    
    result = await db.execute(query)
    by_diagnosis = {row[0]: row[1] for row in result.all()}
    
    # Average confidence
    query = select(func.avg(Prediction.confidence))
    if current_user and current_user.user_id:
        query = query.where(Prediction.user_id == current_user.user_id)
    
    result = await db.execute(query)
    avg_confidence = result.scalar() or 0
    
    return {
        "total_predictions": total,
        "by_diagnosis": by_diagnosis,
        "average_confidence": round(avg_confidence, 3)
    }
