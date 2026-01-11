"""
ECG-RAMBA Database Models
=========================
SQLAlchemy ORM models for User, Patient, and Prediction entities.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship, DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class."""
    pass


class User(Base):
    """
    User model for authentication.
    
    Attributes:
        id: Primary key
        username: Unique username
        email: User email (optional)
        hashed_password: Bcrypt hashed password
        role: User role (user, admin, doctor)
        is_active: Account status
        created_at: Creation timestamp
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    role = Column(String(20), default="user")  # user, admin, doctor
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="user")


class Patient(Base):
    """
    Patient record model.
    
    Attributes:
        id: Primary key
        patient_id: External patient ID (hospital system)
        name: Patient name
        age: Patient age
        gender: Patient gender
        notes: Clinical notes
        created_at: Creation timestamp
    """
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), unique=True, index=True, nullable=True)
    name = Column(String(100), nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(10), nullable=True)  # M, F, Other
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="patient")


class Prediction(Base):
    """
    Prediction result model.
    
    Stores each ECG analysis result with:
    - Diagnosis and confidence
    - Signal hash for deduplication
    - Disentanglement scores (morphology/rhythm)
    - Reference to user and patient
    
    Attributes:
        id: Primary key
        user_id: User who made the prediction
        patient_id: Associated patient (optional)
        diagnosis: Predicted diagnosis label
        confidence: Confidence score (0-1)
        probability_normal: Normal sinus probability
        probability_afib: Atrial fibrillation probability
        probability_other: Other arrhythmia probability
        morphology_score: Disentanglement morphology score
        rhythm_score: Disentanglement rhythm score
        signal_hash: SHA256 hash of signal for caching
        inference_time: Model inference time in seconds
        created_at: Creation timestamp
    """
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    
    # Prediction results
    diagnosis = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    probability_normal = Column(Float, nullable=True)
    probability_afib = Column(Float, nullable=True)
    probability_other = Column(Float, nullable=True)
    
    # Disentanglement scores
    morphology_score = Column(Float, nullable=True)
    rhythm_score = Column(Float, nullable=True)
    
    # Metadata
    signal_hash = Column(String(64), index=True, nullable=True)
    inference_time = Column(Float, nullable=True)
    cached = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="predictions")
    patient = relationship("Patient", back_populates="predictions")
    
    def to_dict(self):
        """Convert prediction to dictionary for API response."""
        return {
            "id": self.id,
            "diagnosis": self.diagnosis,
            "confidence": self.confidence,
            "probability": {
                "normal": self.probability_normal,
                "afib": self.probability_afib,
                "other": self.probability_other,
            },
            "disentanglement": {
                "morphology_score": self.morphology_score,
                "rhythm_score": self.rhythm_score,
            },
            "inference_time": self.inference_time,
            "cached": self.cached,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
