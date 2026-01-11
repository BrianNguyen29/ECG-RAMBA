"""
ECG-RAMBA Configuration Settings
================================
Centralized configuration using Pydantic Settings for type-safe environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import secrets


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ==========================================================================
    # App Settings
    # ==========================================================================
    APP_NAME: str = "ECG-RAMBA API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # ==========================================================================
    # JWT Authentication
    # ==========================================================================
    SECRET_KEY: str = secrets.token_urlsafe(32)  # Auto-generate if not set
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # ==========================================================================
    # Database - Supports SQLite (local) or PostgreSQL (Railway)
    # ==========================================================================
    DATABASE_URL: str = "sqlite+aiosqlite:///./ecg_ramba.db"
    
    # ==========================================================================
    # CORS - Allowed origins (comma-separated for multiple)
    # ==========================================================================
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    
    # ==========================================================================
    # Rate Limiting
    # ==========================================================================
    RATE_LIMIT_PREDICT: str = "30/minute"
    RATE_LIMIT_UPLOAD: str = "60/minute"
    
    # ==========================================================================
    # Model Settings
    # ==========================================================================
    MODEL_DEVICE: str = "cpu"
    PRELOAD_MODELS: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Singleton settings instance
settings = Settings()
