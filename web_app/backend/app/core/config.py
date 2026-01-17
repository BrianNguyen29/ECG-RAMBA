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
    ENVIRONMENT: str = "development"  # development | staging | production
    
    # ==========================================================================
    # JWT Authentication
    # ==========================================================================
    # CRITICAL: Set this in .env for production! Auto-generated keys invalidate tokens on restart.
    SECRET_KEY: str = ""  # Will be auto-generated with warning if not set
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # ==========================================================================
    # Database - Supports SQLite (local) or PostgreSQL (Railway)
    # ==========================================================================
    DATABASE_URL: str = "sqlite+aiosqlite:///./ecg_ramba.db"
    
    # ==========================================================================
    # CORS - Allowed origins (comma-separated for multiple)
    # In production, only allow your actual domain (no localhost)
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
    
    # ==========================================================================
    # DeepTutor Integration (Phase 8)
    # ==========================================================================
    GEMINI_API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Allow extra env vars in .env without errors


# Singleton settings instance
settings = Settings()

# ==========================================================================
# SECRET_KEY Validation - Auto-generate with warning if not set
# ==========================================================================
if not settings.SECRET_KEY:
    import warnings
    _auto_key = secrets.token_urlsafe(32)
    # Use object.__setattr__ to bypass frozen model
    object.__setattr__(settings, 'SECRET_KEY', _auto_key)
    warnings.warn(
        "\n⚠️  SECRET_KEY not set in environment! Using auto-generated key.\n"
        "   This will invalidate all JWT tokens on restart.\n"
        "   Set SECRET_KEY in .env for production.\n",
        UserWarning
    )

# Production CORS validation
if settings.ENVIRONMENT == "production":
    if "localhost" in settings.CORS_ORIGINS:
        import warnings
        warnings.warn(
            "\n⚠️  CORS_ORIGINS contains 'localhost' in production mode!\n"
            "   This is a security risk. Update CORS_ORIGINS in .env.\n",
            UserWarning
        )

