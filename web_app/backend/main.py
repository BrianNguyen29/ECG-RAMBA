# ECG-RAMBA Backend - FastAPI Application
# =========================================
# P0 Improvements: Startup preload, enhanced health check, structured logging

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ecg-ramba")

# OPTIMIZATION: Limit PyTorch threads to avoid CPU contention
import torch
# 4 threads is often a sweet spot for latency vs throughput on standard instances
torch.set_num_threads(4) 
logger.info(f"PyTorch CPU threads set to: {torch.get_num_threads()}")

# =============================================================================
# P0.3: Model Preload at Startup
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle events."""
    # Startup: Preload models and init database
    logger.info("=" * 60)
    logger.info("ECG-RAMBA Backend Starting...")
    logger.info("=" * 60)
    
    # DEBUG: Print all routes
    for route in app.routes:
        logger.info(f"Route: {route.path} [{route.name}]")

    try:
        # Phase 1: Initialize database
        from app.db.database import init_db, close_db
        await init_db()
        logger.info("✓ Database initialized")
        
        from app.core.model_loader import ecg_ramba, DEVICE
        from app.core.config import settings
        
        # =========================================================
        # MODEL LOADING STRATEGY (Configurable)
        # =========================================================
        available = ecg_ramba.get_available_models()
        
        if settings.PRELOAD_MODELS:
            # Eager loading: Load models at startup (slower boot, faster first request)
            logger.info(f"Preloading {len(available)} models (PRELOAD_MODELS=True)...")
            for model_name in available[:5]:
                try:
                    ecg_ramba.load_model(model_name)
                    logger.info(f"  ✓ Loaded: {model_name}")
                except Exception as e:
                    logger.warning(f"  ✗ Failed: {model_name} - {e}")
        else:
            # Lazy loading: Models load on first request (fast boot, slower first request)
            logger.info(f"Available models: {len(available)} (Lazy Loading Enabled)")
        
        logger.info(f"Device: {DEVICE}")
        logger.info(f"PCA Loaded: {ecg_ramba._pca is not None}")
        logger.info(f"MiniRocket Loaded: {ecg_ramba._rocket is not None}")
        logger.info("=" * 60)
        logger.info(f"Backend Ready! (Preload: {settings.PRELOAD_MODELS})")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("ECG-RAMBA Backend Shutting Down...")
    try:
        from app.db.database import close_db
        await close_db()
        logger.info("✓ Database closed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# =============================================================================
# Security Configuration
# =============================================================================
from app.core.config import settings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        # HSTS only in production (Railway provides HTTPS)
        if not settings.DEBUG:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

# Disable Swagger docs in production for security
app = FastAPI(
    title="ECG-RAMBA Classification API",
    description="AI-powered 12-lead ECG Analysis with Pure PyTorch Mamba2 (SSD)",
    version="2.0.0",
    lifespan=lifespan,
    # Security: Disable docs in production
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# =============================================================================
# CORS Configuration - Dynamic from environment
# =============================================================================
from app.core.config import settings

# Parse CORS origins from comma-separated env variable
origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # P2.3: Restricted methods
    allow_headers=["Content-Type", "Authorization"],  # P2.3: Restricted headers
)

# =============================================================================
# P2.4: Global Exception Handler - SECURITY HARDENED
# =============================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    # Log full error for debugging (server-side only)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # SECURITY: Never expose internal error details in production
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": "ERR_INTERNAL",
            # Only show error details in DEBUG mode
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred. Please try again."
        }
    )

# =============================================================================
# API Routes
# =============================================================================
from app.api import router as api_router, limiter, auth_router, history_router
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Register rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Phase 8: DeepTutor Router (Start before generic API)
from app.routers import tutor
logger.info(f"DEBUG: Tutor Router Routes: {len(tutor.router.routes)}")
for r in tutor.router.routes:
    logger.info(f"DEBUG: Tutor Route: {r.path}")
app.include_router(tutor.router, prefix="/api")

# Main API router
app.include_router(api_router, prefix="/api")

# Phase 1: Auth & History routers
app.include_router(auth_router, prefix="/api")
app.include_router(history_router, prefix="/api")

# Phase 9.3: AI Lab Assistant
from app.api.lab import router as lab_router
app.include_router(lab_router, prefix="/api/lab")

# =============================================================================
# P2.2: Prometheus Metrics
# =============================================================================
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize and expose metrics at /metrics
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    excluded_handlers=["/metrics", "/health"],
)
instrumentator.instrument(app).expose(app, include_in_schema=True, tags=["Monitoring"])

# =============================================================================
# Root & Health Endpoints
# =============================================================================
@app.get("/")
async def root():
    return {
        "service": "ECG-RAMBA Classification API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy"}

# =============================================================================
# P0.4: Enhanced Health Check with Model Status
# =============================================================================
@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with model and system status."""
    try:
        from app.core.model_loader import ecg_ramba, DEVICE, CLASSES
        
        # Get Mamba source
        try:
            from src.layers import MAMBA_SOURCE
        except ImportError:
            MAMBA_SOURCE = "Unknown"
        
        models_loaded = list(ecg_ramba._models.keys())
        
        return {
            "status": "healthy",
            "service": "ECG-RAMBA",
            "version": "2.0.0",
            "components": {
                "models_loaded": models_loaded,
                "models_count": len(models_loaded),
                "pca_ready": ecg_ramba._pca is not None,
                "rocket_ready": ecg_ramba._rocket is not None,
            },
            "runtime": {
                "device": str(DEVICE),
                "mamba_backend": MAMBA_SOURCE,
                "classes": CLASSES,
            },
            "endpoints": {
                "predict": "/api/predict",
                "predict_ensemble": "/api/predict/ensemble",
                "upload": "/api/upload",
                "info": "/api/info",
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "degraded",
            "error": str(e)
        }
