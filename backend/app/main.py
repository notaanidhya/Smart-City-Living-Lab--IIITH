from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

from backend.app.schemas.schemas import HealthResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("image_quality_api")

models_state = {
    "loaded": False,
    "mlp_model": None,
    "autoencoder_model": None,
    "device": "cpu"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing Image Quality Assessment service...")
    # Model loading hook will be completed in Phase 5
    models_state["loaded"] = False
    logger.info("Service startup complete.")
    yield
    # Shutdown
    logger.info("Shutting down Image Quality Assessment service...")

app = FastAPI(
    title="AI Image Quality & Defect Detection API",
    description="Full-stack AI/CV service for evaluating visual image quality, detecting degradations, and localizing defects.",
    version="1.0.0",
    lifespan=lifespan
)

cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost:80,*")
cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error processing {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred during image processing."}
    )

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint exposing service and model loading status."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        models_loaded=models_state["loaded"],
        details={"environment": os.getenv("APP_ENV", "development")}
    )

@app.get("/", tags=["System"])
async def root():
    return {
        "message": "AI Image Quality & Defect Detection API is active.",
        "docs": "/docs",
        "health": "/api/health"
    }
