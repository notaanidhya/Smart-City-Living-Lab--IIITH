import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text, inspect
from backend.app.db.session import engine, Base
from backend.app.routers import analysis
from backend.app.services.inference import inference_service
from backend.app.schemas.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("image_quality_api")

UPLOAD_DIR = os.getenv(
    "UPLOAD_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
)
os.makedirs(os.path.join(UPLOAD_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_DIR, "heatmaps"), exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)

    # Automatic schema migration for existing deployments
    try:
        inspector = inspect(engine)
        if "analysis_results" in inspector.get_table_names():
            columns = [c["name"] for c in inspector.get_columns("analysis_results")]
            if "session_id" not in columns:
                logger.info("Migrating database: adding missing 'session_id' column to 'analysis_results' table...")
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE analysis_results ADD COLUMN session_id VARCHAR(64)"))
                    try:
                        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_analysis_results_session_id ON analysis_results (session_id)"))
                    except Exception:
                        pass
                logger.info("Database migration completed successfully.")
    except Exception as e:
        logger.warning(f"Auto-migration check encountered non-fatal error: {e}")

    logger.info("Loading PyTorch CV models into memory...")
    try:
        inference_service.load_models()
        logger.info("Models loaded successfully. Service ready for inference.")
    except Exception as e:
        logger.error(f"Failed to load models during startup: {e}", exc_info=True)

    yield
    
    logger.info("Shutting down Image Quality Assessment service...")

app = FastAPI(
    title="PixelShamer — AI Image Quality & Defect Detection API",
    description="Full-stack AI/CV REST API for evaluating digital image visual quality, detecting degradations, and localizing defects.",
    version="1.0.0",
    lifespan=lifespan
)

cors_origins_str = os.getenv("CORS_ORIGINS", "*")
if cors_origins_str == "*":
    cors_origins = ["*"]
    allow_creds = False
else:
    cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]
    allow_creds = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=allow_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

app.include_router(analysis.router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error processing {request.url.path}: {str(exc)}", exc_info=True)
    origin = request.headers.get("origin", "*")
    response = JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.api_route("/api/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["System"])
@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint exposing service readiness and model loading status."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        models_loaded=inference_service.is_ready,
        details={
            "environment": os.getenv("APP_ENV", "development"),
            "device": inference_service.device
        }
    )

FRONTEND_DIST = os.getenv("FRONTEND_DIST", "")

if FRONTEND_DIST and os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.api_route("/", methods=["GET", "HEAD"], tags=["System"])
    async def root():
        return {
            "message": "AI Image Quality & Defect Detection API is active.",
            "docs": "/docs",
            "health": "/api/health"
        }
