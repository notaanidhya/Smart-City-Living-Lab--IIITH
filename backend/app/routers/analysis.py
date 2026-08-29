import os
import io
import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query, Header, status
from sqlalchemy.orm import Session
from PIL import Image

from backend.app.db.session import get_db
from backend.app.models.db_models import AnalysisRecord
from backend.app.schemas.schemas import AnalysisResponse, PaginatedResultsResponse
from backend.app.services.inference import inference_service

logger = logging.getLogger("image_quality_api.router")

router = APIRouter(prefix="/api", tags=["Analysis"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15")) * 1024 * 1024
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads"))

@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def analyze_image_endpoint(
    image: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, WEBP, BMP)"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    db: Session = Depends(get_db)
):
    """
    Evaluates visual quality of an uploaded image, detects degradations,
    localizes defects with a spatial heatmap, and stores the results.
    """
    # 1. Validate extension
    filename = image.filename or "uploaded_image.jpg"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 2. Read bytes & validate size
    try:
        content = await image.read()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to read uploaded file.")

    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty (0 bytes).")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
        )

    # 3. Validate image integrity via PIL verify()
    try:
        pil_img = Image.open(io.BytesIO(content))
        pil_img.verify()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or invalid image file. Header check failed."
        )

    # 4. Execute AI / CV Inference Pipeline
    try:
        result = inference_service.analyze_image(
            image_bytes=content,
            original_filename=filename,
            upload_dir=UPLOAD_DIR
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        logger.error(f"Inference error processing '{filename}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error occurred during computer vision feature extraction and model inference."
        )

    # 5. Persist to Database
    db_record = AnalysisRecord(
        session_id=x_session_id or "default_session",
        filename=result["filename"],
        stored_filename=result["stored_filename"],
        quality_score=result["quality_score"],
        quality_label=result["quality_label"],
        issues=result["issues"],
        statistics=result["statistics"],
        image_url=result["image_url"],
        heatmap_url=result["heatmap_url"]
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return AnalysisResponse(
        id=db_record.id,
        session_id=db_record.session_id,
        filename=db_record.filename,
        quality_score=db_record.quality_score,
        quality_label=db_record.quality_label,
        issues=db_record.issues,
        statistics=db_record.statistics,
        image_url=db_record.image_url,
        heatmap_url=db_record.heatmap_url,
        processed_at=db_record.upload_time
    )

@router.get("/results", response_model=PaginatedResultsResponse)
def list_results(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    quality_label: Optional[str] = Query(None, description="Filter by quality label (ACCEPTABLE, DEGRADED, DEFECTIVE)"),
    scope: str = Query("session", description="Filter by 'session' (user private history) or 'global' (all history)"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    db: Session = Depends(get_db)
):
    """Retrieves paginated history of previous image evaluations."""
    query = db.query(AnalysisRecord)
    if scope == "session" and x_session_id:
        query = query.filter(AnalysisRecord.session_id == x_session_id)
    if quality_label:
        query = query.filter(AnalysisRecord.quality_label == quality_label.upper())

    total = query.count()
    total_pages = max(1, (total + limit - 1) // limit)
    offset = (page - 1) * limit

    records = query.order_by(AnalysisRecord.upload_time.desc()).offset(offset).limit(limit).all()

    items = [
        AnalysisResponse(
            id=r.id,
            session_id=r.session_id,
            filename=r.filename,
            quality_score=round(r.quality_score, 1),
            quality_label=r.quality_label,
            issues=r.issues,
            statistics=r.statistics,
            image_url=r.image_url,
            heatmap_url=r.heatmap_url,
            processed_at=r.upload_time
        )
        for r in records
    ]

    return PaginatedResultsResponse(
        total=total,
        page=page,
        limit=limit,
        pages=total_pages,
        items=items
    )

@router.get("/results/{record_id}", response_model=AnalysisResponse)
def get_result_detail(record_id: int, db: Session = Depends(get_db)):
    """Retrieves full analysis result details by ID."""
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis result with ID {record_id} not found."
        )

    return AnalysisResponse(
        id=record.id,
        filename=record.filename,
        quality_score=round(record.quality_score, 1),
        quality_label=record.quality_label,
        issues=record.issues,
        statistics=record.statistics,
        image_url=record.image_url,
        heatmap_url=record.heatmap_url,
        processed_at=record.upload_time
    )

@router.delete("/results/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_result(record_id: int, db: Session = Depends(get_db)):
    """Deletes an analysis result from history."""
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis result with ID {record_id} not found."
        )
    db.delete(record)
    db.commit()
    return None
