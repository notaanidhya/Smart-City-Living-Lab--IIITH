from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class IssueDetail(BaseModel):
    type: str = Field(..., description="Issue type: blur, underexposure, overexposure, noise, corruption, defect")
    severity: str = Field(..., description="Severity level: low, medium, high")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score [0, 1]")
    details: Optional[str] = Field(None, description="Optional diagnostic details or feature trigger")

class QualityStatistics(BaseModel):
    laplacian_variance: Optional[float] = None
    mean_brightness: Optional[float] = None
    rms_contrast: Optional[float] = None
    estimated_noise_sigma: Optional[float] = None
    saturation_mean: Optional[float] = None
    glcm_contrast: Optional[float] = None
    reconstruction_error: Optional[float] = None
    additional_metrics: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    id: Optional[int] = None
    filename: str
    quality_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score (0-100)")
    quality_label: str = Field(..., description="ACCEPTABLE, DEGRADED, or DEFECTIVE")
    issues: List[IssueDetail] = Field(default_factory=list)
    statistics: Optional[QualityStatistics] = None
    heatmap_url: Optional[str] = None
    processed_at: datetime = Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    models_loaded: bool = False
    details: Optional[Dict[str, Any]] = None

class AnalysisHistoryItem(BaseModel):
    id: int
    filename: str
    quality_score: float
    quality_label: str
    issue_count: int
    created_at: datetime
    thumbnail_url: Optional[str] = None
