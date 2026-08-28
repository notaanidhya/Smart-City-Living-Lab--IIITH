from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class IssueDetail(BaseModel):
    type: str = Field(..., description="Issue type: blur, underexposure, overexposure, noise, corruption, defect")
    severity: str = Field(..., description="Severity level: low, medium, high")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score [0, 1]")
    details: Optional[str] = Field(None, description="Optional diagnostic details or feature trigger")

class AnalysisResponse(BaseModel):
    id: Optional[int] = None
    filename: str
    quality_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score (0-100)")
    quality_label: str = Field(..., description="ACCEPTABLE, DEGRADED, or DEFECTIVE")
    issues: List[IssueDetail] = Field(default_factory=list)
    statistics: Optional[Dict[str, Any]] = None
    image_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    processed_at: Optional[datetime] = None

class PaginatedResultsResponse(BaseModel):
    total: int
    page: int
    limit: int
    pages: int
    items: List[AnalysisResponse]

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    models_loaded: bool = False
    details: Optional[Dict[str, Any]] = None
