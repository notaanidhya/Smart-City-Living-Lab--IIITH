import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from backend.app.db.session import Base

class AnalysisRecord(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True, index=True)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)
    quality_score = Column(Float, nullable=False)
    quality_label = Column(String(50), nullable=False, index=True)
    issues = Column(JSON, nullable=False, default=list)
    statistics = Column(JSON, nullable=False, default=dict)
    image_url = Column(String(500), nullable=False)
    heatmap_url = Column(String(500), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "quality_score": round(self.quality_score, 1),
            "quality_label": self.quality_label,
            "issues": self.issues,
            "statistics": self.statistics,
            "image_url": self.image_url,
            "heatmap_url": self.heatmap_url,
            "processed_at": self.upload_time.isoformat() if self.upload_time else None
        }
