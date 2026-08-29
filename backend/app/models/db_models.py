import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from backend.app.db.session import Base

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)

class AnalysisRecord(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(64), nullable=True, index=True)
    filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True, index=True)
    upload_time = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    quality_score = Column(Float, nullable=False)
    quality_label = Column(String(50), nullable=False, index=True)
    issues = Column(JSON, nullable=False, default=list)
    statistics = Column(JSON, nullable=False, default=dict)
    image_url = Column(String(500), nullable=False)
    heatmap_url = Column(String(500), nullable=True)

    def to_dict(self):
        iso_str = None
        if self.upload_time:
            dt = self.upload_time
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            iso_str = dt.isoformat()

        return {
            "id": self.id,
            "session_id": self.session_id,
            "filename": self.filename,
            "quality_score": round(self.quality_score, 1),
            "quality_label": self.quality_label,
            "issues": self.issues,
            "statistics": self.statistics,
            "image_url": self.image_url,
            "heatmap_url": self.heatmap_url,
            "processed_at": iso_str
        }
