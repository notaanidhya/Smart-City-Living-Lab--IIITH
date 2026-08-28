import React, { useEffect } from "react";
import { X, Layers, AlertCircle, FileText } from "lucide-react";
import ImageViewer from "./ImageViewer";
import DiagnosticsPanel from "./DiagnosticsPanel";
import MetricsMatrix from "./MetricsMatrix";

export default function DetailModal({ item, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  if (!item) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div className="modal-title mono">
            <FileText size={16} className="text-highlight" />
            <span>AUDIT INSPECTION: {item.filename}</span>
            <span className="text-muted">(ID: #{item.id})</span>
          </div>
          <button className="btn btn-ghost modal-close-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        {/* Modal Content */}
        <div className="modal-body-scroll">
          <div className="grid-2col modal-grid">
            <ImageViewer result={item} previewUrl={null} />
            <DiagnosticsPanel result={item} isAnalyzing={false} />
          </div>
          <div style={{ marginTop: "1.25rem" }}>
            <MetricsMatrix statistics={item.statistics} />
          </div>
        </div>
      </div>
    </div>
  );
}
