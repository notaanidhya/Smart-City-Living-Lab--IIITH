import React, { useState, useEffect } from "react";
import { Eye, Layers, Sliders, Image, Sparkles, Maximize2 } from "lucide-react";
import { getAssetUrl } from "../api/client";

export default function ImageViewer({ result, previewUrl }) {
  const [viewMode, setViewMode] = useState("overlay"); // "original", "heatmap", "overlay"
  const [overlayOpacity, setOverlayOpacity] = useState(70);
  const [imageMeta, setImageMeta] = useState({ width: null, height: null });

  const originalSrc = previewUrl || (result?.image_url ? getAssetUrl(result.image_url) : null);
  const heatmapSrc = result?.heatmap_url ? getAssetUrl(result.heatmap_url) : null;

  // Read natural dimensions
  const handleImageLoaded = (e) => {
    setImageMeta({
      width: e.target.naturalWidth,
      height: e.target.naturalHeight,
    });
  };

  return (
    <div className="workbench-panel image-viewer-panel">
      <div className="panel-header">
        <div className="panel-title">
          <Layers size={15} />
          <span>Spatial Inspection Viewport</span>
        </div>

        {/* View mode toggle tabs (Geometric / Monospace - No generic pills) */}
        {heatmapSrc && (
          <div className="view-mode-tabs mono">
            <button
              className={`mode-tab ${viewMode === "original" ? "active" : ""}`}
              onClick={() => setViewMode("original")}
            >
              Original
            </button>
            <button
              className={`mode-tab ${viewMode === "overlay" ? "active" : ""}`}
              onClick={() => setViewMode("overlay")}
            >
              Overlay ({overlayOpacity}%)
            </button>
            <button
              className={`mode-tab ${viewMode === "heatmap" ? "active" : ""}`}
              onClick={() => setViewMode("heatmap")}
            >
              Raw Heatmap
            </button>
          </div>
        )}
      </div>

      <div className="image-viewport-container">
        {originalSrc ? (
          <div className="image-canvas-wrapper">
            {/* Base Image */}
            <img
              src={viewMode === "heatmap" && heatmapSrc ? heatmapSrc : originalSrc}
              alt="Inspection Target"
              className="canvas-image base-layer"
              onLoad={handleImageLoaded}
            />

            {/* Overlay Layer (Shown only in overlay mode) */}
            {viewMode === "overlay" && heatmapSrc && (
              <img
                src={heatmapSrc}
                alt="Anomaly Heatmap Overlay"
                className="canvas-image overlay-layer"
                style={{ opacity: overlayOpacity / 100 }}
              />
            )}
          </div>
        ) : (
          <div className="empty-viewport mono">
            <Eye size={36} className="text-muted" />
            <span>Select or drop an image above to begin diagnostic inspection.</span>
          </div>
        )}
      </div>

      {/* Viewport Footer Telemetry & Controls */}
      {originalSrc && (
        <div className="viewport-footer">
          {/* Metadata telemetry */}
          <div className="viewport-meta mono">
            <span className="meta-item">
              <span className="text-muted">FILE:</span> {result?.filename || "Input Stream"}
            </span>
            {imageMeta.width && (
              <span className="meta-item">
                <span className="text-muted">RES:</span> {imageMeta.width}×{imageMeta.height} px
              </span>
            )}
            {result?.processed_at && (
              <span className="meta-item">
                <span className="text-muted">ANALYZED:</span>{" "}
                {new Date(result.processed_at).toLocaleTimeString()}
              </span>
            )}
          </div>

          {/* Opacity slider for Overlay Mode */}
          {heatmapSrc && viewMode === "overlay" && (
            <div className="opacity-slider-control mono">
              <Sliders size={13} className="text-secondary" />
              <span>Heatmap Blend:</span>
              <input
                type="range"
                min="10"
                max="100"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                className="opacity-slider"
              />
              <span className="slider-value">{overlayOpacity}%</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
