import React, { useState } from "react";
import Header from "./components/Header";
import UploadZone from "./components/UploadZone";
import ImageViewer from "./components/ImageViewer";
import DiagnosticsPanel from "./components/DiagnosticsPanel";
import MetricsMatrix from "./components/MetricsMatrix";
import HistoryTable from "./components/HistoryTable";
import { analyzeImage } from "./api/client";
import { AlertCircle, X } from "lucide-react";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("workspace");
  const [activeResult, setActiveResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  const handleFileSelected = async (file) => {
    if (!file) return;
    setErrorMsg(null);

    // Create local preview immediately
    const localUrl = URL.createObjectURL(file);
    setPreviewUrl(localUrl);
    setIsAnalyzing(true);

    try {
      const data = await analyzeImage(file);
      setActiveResult(data);
    } catch (err) {
      console.error("Analysis failed:", err);
      const detail = err.response?.data?.detail || err.message || "Failed to analyze image.";
      setErrorMsg(detail);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header with Navigation and Live Health Status */}
      <Header activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="main-content">
        {/* Global Error Banner */}
        {errorMsg && (
          <div className="error-banner mono">
            <div className="error-content">
              <AlertCircle size={16} className="text-status-defective" />
              <span>{errorMsg}</span>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={() => setErrorMsg(null)}>
              <X size={14} />
            </button>
          </div>
        )}

        {/* View Switcher */}
        {activeTab === "workspace" ? (
          <div className="workspace-view">
            {/* Upload Zone & Benchmark Sample Loaders */}
            <UploadZone onFileSelected={handleFileSelected} isAnalyzing={isAnalyzing} />

            {/* Split Screen Dual Workspace */}
            <div className="grid-2col workspace-grid">
              {/* Left Column: Spatial Viewport & Heatmap Comparator */}
              <div className="workspace-col-left">
                <ImageViewer result={activeResult} previewUrl={previewUrl} />
              </div>

              {/* Right Column: Real-time Telemetry & Issues Stream */}
              <div className="workspace-col-right">
                <DiagnosticsPanel result={activeResult} isAnalyzing={isAnalyzing} />
              </div>
            </div>

            {/* Bottom: 22-Metric Telemetry Matrix */}
            {activeResult && activeResult.statistics && (
              <div className="metrics-matrix-wrapper">
                <MetricsMatrix statistics={activeResult.statistics} />
              </div>
            )}
          </div>
        ) : (
          <HistoryTable />
        )}
      </main>
    </div>
  );
}
