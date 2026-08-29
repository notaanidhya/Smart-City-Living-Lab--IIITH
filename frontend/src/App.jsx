import React, { useState, useEffect } from "react";
import Header from "./components/Header";
import UploadZone from "./components/UploadZone";
import ImageViewer from "./components/ImageViewer";
import DiagnosticsPanel from "./components/DiagnosticsPanel";
import MetricsMatrix from "./components/MetricsMatrix";
import HistoryTable from "./components/HistoryTable";
import WalkthroughTour from "./components/WalkthroughTour";
import { analyzeImage } from "./api/client";
import { AlertCircle, X } from "lucide-react";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("workspace");
  const [activeResult, setActiveResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [isTourOpen, setIsTourOpen] = useState(false);

  // Auto-launch walkthrough for first-time visitors
  useEffect(() => {
    try {
      const hasSeenTour = localStorage.getItem("pixelshamer_tour_seen");
      if (!hasSeenTour) {
        const timer = setTimeout(() => {
          setIsTourOpen(true);
        }, 800);
        return () => clearTimeout(timer);
      }
    } catch (e) {}
  }, []);

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

  const handleTourLoadPreset = async (presetId = "defect") => {
    try {
      const presetFileMap = {
        clean: "/samples/sample_pristine___clean.jpg",
        defect: "/samples/sample_synthetic_defect.jpg",
        blur: "/samples/sample_blur_defocus.jpg",
        noise: "/samples/sample_gaussian_noise.jpg",
      };
      const filePath = presetFileMap[presetId] || "/samples/sample_synthetic_defect.jpg";
      const response = await fetch(filePath);
      if (!response.ok) throw new Error("Preset file not accessible");
      const blob = await response.blob();
      const filename = filePath.split("/").pop();
      const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
      handleFileSelected(file);
    } catch (err) {
      console.error("Failed to load tour preset:", err);
    }
  };

  return (
    <div className="app-container">
      {/* Header with Navigation and Live Health Status */}
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onStartTour={() => setIsTourOpen(true)}
      />

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

      {/* Interactive Walkthrough Tour */}
      <WalkthroughTour
        isOpen={isTourOpen}
        onClose={() => setIsTourOpen(false)}
        onSwitchTab={setActiveTab}
        onLoadPreset={handleTourLoadPreset}
      />
    </div>
  );
}
