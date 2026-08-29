import React, { useState, useEffect } from "react";
import Header from "./components/Header";
import UploadZone from "./components/UploadZone";
import ImageViewer from "./components/ImageViewer";
import DiagnosticsPanel from "./components/DiagnosticsPanel";
import MetricsMatrix from "./components/MetricsMatrix";
import HistoryTable from "./components/HistoryTable";
import WalkthroughTour from "./components/WalkthroughTour";
import { analyzeImage, getPresetAnalysis } from "./api/client";
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

  const handlePresetSelected = async (preset) => {
    if (!preset) return;
    setErrorMsg(null);
    setPreviewUrl(preset.file);
    setIsAnalyzing(true);

    try {
      // 1. Fast path: load pre-computed telemetry directly from Neon DB (instant ~50ms)
      const data = await getPresetAnalysis(preset.id);
      setActiveResult(data);
    } catch (err) {
      console.warn("Fast preset fetch failed, falling back to direct analysis...", err);
      // Fallback: fetch blob and run standard analysis
      try {
        const response = await fetch(preset.file);
        const blob = await response.blob();
        const filename = preset.file.split("/").pop();
        const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
        const data = await analyzeImage(file);
        setActiveResult(data);
      } catch (fallbackErr) {
        const detail = fallbackErr.response?.data?.detail || fallbackErr.message || "Failed to load preset.";
        setErrorMsg(detail);
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTourLoadPreset = async (presetId = "defect") => {
    const presetFileMap = {
      clean: { id: "clean", file: "/samples/sample_pristine___clean.jpg" },
      defect: { id: "defect", file: "/samples/sample_synthetic_defect.jpg" },
      blur: { id: "blur", file: "/samples/sample_blur_defocus.jpg" },
      noise: { id: "noise", file: "/samples/sample_gaussian_noise.jpg" },
    };
    const targetPreset = presetFileMap[presetId] || presetFileMap["defect"];
    handlePresetSelected(targetPreset);
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
            <UploadZone
              onFileSelected={handleFileSelected}
              onPresetSelected={handlePresetSelected}
              isAnalyzing={isAnalyzing}
            />

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
        hasActiveResult={!!activeResult}
      />
    </div>
  );
}
