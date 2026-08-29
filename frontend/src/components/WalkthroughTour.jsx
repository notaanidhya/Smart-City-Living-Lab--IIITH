import React, { useState, useEffect, useRef } from "react";
import {
  Layers,
  Sliders,
  Activity,
  CheckCircle2,
  ChevronRight,
  ChevronLeft,
  X,
  HelpCircle,
  FileImage,
  Maximize2,
  Terminal,
  Zap
} from "lucide-react";
import "./WalkthroughTour.css";

const TOUR_STEPS = [
  {
    id: "welcome",
    targetId: null, // Center modal
    placement: "center",
    title: "SYSTEM INITIALIZATION",
    subtitle: "PixelShamer • Hybrid AI Quality & Defect Detection",
    description: (
      <>
        Welcome to <strong>PixelShamer</strong>, an industrial-grade computer vision system designed to evaluate digital image quality, diagnose multi-label degradations, and localize physical defects with zero external API dependencies.
      </>
    ),
    features: [
      "Dual-Engine: 22 deterministic metrics + PyTorch ConvAutoencoder",
      "High-definition 256×256 spatial anomaly heatmaps",
      "PCHIP strictly monotonic quality calibration (0–100 Index)",
      "Client session isolation with persistent PostgreSQL audit trail"
    ],
    actionLabel: "Begin Tour",
  },
  {
    id: "presets",
    targetId: "tour-preset-chips",
    placement: "bottom",
    title: "BENCHMARK PRESET CHIPS",
    subtitle: "Instant Testing Without File Uploads",
    description: (
      <>
        Test the system instantly! Click any of the <strong>8 curated benchmark presets</strong> to evaluate specific degradation families (Nominal Clean, Physical Scratch, Defocus Blur, Exposure, Sensor Noise, or Multi-Degraded).
      </>
    ),
    features: [
      "Zero-latency demonstration of all 6 degradation families",
      "Evaluates continuous scoring and issue classification live",
      "Pre-loaded from the official 1,280-image benchmark suite"
    ],
    actionLabel: "Try Physical Defect",
    onAction: (helpers) => {
      helpers.loadPresetSample("defect");
    },
  },
  {
    id: "dropzone",
    targetId: "tour-dropzone",
    placement: "bottom",
    title: "STREAM & FILE INGESTION",
    subtitle: "Drag & Drop, Browse, or Clipboard Paste",
    description: (
      <>
        Upload your own inspection images via <strong>drag-and-drop</strong>, file browsing, or by pressing <strong>Ctrl+V</strong> to paste directly from your clipboard.
      </>
    ),
    features: [
      "Supports JPEG, PNG, WEBP, and BMP formats (up to 15 MB)",
      "Strict PIL header integrity check prevents corrupted uploads",
      "In-memory stream processing with zero disk leakage"
    ],
  },
  {
    id: "viewport",
    targetId: "tour-viewport",
    placement: "right",
    title: "3-MODE SPATIAL COMPARATOR",
    subtitle: "Original, Jet Overlay & Raw Heatmap",
    description: (
      <>
        Inspect anomalies with spatial precision. Toggle between:
        <br />
        • <strong>Original</strong>: Full-resolution raw capture.
        <br />
        • <strong>Overlay</strong>: Jet colormap blend with an adjustable opacity slider.
        <br />
        • <strong>Raw Heatmap</strong>: Unblended autoencoder reconstruction residual.
      </>
    ),
    features: [
      "Jet colormap: Red = Anomaly peak, Blue = Nominal surface",
      "4-Quadrant tiling preserves high-definition hairline scratches",
      "Full-screen magnification support"
    ],
  },
  {
    id: "diagnostics",
    targetId: "tour-diagnostics",
    placement: "left",
    title: "CONTINUOUS SCORE & TRIAGE",
    subtitle: "PCHIP Calibration & Compound Penalty",
    description: (
      <>
        The <strong>Quality Score (0–100)</strong> is derived from multi-label issue confidences fused with autoencoder reconstruction error. Scores are calibrated via <strong>PCHIP Monotonic Splines</strong> to eliminate artificial score plateaus.
      </>
    ),
    features: [
      "Triage Status: ACCEPTABLE (≥75), DEGRADED (40–74), DEFECTIVE (<40)",
      "Diminishing-returns compounding for multi-issue co-occurrences",
      "Per-issue severity categorization (low, medium, high)"
    ],
  },
  {
    id: "metrics",
    targetId: "tour-metrics-matrix",
    placement: "top",
    title: "22-METRIC TELEMETRY MATRIX",
    subtitle: "Deterministic Computer Vision Diagnostics",
    description: (
      <>
        Examine deep signal processing metrics grouped into 7 mathematical families: <strong>Sharpness</strong> (Laplacian, FFT), <strong>Exposure</strong>, <strong>Contrast</strong>, <strong>Immerkær Noise</strong>, <strong>GLCM Texture</strong>, and <strong>DCT Blockiness</strong>.
      </>
    ),
    features: [
      "Exposes exact numerical indicators for explainable inspection",
      "Reference-free noise sigma estimation (ISO 15739 standard)",
      "Expandable family cards with normal ranges and physical units"
    ],
  },
  {
    id: "history",
    targetId: "tour-history-nav",
    placement: "bottom",
    title: "AUDIT TRAIL & PRIVACY",
    subtitle: "Session Isolation & Global Feed",
    description: (
      <>
        Every analysis is logged in the database. Switch to <strong>Audit History</strong> to review previous inspections. <strong>My Session</strong> keeps your tests private, while <strong>Global Feed</strong> shows public benchmark runs.
      </>
    ),
    features: [
      "Full pagination with sorting and status filtering",
      "Inspect detailed diagnostic snapshots for past records",
      "Exportable audit logs for quality compliance"
    ],
    actionLabel: "Finish Walkthrough",
  },
];

export default function WalkthroughTour({
  isOpen,
  onClose,
  onSwitchTab,
  onLoadPreset,
}) {
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [targetRect, setTargetRect] = useState(null);
  const [popoverPos, setPopoverPos] = useState({ top: "50%", left: "50%", transform: "translate(-50%, -50%)" });
  const popoverRef = useRef(null);

  const step = TOUR_STEPS[currentStepIdx];
  const isFirstStep = currentStepIdx === 0;
  const isLastStep = currentStepIdx === TOUR_STEPS.length - 1;

  // Calculate target element rect & position popover
  const updatePosition = () => {
    if (!isOpen || !step) return;

    if (!step.targetId) {
      setTargetRect(null);
      setPopoverPos({
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        position: "fixed",
      });
      return;
    }

    const el = document.getElementById(step.targetId);
    if (!el) {
      setTargetRect(null);
      setPopoverPos({
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        position: "fixed",
      });
      return;
    }

    // Scroll target into view
    el.scrollIntoView({ behavior: "smooth", block: "center" });

    const rect = el.getBoundingClientRect();
    const padding = 8;

    setTargetRect({
      top: rect.top - padding,
      left: rect.left - padding,
      width: rect.width + padding * 2,
      height: rect.height + padding * 2,
    });

    const popoverWidth = 420;
    const popoverHeight = 350;
    const placement = step.placement || "bottom";

    // Desktop 2-column layout checks (no overlap)
    if (window.innerWidth >= 960) {
      if (placement === "left") {
        // Place strictly to the LEFT of the target panel (e.g. over viewport)
        let left = rect.left - popoverWidth - 24;
        if (left < 16) left = 16;
        let top = Math.max(16, Math.min(window.innerHeight - popoverHeight - 16, rect.top + 20));
        setPopoverPos({ top: `${top}px`, left: `${left}px`, transform: "none", position: "fixed" });
        return;
      } else if (placement === "right") {
        // Place strictly to the RIGHT of the target panel (e.g. over diagnostics)
        let left = rect.right + 24;
        if (left + popoverWidth > window.innerWidth - 16) left = window.innerWidth - popoverWidth - 16;
        let top = Math.max(16, Math.min(window.innerHeight - popoverHeight - 16, rect.top + 20));
        setPopoverPos({ top: `${top}px`, left: `${left}px`, transform: "none", position: "fixed" });
        return;
      } else if (placement === "top") {
        // Place strictly ABOVE the target panel (e.g. above metrics matrix)
        let top = rect.top - popoverHeight - 24;
        if (top < 16) top = 16;
        let left = Math.max(16, Math.min(window.innerWidth - popoverWidth - 16, rect.left + (rect.width - popoverWidth) / 2));
        setPopoverPos({ top: `${top}px`, left: `${left}px`, transform: "none", position: "fixed" });
        return;
      } else if (placement === "bottom") {
        // Place strictly BELOW the target panel (e.g. below presets)
        let top = rect.bottom + 20;
        if (top + popoverHeight > window.innerHeight - 16) top = window.innerHeight - popoverHeight - 16;
        let left = Math.max(16, Math.min(window.innerWidth - popoverWidth - 16, rect.left + (rect.width - popoverWidth) / 2));
        setPopoverPos({ top: `${top}px`, left: `${left}px`, transform: "none", position: "fixed" });
        return;
      }
    }

    // Small screen / mobile fallback
    let top = rect.bottom + 16;
    if (top + popoverHeight > window.innerHeight - 16) {
      top = Math.max(16, rect.top - popoverHeight - 16);
    }
    let left = Math.max(16, (window.innerWidth - popoverWidth) / 2);
    setPopoverPos({ top: `${top}px`, left: `${left}px`, transform: "none", position: "fixed" });
  };

  useEffect(() => {
    if (!isOpen) return;

    // Handle view tabs during tour
    if (step.id === "history") {
      onSwitchTab?.("workspace"); // Keep in workspace to highlight history button
    }

    const timer = setTimeout(updatePosition, 120);
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);

    return () => {
      clearTimeout(timer);
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [currentStepIdx, isOpen]);

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        handleClose();
      } else if (e.key === "ArrowRight" || e.key === "Enter") {
        handleNext();
      } else if (e.key === "ArrowLeft") {
        handlePrev();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, currentStepIdx]);

  const handleNext = () => {
    if (isLastStep) {
      handleClose();
    } else {
      setCurrentStepIdx((prev) => prev + 1);
    }
  };

  const handlePrev = () => {
    if (!isFirstStep) {
      setCurrentStepIdx((prev) => prev - 1);
    }
  };

  const handleClose = () => {
    try {
      localStorage.setItem("pixelshamer_tour_seen", "true");
    } catch (e) {}
    setCurrentStepIdx(0);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="tour-overlay-container">
      {/* SVG Backdrop with smooth spotlight cutout */}
      <svg className="tour-backdrop-svg">
        <defs>
          <mask id="tour-spotlight-mask">
            <rect x="0" y="0" width="100vw" height="100vh" fill="white" />
            {targetRect && (
              <rect
                x={targetRect.left}
                y={targetRect.top}
                width={targetRect.width}
                height={targetRect.height}
                rx="6"
                ry="6"
                fill="black"
              />
            )}
          </mask>
        </defs>
        <rect
          x="0"
          y="0"
          width="100vw"
          height="100vh"
          className="tour-mask-bg"
          mask="url(#tour-spotlight-mask)"
        />
      </svg>

      {/* Target outline pulse ring */}
      {targetRect && (
        <div
          className="tour-spotlight-pulse"
          style={{
            top: `${targetRect.top}px`,
            left: `${targetRect.left}px`,
            width: `${targetRect.width}px`,
            height: `${targetRect.height}px`,
          }}
        />
      )}

      {/* Popover Card */}
      <div
        ref={popoverRef}
        className="tour-popover"
        style={popoverPos}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="tour-popover-header">
          <div className="tour-step-badge mono">
            <span className="tour-step-dot"></span>
            <span>STEP {String(currentStepIdx + 1).padStart(2, "0")} / {String(TOUR_STEPS.length).padStart(2, "0")}</span>
          </div>
          <button
            className="tour-close-btn"
            onClick={handleClose}
            title="Exit Walkthrough (Esc)"
          >
            <X size={15} />
          </button>
        </div>

        {/* Title & Subtitle */}
        <div className="tour-title mono">
          <Terminal size={15} className="text-highlight" />
          <span>{step.title}</span>
        </div>
        {step.subtitle && (
          <div className="text-muted mono" style={{ fontSize: "0.75rem", marginBottom: "0.6rem" }}>
            {step.subtitle}
          </div>
        )}

        {/* Description */}
        <div className="tour-description">{step.description}</div>

        {/* Feature Highlights */}
        {step.features && (
          <ul className="tour-feature-list">
            {step.features.map((feat, i) => (
              <li key={i} className="tour-feature-item">
                <span className="tour-feature-bullet">▸</span>
                <span>{feat}</span>
              </li>
            ))}
          </ul>
        )}

        {/* Interactive Action Box (if step has an action) */}
        {step.onAction && (
          <div className="tour-action-box mono">
            <span className="tour-action-text">Interactive Demonstration:</span>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => step.onAction({ loadPresetSample: onLoadPreset })}
              style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem" }}
            >
              <Zap size={12} className="text-highlight" />
              <span>{step.actionLabel || "Run Action"}</span>
            </button>
          </div>
        )}

        {/* Footer Navigation */}
        <div className="tour-popover-footer">
          {/* Step Progress Pills */}
          <div className="tour-progress-bar">
            {TOUR_STEPS.map((_, i) => (
              <div
                key={i}
                className={`tour-progress-pill ${
                  i === currentStepIdx
                    ? "active"
                    : i < currentStepIdx
                    ? "completed"
                    : ""
                }`}
                onClick={() => setCurrentStepIdx(i)}
                style={{ cursor: "pointer" }}
                title={`Go to step ${i + 1}`}
              />
            ))}
          </div>

          {/* Nav Buttons */}
          <div className="tour-nav-buttons mono">
            {!isFirstStep && (
              <button
                className="btn btn-secondary btn-sm"
                onClick={handlePrev}
              >
                <ChevronLeft size={13} />
                <span>Prev</span>
              </button>
            )}
            <button
              className="btn btn-primary btn-sm"
              onClick={handleNext}
            >
              <span>{isLastStep ? "Finish" : isFirstStep ? "Start" : "Next"}</span>
              <ChevronRight size={13} />
            </button>
          </div>
        </div>

        {/* Hotkey hint */}
        <div className="tour-hotkey-hint mono">
          <span><kbd className="tour-kbd">Esc</kbd> Exit</span>
          <span>•</span>
          <span><kbd className="tour-kbd">→</kbd> Next</span>
          <span>•</span>
          <span><kbd className="tour-kbd">←</kbd> Prev</span>
        </div>
      </div>
    </div>
  );
}
