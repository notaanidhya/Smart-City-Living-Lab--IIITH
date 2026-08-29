import React from "react";
import { Gauge, AlertTriangle, CheckCircle2, XCircle, ShieldAlert, Cpu } from "lucide-react";

export default function DiagnosticsPanel({ result, isAnalyzing }) {
  if (isAnalyzing) {
    return (
      <div className="workbench-panel diagnostics-panel" id="tour-diagnostics">
        <div className="panel-header">
          <div className="panel-title">
            <Cpu size={15} className="spin text-highlight" />
            <span>Neural Inference in Progress...</span>
          </div>
        </div>
        <div className="analyzing-skeleton mono">
          <div className="skeleton-line-long"></div>
          <div className="skeleton-line-short"></div>
          <div className="skeleton-grid"></div>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="workbench-panel diagnostics-panel" id="tour-diagnostics">
        <div className="panel-header">
          <div className="panel-title">
            <Gauge size={15} />
            <span>Real-time Telemetry & Diagnostics</span>
          </div>
        </div>
        <div className="empty-diagnostics mono">
          <Cpu size={32} className="text-muted" />
          <span>Awaiting image input. Load a sample preset or drop an image to execute assessment.</span>
        </div>
      </div>
    );
  }

  const score = result.quality_score ?? 0;
  const label = result.quality_label || "ACCEPTABLE";
  const issues = result.issues || [];

  const getStatusIcon = (lbl) => {
    switch (lbl) {
      case "ACCEPTABLE":
        return <CheckCircle2 size={16} className="text-status-acceptable" />;
      case "DEGRADED":
        return <AlertTriangle size={16} className="text-status-degraded" />;
      case "DEFECTIVE":
        return <XCircle size={16} className="text-status-defective" />;
      default:
        return null;
    }
  };

  return (
    <div className="workbench-panel diagnostics-panel" id="tour-diagnostics">
      <div className="panel-header">
        <div className="panel-title">
          <Gauge size={15} />
          <span>Quality Telemetry & Decision Engine</span>
        </div>
        <span className={`status-tag ${label.toLowerCase()} mono`}>
          {getStatusIcon(label)}
          <span>{label}</span>
        </span>
      </div>

      <div className="panel-body">
        {/* Score Readout Hero */}
        <div className="score-hero-container">
          <div className="score-main-block">
            <div className="score-numerical-hero mono">
              <span className="score-digits text-highlight">{score.toFixed(1)}</span>
              <span className="score-max text-muted">/100</span>
            </div>
            <div className="score-explanation mono">
              <span className="score-summary-title">COMPOSITE QUALITY INDEX</span>
              <span className="score-formula-note text-secondary">
                Fused from 70% MLP multi-label classifiers + 30% Convolutional Autoencoder spatial residuals.
              </span>
            </div>
          </div>

          {/* Segmented Quality Progress Bar */}
          <div className="score-bar-track">
            <div
              className={`score-bar-fill fill-${label.toLowerCase()}`}
              style={{ width: `${Math.min(100, Math.max(0, score))}%` }}
            ></div>
          </div>
        </div>

        {/* Detected Issues Stream */}
        <div className="issues-section">
          <div className="issues-header mono">
            <ShieldAlert size={14} className="text-secondary" />
            <span>DETECTED ISSUES & DEFECT CLASSIFICATIONS ({issues.length})</span>
          </div>

          {issues.length === 0 ? (
            <div className="nominal-status-box mono">
              <CheckCircle2 size={18} className="text-status-acceptable" />
              <div>
                <div className="nominal-title">NO DEFECTS DETECTED</div>
                <div className="nominal-sub text-secondary">
                  Image features comply with standard exposure, sharpness, and clean texture thresholds.
                </div>
              </div>
            </div>
          ) : (
            <div className="issues-list">
              {issues.map((issue, idx) => (
                <div key={idx} className="issue-row-card">
                  <div className="issue-card-top">
                    <div className="issue-name-group">
                      <span className="issue-type-name mono">
                        {issue.type.toUpperCase()}
                      </span>
                      <span className={`severity-chip severity-${issue.severity.toLowerCase()} mono`}>
                        {issue.severity.toUpperCase()} SEVERITY
                      </span>
                    </div>
                    <div className="issue-confidence mono">
                      <span className="conf-label text-muted">CONFIDENCE:</span>
                      <span className="conf-val text-highlight">
                        {(issue.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Confidence Bar */}
                  <div className="confidence-mini-track">
                    <div
                      className="confidence-mini-fill"
                      style={{ width: `${Math.min(100, issue.confidence * 100)}%` }}
                    ></div>
                  </div>

                  {issue.details && (
                    <div className="issue-diagnostic-detail mono text-secondary">
                      {issue.details}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
