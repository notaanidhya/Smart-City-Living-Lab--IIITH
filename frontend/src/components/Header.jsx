import React, { useEffect, useState } from "react";
import { Activity, Layers, History, ShieldCheck, AlertCircle, RefreshCw } from "lucide-react";
import { checkHealth } from "../api/client";

export default function Header({ activeTab, onTabChange }) {
  const [health, setHealth] = useState({ status: "checking", models_loaded: false, latency: null });

  const fetchHealth = async () => {
    try {
      const data = await checkHealth();
      setHealth({
        status: data.status === "ok" ? "online" : "degraded",
        models_loaded: data.models_loaded,
        latency: data.latency,
      });
    } catch (err) {
      setHealth({ status: "offline", models_loaded: false, latency: null });
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 25000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="site-header">
      <div className="header-inner">
        {/* Brand identity */}
        <div className="brand-section">
          <div className="brand-logo">
            <Layers size={20} className="text-highlight" />
          </div>
          <div>
            <div className="brand-title">
              VisionQuality<span className="brand-sub">.ai</span>
            </div>
            <div className="brand-caption mono">
              Smart City Living Lab • Hybrid CV & DL Inspection Engine
            </div>
          </div>
        </div>

        {/* View Switcher Tabs (Square / Tabbed - No generic pills) */}
        <nav className="header-nav">
          <button
            className={`nav-tab ${activeTab === "workspace" ? "active" : ""}`}
            onClick={() => onTabChange("workspace")}
          >
            <Activity size={15} />
            <span>Inspection Workbench</span>
          </button>
          <button
            className={`nav-tab ${activeTab === "history" ? "active" : ""}`}
            onClick={() => onTabChange("history")}
          >
            <History size={15} />
            <span>Audit History</span>
          </button>
        </nav>

        {/* System Telemetry & Health */}
        <div className="header-status">
          {health.status === "online" ? (
            <div className="telemetry-badge online mono">
              <span className="status-dot online"></span>
              <span>API ONLINE</span>
              {health.latency && <span className="latency">({health.latency}ms)</span>}
              {health.models_loaded && (
                <span className="models-tag">
                  <ShieldCheck size={12} /> MODELS READY
                </span>
              )}
            </div>
          ) : health.status === "checking" ? (
            <div className="telemetry-badge checking mono">
              <RefreshCw size={12} className="spin" />
              <span>CONNECTING...</span>
            </div>
          ) : (
            <div className="telemetry-badge offline mono">
              <span className="status-dot offline"></span>
              <span>BACKEND OFFLINE</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
