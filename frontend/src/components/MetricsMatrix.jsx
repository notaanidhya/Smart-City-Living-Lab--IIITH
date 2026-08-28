import React, { useState } from "react";
import { Terminal, ChevronDown, ChevronUp, BarChart2 } from "lucide-react";

export default function MetricsMatrix({ statistics }) {
  const [isOpen, setIsOpen] = useState(true);

  if (!statistics) return null;

  const all = statistics.all_features || {};
  const reconErr = statistics.reconstruction_error ?? null;

  const metricGroups = [
    {
      category: "Sharpness & Focus",
      items: [
        { key: "laplacian_variance", label: "Laplacian Variance", val: all.laplacian_variance ?? statistics.laplacian_variance, unit: "var" },
        { key: "tenengrad_mean", label: "Tenengrad Mean", val: all.tenengrad_mean, unit: "grad" },
        { key: "fft_high_freq_ratio", label: "FFT High-Freq Ratio", val: all.fft_high_freq_ratio, unit: "ratio" },
        { key: "edge_density", label: "Canny Edge Density", val: all.edge_density, unit: "frac" },
      ],
    },
    {
      category: "Exposure & Dynamic Range",
      items: [
        { key: "mean_luminance", label: "Mean Luminance", val: all.mean_luminance ?? statistics.mean_luminance, unit: "/255" },
        { key: "dark_pixel_ratio", label: "Crushed Black Ratio", val: all.dark_pixel_ratio, unit: "<10" },
        { key: "bright_pixel_ratio", label: "Clipped Highlight Ratio", val: all.bright_pixel_ratio, unit: ">245" },
        { key: "histogram_skewness", label: "Histogram Skewness", val: all.histogram_skewness, unit: "skew" },
        { key: "rms_contrast", label: "RMS Contrast", val: all.rms_contrast ?? statistics.rms_contrast, unit: "std/μ" },
        { key: "michelson_contrast", label: "Michelson Contrast", val: all.michelson_contrast, unit: "range" },
      ],
    },
    {
      category: "Noise & Structural Texture",
      items: [
        { key: "noise_sigma_immerkaar", label: "Immerkær Noise σ", val: all.noise_sigma_immerkaar ?? statistics.noise_sigma_immerkaar, unit: "σ" },
        { key: "flat_region_variance", label: "Flat Region Variance", val: all.flat_region_variance, unit: "var" },
        { key: "snr_proxy", label: "Signal-to-Noise Proxy", val: all.snr_proxy, unit: "SNR" },
        { key: "glcm_contrast", label: "GLCM Contrast", val: all.glcm_contrast ?? statistics.glcm_contrast, unit: "tex" },
        { key: "glcm_homogeneity", label: "GLCM Homogeneity", val: all.glcm_homogeneity, unit: "homo" },
        { key: "glcm_energy", label: "GLCM Energy (ASM)", val: all.glcm_energy, unit: "ASM" },
      ],
    },
    {
      category: "Color, Compression & Anomaly",
      items: [
        { key: "mean_saturation", label: "HSV Mean Saturation", val: all.mean_saturation ?? statistics.mean_saturation, unit: "0-1" },
        { key: "channel_imbalance", label: "RGB Channel Imbalance", val: all.channel_imbalance, unit: "dev" },
        { key: "colorfulness", label: "Hasler Colorfulness", val: all.colorfulness, unit: "idx" },
        { key: "dct_blockiness", label: "8×8 DCT Blockiness", val: all.dct_blockiness ?? statistics.dct_blockiness, unit: "jump" },
        { key: "hf_energy_loss", label: "HF Energy Loss", val: all.hf_energy_loss, unit: "ratio" },
        { key: "reconstruction_error", label: "Autoencoder Peak Error", val: reconErr, unit: "MSE" },
      ],
    },
  ];

  const formatValue = (val) => {
    if (val === undefined || val === null) return "—";
    if (typeof val === "number") {
      if (Math.abs(val) >= 1000) return val.toLocaleString(undefined, { maximumFractionDigits: 1 });
      if (Math.abs(val) < 0.001 && val !== 0) return val.toExponential(2);
      return val.toFixed(3);
    }
    return String(val);
  };

  return (
    <div className="workbench-panel metrics-panel">
      <button className="metrics-toggle-header" onClick={() => setIsOpen(!isOpen)}>
        <div className="panel-title">
          <BarChart2 size={15} />
          <span>Extracted 22-Metric Computer Vision Matrix</span>
        </div>
        <div className="toggle-indicator mono">
          <span>{isOpen ? "COLLAPSE TELEMETRY" : "EXPAND TELEMETRY"}</span>
          {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </button>

      {isOpen && (
        <div className="metrics-body">
          <div className="metrics-groups-grid">
            {metricGroups.map((grp, gIdx) => (
              <div key={gIdx} className="metric-group-card">
                <div className="metric-group-title mono">{grp.category}</div>
                <div className="metric-rows-table mono">
                  {grp.items.map((m, mIdx) => (
                    <div key={mIdx} className="metric-row-item">
                      <span className="metric-label text-secondary">{m.label}</span>
                      <div className="metric-value-box">
                        <span className="metric-val text-highlight">{formatValue(m.val)}</span>
                        {m.unit && <span className="metric-unit text-muted">{m.unit}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
