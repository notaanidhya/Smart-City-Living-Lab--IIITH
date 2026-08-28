import axios from "axios";

// Default to same origin (for Docker Nginx proxy) or localhost:8000 for local dev
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 45000,
});

export const getAssetUrl = (path) => {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
};

export const checkHealth = async () => {
  const t0 = performance.now();
  const response = await apiClient.get("/api/health");
  const latency = Math.round(performance.now() - t0);
  return { ...response.data, latency };
};

export const analyzeImage = async (file) => {
  const formData = new FormData();
  formData.append("image", file);

  const response = await apiClient.post("/api/analyze", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const getResults = async (page = 1, limit = 10, qualityLabel = null) => {
  const params = { page, limit };
  if (qualityLabel && qualityLabel !== "ALL") {
    params.quality_label = qualityLabel;
  }
  const response = await apiClient.get("/api/results", { params });
  return response.data;
};

export const getResultById = async (id) => {
  const response = await apiClient.get(`/api/results/${id}`);
  return response.data;
};

export const deleteResult = async (id) => {
  await apiClient.delete(`/api/results/${id}`);
  return true;
};

export default apiClient;
