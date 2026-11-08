// Talks to the MalariaVision FastAPI server (backend/main.py).
// Locally it runs on port 8000; in production VITE_API_URL points to Render.
export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

async function getJSON(path, options) {
  const response = await fetch(API_URL + path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

const query = (s) => new URLSearchParams({ filter: s.filter, ksize: s.ksize, morph: s.morph });

export const api = {
  health: () => getJSON("/api/health"),
  samples: () => getJSON("/api/samples"),
  metrics: () => getJSON("/api/metrics"),
  analyzeSample: (name, settings) => getJSON(`/api/samples/${name}/analyze?${query(settings)}`),
  analyzeUpload: (file, settings) => {
    const form = new FormData();
    form.append("file", file);
    Object.entries({ filter: settings.filter, ksize: settings.ksize, morph: settings.morph })
      .forEach(([k, v]) => form.append(k, v));
    return getJSON("/api/analyze", { method: "POST", body: form });
  },
  url: (path) => API_URL + path,
};

export const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)}%`;
