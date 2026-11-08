"""
MalariaVision API - serves the image-processing pipeline and the CNN to the
React web app.

    uvicorn backend.main:app --reload --port 8000     (run from the project folder)

Interactive API docs: http://localhost:8000/docs

All image processing happens here, in Python with OpenCV (pipeline.py).
The React app only displays the results.
"""
import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.model import OnnxClassifier
from pipeline import (FILTERS, MIN_STAIN_GAP, STEPS, decode_image, load_image, run_pipeline,
                      to_hsv)

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUTPUTS = ROOT / "outputs"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
KERNEL_SIZES = (3, 5, 7)
MORPH_SIZES = (3, 5)

app = FastAPI(title="MalariaVision API", version="0.1",
              description="Malaria parasite detection from blood-smear cell images.")
app.add_middleware(CORSMiddleware,
                   allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/static/samples", StaticFiles(directory=SAMPLES), name="samples")
app.mount("/static/outputs", StaticFiles(directory=OUTPUTS), name="outputs")

model = OnnxClassifier(ROOT / "models" / "malariavision.onnx")


def sample_index():
    """name -> (path, true label) for the demo cells in samples/."""
    index = {}
    for path in sorted(SAMPLES.glob("*.png")):
        label, number = path.stem.split("_")
        index[path.stem] = (path, label.title(), int(number))
    return index


def png_data_url(img, max_side=256):
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def brightness_histogram(rgb, bins=32):
    """Histogram of the V (brightness) channel over the cell pixels only."""
    v = to_hsv(rgb)[:, :, 2]
    counts, _ = np.histogram(v[v > 10], bins=bins, range=(0, 256))
    return counts.tolist()


def check_settings(filter_method, ksize, morph):
    if filter_method not in FILTERS:
        raise HTTPException(422, f"filter must be one of {list(FILTERS)}")
    if ksize not in KERNEL_SIZES:
        raise HTTPException(422, f"ksize must be one of {list(KERNEL_SIZES)}")
    if morph not in MORPH_SIZES:
        raise HTTPException(422, f"morph must be one of {list(MORPH_SIZES)}")


def analyze(rgb, filter_method, ksize, morph, ground_truth=None):
    check_settings(filter_method, ksize, morph)
    r = run_pipeline(rgb, filter_method, ksize, morph)
    label, p = model.predict(r.cnn_input)
    return {
        "verdict": {"label": label, "p_parasitized": p,
                    "confidence": p if label == "Parasitized" else 1 - p},
        "ground_truth": ground_truth,
        "segmentation": {"spots": len(r.spot_contours), "spot_area": r.spot_area,
                         "otsu_cell": float(r.otsu_cell), "otsu_stain": float(r.otsu_stain),
                         "stain_gap": r.stain_gap, "min_stain_gap": MIN_STAIN_GAP},
        "steps": [{"key": key, "title": title, "module": module, "note": note,
                   "image": png_data_url(r.steps[key])}
                  for key, title, module, note in STEPS],
        "histogram": {"before": brightness_histogram(r.steps["2_resized"]),
                      "after": brightness_histogram(r.steps["4_enhanced"])},
        "settings": {"filter": filter_method, "ksize": ksize, "morph": morph},
    }


@app.get("/")
def root():
    return {"name": "MalariaVision API", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {"status": "ok", "model": "MobileNetV2 (ONNX Runtime)"}


@app.get("/api/samples")
def samples():
    return [{"name": name, "label": label, "number": number,
             "url": f"/static/samples/{path.name}"}
            for name, (path, label, number) in sample_index().items()]


@app.get("/api/samples/{name}/analyze")
def analyze_sample(name: str, filter: str = "Median", ksize: int = 3, morph: int = 3):
    index = sample_index()
    if name not in index:
        raise HTTPException(404, "Unknown sample")
    path, label, _ = index[name]
    return analyze(load_image(path), filter, ksize, morph, ground_truth=label)


@app.post("/api/analyze")
def analyze_upload(file: UploadFile = File(...), filter: str = Form("Median"),
                   ksize: int = Form(3), morph: int = Form(3)):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 5 MB")
    try:
        rgb = decode_image(data)
    except ValueError as err:
        raise HTTPException(400, str(err))
    return analyze(rgb, filter, ksize, morph)


@app.get("/api/metrics")
def metrics():
    return json.loads((OUTPUTS / "metrics.json").read_text())
