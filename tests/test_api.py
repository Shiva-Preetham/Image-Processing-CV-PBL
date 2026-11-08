"""
Checks for the MalariaVision API.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_samples_listed():
    samples = client.get("/api/samples").json()
    assert len(samples) == 12
    assert {s["label"] for s in samples} == {"Parasitized", "Uninfected"}
    assert client.get(samples[0]["url"]).status_code == 200


@pytest.mark.parametrize("name,expected", [("parasitized_2", "Parasitized"),
                                           ("uninfected_1", "Uninfected")])
def test_analyze_sample(name, expected):
    body = client.get(f"/api/samples/{name}/analyze").json()
    assert body["verdict"]["label"] == expected
    assert body["ground_truth"] == expected
    assert len(body["steps"]) == 10
    assert all(s["image"].startswith("data:image/png;base64,") for s in body["steps"])
    assert len(body["histogram"]["before"]) == len(body["histogram"]["after"]) == 32


def test_server_matches_pytorch_on_all_samples():
    torch_classifier = pytest.importorskip("classifier")
    from pipeline import load_image, preprocess

    model = torch_classifier.load_trained()
    for s in client.get("/api/samples").json():
        served = client.get(f"/api/samples/{s['name']}/analyze").json()["verdict"]
        _, p = torch_classifier.predict(model, preprocess(load_image(ROOT / "samples" /
                                                                       f"{s['name']}.png")))
        assert abs(served["p_parasitized"] - p) < 1e-4, s["name"]


def test_upload():
    png = (ROOT / "samples" / "parasitized_3.png").read_bytes()
    r = client.post("/api/analyze", files={"file": ("cell.png", png, "image/png")},
                    data={"filter": "Gaussian", "ksize": "5", "morph": "5"})
    assert r.status_code == 200
    assert r.json()["settings"] == {"filter": "Gaussian", "ksize": 5, "morph": 5}


def test_rejects_bad_input():
    assert client.get("/api/samples/nope/analyze").status_code == 404
    assert client.get("/api/samples/../app/analyze").status_code == 404
    assert client.get("/api/samples/uninfected_1/analyze?filter=Sobel").status_code == 422
    assert client.get("/api/samples/uninfected_1/analyze?ksize=4").status_code == 422
    r = client.post("/api/analyze", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_metrics():
    m = client.get("/api/metrics").json()
    assert 0.9 < m["accuracy"] <= 1 and "confusion_matrix" in m
