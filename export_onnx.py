"""
MalariaVision - export the trained PyTorch model to ONNX for deployment.

    python export_onnx.py

ONNX is a standard file format for trained networks. The web server loads
models/malariavision.onnx with ONNX Runtime, a small inference-only library,
so the free cloud server does not need the whole of PyTorch in memory.
The network and its weights are exactly the same; this script checks that
both versions give the same answers on the demo samples.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from classifier import MODEL_PATH, load_trained, predict, to_tensor
from pipeline import IMG_SIZE, load_image, preprocess

ROOT = Path(__file__).parent
ONNX_PATH = ROOT / "models" / "malariavision.onnx"


def export():
    model = load_trained()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(model, dummy, str(ONNX_PATH), input_names=["image"],
                      output_names=["logits"], opset_version=17,
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    return model


def verify(model):
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    worst = 0.0
    for path in sorted((ROOT / "samples").glob("*.png")):
        img = preprocess(load_image(path))
        _, p_torch = predict(model, img)
        logits = session.run(None, {"image": to_tensor(img).unsqueeze(0).numpy()})[0][0]
        p_onnx = float(np.exp(logits[1]) / np.exp(logits).sum())
        worst = max(worst, abs(p_torch - p_onnx))
        print(f"{path.stem:15s} PyTorch {p_torch:.5f}   ONNX {p_onnx:.5f}")
    print(f"Largest difference in probability: {worst:.2e}")
    assert worst < 1e-4, "ONNX model does not match the PyTorch model"


if __name__ == "__main__":
    assert MODEL_PATH.exists(), "Train the model first: python train.py"
    verify(export())
    print(f"Saved {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")
