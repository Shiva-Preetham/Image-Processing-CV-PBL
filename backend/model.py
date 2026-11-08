"""
Stage 9 on the web server: the same trained MobileNetV2, run with ONNX Runtime.

models/malariavision.onnx is produced by export_onnx.py from the PyTorch
model. ONNX Runtime only runs a trained network (no training), so it needs
far less memory than PyTorch, which matters on a free cloud server.
"""
import numpy as np
import onnxruntime as ort

CLASSES = ["Uninfected", "Parasitized"]

# Same ImageNet normalisation as classifier.py (repeated here so the
# server does not need to import PyTorch).
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OnnxClassifier:
    def __init__(self, path):
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def predict(self, cnn_input_rgb):
        """Returns (class name, probability of Parasitized)."""
        x = ((cnn_input_rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        logits = self.session.run(None, {"image": x})[0][0]
        e = np.exp(logits - logits.max())
        p_parasitized = float(e[1] / e.sum())
        return CLASSES[int(p_parasitized >= 0.5)], p_parasitized
