"""Draws the system architecture diagram for the report (report/assets/architecture.png)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "assets"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


def box(ax, x, y, w, h, title, lines=(), fill="white", bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.2, edgecolor="black", facecolor=fill))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold" if bold else "normal")
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.55 - i * 0.24, line, ha="center", va="top", fontsize=8.5)


def arrow(ax, p, q, label=None, both=False, offset=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=12,
                                 linewidth=1.1, color="black"))
    if label:
        ax.text((p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1], label,
                ha="center", va="bottom", fontsize=8, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Lane labels
    ax.add_patch(FancyBboxPatch((0.15, 4.35), 10.7, 2.65, boxstyle="round,pad=0.02,rounding_size=0.1",
                                linewidth=0.8, edgecolor="#888", facecolor="#f4f4f4", linestyle="--"))
    ax.text(0.35, 6.85, "A. Offline: training and evaluation (laptop CPU)", fontsize=10, va="top",
            fontweight="bold", color="#333")
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 10.7, 3.95, boxstyle="round,pad=0.02,rounding_size=0.1",
                                linewidth=0.8, edgecolor="#888", facecolor="#f4f4f4", linestyle="--"))
    ax.text(0.35, 3.95, "B. Online: web application", fontsize=10, va="top", fontweight="bold",
            color="#333")

    # Offline lane
    y = 4.6
    box(ax, 0.4, y, 1.9, 1.5, "Dataset", ["NIH Malaria Cell", "Images", "27,558 cells"])
    box(ax, 2.75, y, 1.9, 1.5, "pipeline.py", ["Stages 1-5", "crop, resize, CLAHE,", "median filter"])
    box(ax, 5.1, y, 1.9, 1.5, "train.py", ["MobileNetV2,", "transfer learning", "(PyTorch)"])
    box(ax, 7.45, y, 1.5, 1.5, "Model", [".pt weights", "metrics.json", "figures"])
    box(ax, 9.3, y, 1.4, 1.5, "export_onnx.py", ["same network", "as .onnx file"], bold=True)
    for a, b in ((2.3, 2.75), (4.65, 5.1), (7.0, 7.45), (8.95, 9.3)):
        arrow(ax, (a, y + 0.75), (b, y + 0.75))

    # Online lane
    box(ax, 0.4, 1.55, 1.6, 1.6, "Browser", ["user picks or", "uploads a cell"])
    box(ax, 2.6, 1.55, 1.9, 1.6, "React web app", ["hosted on Vercel", "shows steps, verdict,", "metrics, syllabus map"])
    box(ax, 5.95, 0.45, 1.95, 3.3, "FastAPI server", ["hosted on Render", "backend/main.py", "", "/api/samples", "/api/analyze",
                                                      "/api/metrics", "/api/health"])
    box(ax, 8.4, 2.25, 2.3, 1.5, "ONNX Runtime", ["Stage 9: MobileNetV2", "Parasitized /", "Uninfected"])
    box(ax, 8.4, 0.45, 2.3, 1.5, "pipeline.py", ["Stages 1-8 (OpenCV):", "enhance, Otsu,", "morphology, contours"])
    arrow(ax, (2.0, 2.35), (2.6, 2.35), "HTTPS", both=True)
    arrow(ax, (4.5, 2.65), (5.95, 2.65), "image + settings")
    arrow(ax, (5.95, 1.95), (4.5, 1.95))
    ax.text(5.22, 1.82, "JSON: 10 step images,", ha="center", va="top", fontsize=8, style="italic")
    ax.text(5.22, 1.58, "verdict, thresholds", ha="center", va="top", fontsize=8, style="italic")
    arrow(ax, (7.9, 3.0), (8.4, 3.0))
    arrow(ax, (7.9, 1.2), (8.4, 1.2))
    # Trained model feeds the server
    arrow(ax, (10.0, 4.6), (10.0, 3.75), "model file", offset=(0.42, -0.05))

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "architecture.png", dpi=200, facecolor="white")
    print("saved", OUT / "architecture.png")


GREEN, BLUE, RED, GREY = "#2E9E5B", "#2F6FED", "#EE4B5C", "#9A9A9A"


def neural_network():
    """Simplified neural-network picture of our MobileNetV2 classifier
    (report/assets/neural_network.png). Large text so it stays readable
    at README and A4 width."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(12, 7.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.8)
    ax.axis("off")

    ax.text(6, 7.45, "MalariaVision: Neural Network Architecture", ha="center", va="center",
            fontsize=22, fontweight="bold")
    ax.text(6, 7.05, "Simplified view of our MobileNetV2 classifier (transfer learning)", ha="center",
            va="center", fontsize=13, color="#555")

    # Input cell image
    cell = plt.imread(Path(__file__).resolve().parent.parent / "samples" / "parasitized_2.png")
    ax.imshow(cell, extent=(0.15, 1.65, 3.1, 4.6), zorder=3)
    ax.text(0.9, 2.85, "Cell image", ha="center", va="top", fontsize=14, fontweight="bold")
    ax.text(0.9, 2.5, "128 × 128 × 3", ha="center", va="top", fontsize=13, color="#444")

    r = 0.3
    X_IN, X_H1, X_H2, X_OUT = 2.6, 4.55, 6.85, 8.95
    ax.annotate("", xy=(X_IN - r - 0.08, 3.85), xytext=(1.75, 3.85),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
    layers = {
        "in": (X_IN, [5.55, 4.65, 3.75, 2.05], GREEN, 2.9),
        "h1": (X_H1, [5.2, 4.3, 3.4, 2.5], BLUE, None),
        "h2": (X_H2, [4.75, 3.8, 2.85], BLUE, None),
        "out": (X_OUT, [4.35, 3.2], RED, None),
    }
    # Connections first, so circles are drawn on top
    order = ["in", "h1", "h2", "out"]
    for a, b in zip(order, order[1:]):
        xa, ya, _, _ = layers[a]
        xb, yb, _, _ = layers[b]
        for y1 in ya:
            for y2 in yb:
                ax.plot([xa + r, xb - r], [y1, y2], color=GREY, lw=0.9, zorder=1)
    for key, (x, ys, color, dots) in layers.items():
        for y in ys:
            ax.add_patch(plt.Circle((x, y), r, color=color, zorder=2))
        if dots:
            ax.text(x, dots, "⋮", ha="center", va="center", fontsize=28, color="#555")
    for label, y in (("x₁", 5.55), ("x₂", 4.65), ("x₃", 3.75), ("xₙ", 2.05)):
        ax.text(X_IN, y, label, ha="center", va="center", fontsize=14, color="white",
                fontweight="bold", zorder=3)

    # Column labels
    mid = (X_H1 + X_H2) / 2
    ax.text(X_IN, 6.35, "Input layer", ha="center", fontsize=16, fontweight="bold", color=GREEN)
    ax.text(mid, 6.35, "Hidden layers", ha="center", fontsize=16, fontweight="bold", color=BLUE)
    ax.text(mid, 6.15, "52 convolution layers\n(17 inverted residual blocks)", ha="center", va="top",
            fontsize=12, color="#333")
    ax.text(X_OUT, 6.35, "Output layer", ha="center", fontsize=16, fontweight="bold", color=RED)
    ax.text(X_OUT, 6.15, "Softmax, 2 classes", ha="center", va="top", fontsize=12, color="#333")
    ax.text(X_IN, 1.6, "49,152 pixel values\n(after Stages 1-5)", ha="center", va="top", fontsize=12)
    ax.text(X_H1, 1.6, "Early layers\nedges, colours\n(frozen)", ha="center", va="top", fontsize=12)
    ax.text(X_H2, 1.6, "Later layers\npurple spots, shapes\n(fine-tuned)", ha="center", va="top", fontsize=12)
    ax.text(X_OUT + 0.45, 4.35, "Parasitized\n98.6%", ha="left", va="center", fontsize=14, fontweight="bold")
    ax.text(X_OUT + 0.45, 3.2, "Uninfected\n1.4%", ha="left", va="center", fontsize=14)

    # Legend
    for i, (name, color) in enumerate((("Input", GREEN), ("Hidden", BLUE), ("Output", RED))):
        y = 1.95 - i * 0.38
        ax.add_patch(plt.Circle((10.0, y), 0.13, color=color))
        ax.text(10.22, y, name, va="center", fontsize=13, fontweight="bold")

    # Training facts
    ax.add_patch(FancyBboxPatch((0.05, 0.12), 11.9, 0.5, boxstyle="round,pad=0.02,rounding_size=0.1",
                                linewidth=0.8, edgecolor="#ccc", facecolor="#fafafa"))
    ax.text(6, 0.37, "ImageNet pre-trained  ·  Adam  ·  learning rate 0.001 → 0.0001  ·  "
            "cross-entropy loss  ·  19,282 training images  ·  96.6% accuracy",
            ha="center", va="center", fontsize=10)

    fig.savefig(OUT / "neural_network.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "neural_network.png")


def neuron():
    """One node of the network in detail (report/assets/neuron.png)."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(6, 4.7, "One node (neuron), zoomed in", ha="center", va="center", fontsize=20,
            fontweight="bold")
    cx, cy, R = 5.3, 2.45, 1.15
    ax.add_patch(plt.Circle((cx, cy), R, color="#333", zorder=2))
    ax.add_patch(plt.matplotlib.patches.Wedge((cx, cy), R, -90, 90, color="#666", zorder=2))
    ax.text(cx - 0.6, cy, "Σ wᵢxᵢ + b", ha="center", va="center", fontsize=12, color="white", zorder=3)
    ax.text(cx + 0.56, cy, "f(z)", ha="center", va="center", fontsize=16, color="white", zorder=3)
    ax.text(cx - 0.9, cy + R + 0.28, "Weighted sum", ha="center", fontsize=13, color="#333")
    ax.text(cx + 0.95, cy + R + 0.28, "Activation", ha="center", fontsize=13, color="#333")

    ax.text(0.9, 4.0, "Inputs", ha="center", fontsize=14, fontweight="bold", color=GREEN)
    for label, y in (("x₁", 3.4), ("x₂", 2.45), ("xₙ", 1.3)):
        ax.text(0.9, y, label, ha="center", va="center", fontsize=17)
        ax.annotate("", xy=(cx - R * 0.98, cy + (y - cy) * 0.3), xytext=(1.25, y),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))
    ax.text(0.9, 1.9, "⋮", ha="center", va="center", fontsize=22, color="#555")
    ax.text(1.4, 0.55, "Weights w₁ … wₙ on the arrows (learned in training)", ha="left", fontsize=13,
            color="#333")

    ax.annotate("", xy=(7.75, cy), xytext=(cx + R, cy), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))
    ax.text(7.1, cy + 0.2, "output", ha="center", fontsize=14)

    ax.add_patch(FancyBboxPatch((8.0, 0.75), 3.85, 3.35, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=0.8, edgecolor="#ccc", facecolor="#f7f7f7"))
    lines = [("z = Σ wᵢxᵢ + b", True), ("output = f(z)", True), ("", False),
             ("Hidden layers: f = ReLU6", True), ("ReLU6(z) = min(max(0, z), 6)", False), ("", False),
             ("Output layer: f = Softmax", True), ("turns the 2 scores into", False),
             ("probabilities that add up to 1", False)]
    y = 3.85
    for text, bold in lines:
        if text:
            ax.text(8.2, y, text, ha="left", va="center", fontsize=12.5 if bold else 11.5,
                    fontweight="bold" if bold else "normal", color="#222" if bold else "#444")
        y -= 0.34

    fig.savefig(OUT / "neuron.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "neuron.png")


if __name__ == "__main__":
    main()
    neural_network()
    neuron()
