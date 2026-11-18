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
    """Simplified neural-network picture of our MobileNetV2 classifier,
    with one node zoomed in (weights, bias, activation)."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(8, 8.6, "MalariaVision: Neural Network Architecture", ha="center", va="center",
            fontsize=26, fontweight="bold")
    ax.text(8, 8.1, "Simplified view of our MobileNetV2 classifier (transfer learning)", ha="center",
            va="center", fontsize=14, color="#555")

    # Input cell image
    cell = plt.imread(Path(__file__).resolve().parent.parent / "samples" / "parasitized_2.png")
    ax.imshow(cell, extent=(0.2, 1.7, 4.0, 5.5), zorder=3)
    ax.text(0.95, 3.75, "Cell image", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(0.95, 3.4, "128 × 128 × 3", ha="center", va="top", fontsize=11, color="#444")
    ax.annotate("", xy=(2.5, 4.75), xytext=(1.8, 4.75),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

    r = 0.27
    X_IN, X_H1, X_H2, X_OUT = 2.9, 5.0, 7.1, 9.2
    layers = {
        "in": (X_IN, [6.8, 5.8, 4.8, 2.8], GREEN, 3.8),
        "h1": (X_H1, [6.4, 5.4, 4.4, 3.4], BLUE, 2.6),
        "h2": (X_H2, [5.9, 4.9, 3.9], BLUE, None),
        "out": (X_OUT, [5.55, 4.15], RED, None),
    }
    # Connections first, so circles are drawn on top
    order = ["in", "h1", "h2", "out"]
    for a, b in zip(order, order[1:]):
        xa, ya, _, _ = layers[a]
        xb, yb, _, _ = layers[b]
        for y1 in ya:
            for y2 in yb:
                ax.plot([xa + r, xb - r], [y1, y2], color=GREY, lw=0.8, zorder=1)
    for key, (x, ys, color, dots) in layers.items():
        for y in ys:
            ax.add_patch(plt.Circle((x, y), r, color=color, zorder=2))
        if dots:
            ax.text(x, dots, "⋮", ha="center", va="center", fontsize=26, color="#555")
    for i, y in enumerate([6.8, 5.8, 4.8]):
        ax.text(X_IN, y, f"x{'₁₂₃'[i]}", ha="center", va="center", fontsize=11, color="white",
                fontweight="bold", zorder=3)
    ax.text(X_IN, 2.8, "xₙ", ha="center", va="center", fontsize=11, color="white", fontweight="bold",
            zorder=3)

    # Column labels
    ax.text(X_IN, 7.45, "Input layer", ha="center", fontsize=13, fontweight="bold", color=GREEN)
    ax.text(X_IN, 2.15, "49,152 pixel values\n(after Stages 1-5)", ha="center", va="top", fontsize=10.5)
    mid = (X_H1 + X_H2) / 2
    ax.text(mid, 7.45, "Hidden layers", ha="center", fontsize=13, fontweight="bold", color=BLUE)
    ax.text(mid, 7.2, "52 convolution layers\n(17 inverted residual blocks)", ha="center", va="top",
            fontsize=10, color="#333")
    ax.text(X_H1, 2.15, "Early layers\nedges, colours\n(frozen)", ha="center", va="top", fontsize=10.5)
    ax.text(X_H2, 2.15, "Later layers\npurple spots, shapes\n(fine-tuned)", ha="center", va="top",
            fontsize=10.5)
    ax.text(X_OUT, 7.45, "Output layer", ha="center", fontsize=13, fontweight="bold", color=RED)
    ax.text(X_OUT, 7.2, "Softmax\n2 classes", ha="center", va="top", fontsize=10, color="#333")
    ax.text(X_OUT + 0.38, 5.55, "Parasitized\n98.6%", ha="left", va="center", fontsize=11, fontweight="bold")
    ax.text(X_OUT + 0.38, 4.15, "Uninfected\n1.4%", ha="left", va="center", fontsize=11)

    # Zoomed-in node
    ax.add_patch(FancyBboxPatch((11.12, 2.35), 4.76, 4.75, boxstyle="round,pad=0.02,rounding_size=0.15",
                                linewidth=1, edgecolor="#bbb", facecolor="#f7f7f7", zorder=0))
    ax.text(13.57, 6.8, "One node (neuron), zoomed in", ha="center", fontsize=12.5, fontweight="bold")
    cx, cy, R = 13.7, 4.6, 0.95
    ax.add_patch(plt.Circle((cx, cy), R, color="#333", zorder=2))
    ax.add_patch(plt.matplotlib.patches.Wedge((cx, cy), R, -90, 90, color="#666", zorder=2))
    ax.text(cx - 0.47, cy, "Σ wᵢxᵢ + b", ha="center", va="center", fontsize=9.5, color="white", zorder=3)
    ax.text(cx + 0.47, cy, "f(z)", ha="center", va="center", fontsize=11, color="white", zorder=3)
    ax.text(cx - 0.62, cy + R + 0.22, "Weighted sum", ha="center", fontsize=9.5, color="#333")
    ax.text(cx + 0.72, cy + R + 0.22, "Activation", ha="center", fontsize=9.5, color="#333")
    for label, y in (("x₁", 5.7), ("x₂", 4.6), ("xₙ", 3.5)):
        ax.text(11.6, y, label, ha="center", va="center", fontsize=12)
        ax.annotate("", xy=(cx - R * 0.98, cy + (y - cy) * 0.28), xytext=(11.85, y),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
    ax.text(11.6, 6.2, "Inputs", ha="center", fontsize=10, color="#333")
    ax.text(12.35, 6.2, "Weights wᵢ (learned)", ha="left", fontsize=10, color="#333")
    ax.annotate("", xy=(15.7, cy), xytext=(cx + R, cy), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
    ax.text(15.2, cy + 0.15, "output", ha="center", fontsize=10)
    ax.text(13.57, 3.3, "z = Σ wᵢxᵢ + b     output = f(z)", ha="center", va="top", fontsize=10,
            color="#222")
    ax.text(13.57, 2.95, "Hidden layers: f = ReLU6\nOutput layer: f = Softmax", ha="center", va="top",
            fontsize=9.5, color="#333")

    # Legend
    for i, (name, color) in enumerate((("Input", GREEN), ("Hidden", BLUE), ("Output", RED))):
        x = 11.75 + i * 1.45
        ax.add_patch(plt.Circle((x, 1.75), 0.16, color=color))
        ax.text(x + 0.25, 1.75, name, va="center", fontsize=11.5, fontweight="bold")

    # Training facts
    ax.add_patch(FancyBboxPatch((0.4, 0.3), 15.2, 0.8, boxstyle="round,pad=0.02,rounding_size=0.1",
                                linewidth=0.8, edgecolor="#ccc", facecolor="#fafafa"))
    ax.text(8, 0.7, "Pre-trained on ImageNet  ·  Adam  ·  learning rate 0.001 → 0.0001 (fine-tuning)  ·  "
            "cross-entropy loss  ·  19,282 training images  ·  96.6% test accuracy",
            ha="center", va="center", fontsize=10.5)

    fig.savefig(OUT / "neural_network.png", dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "neural_network.png")


if __name__ == "__main__":
    main()
    neural_network()
