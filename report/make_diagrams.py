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


if __name__ == "__main__":
    main()
