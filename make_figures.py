"""
MalariaVision - figures for the report and the viva.

    python make_figures.py

Creates in outputs/:
  dataset_overview.png        random cells of each class + image sizes (Module 1)
  pipeline_parasitized.png    every pipeline stage for an infected cell
  pipeline_uninfected.png     every pipeline stage for a healthy cell
  histogram_equalisation.png  brightness histogram before / after (Module 3)
"""
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classifier import CLASSES, MODEL_PATH, load_trained, predict
from pipeline import STEPS, enhance_contrast, load_image, run_pipeline, to_hsv

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "cell_images"
SAMPLES = ROOT / "samples"
OUT = ROOT / "outputs"


def show(ax, img, title, sub=None):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if sub:
        ax.text(0.5, -0.08, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="#555")
    ax.axis("off")


def dataset_overview(path, per_class=6, seed=3):
    rng = random.Random(seed)
    fig, axes = plt.subplots(2, per_class + 1, figsize=(2 * (per_class + 1), 4.6),
                             gridspec_kw={"width_ratios": [1] * per_class + [1.6]})
    sizes = {}
    for row, name in enumerate(CLASSES):
        files = sorted((DATA / name).glob("*.png"))
        for ax, f in zip(axes[row, :per_class], rng.sample(files, per_class)):
            show(ax, load_image(f), "")
        axes[row, 0].set_title(f"{name}  ({len(files):,} images)", loc="left",
                               fontsize=11, fontweight="bold")
        sizes[name] = [load_image(f).shape[:2] for f in rng.sample(files, 400)]
    # Image properties (Module 1): sizes vary, so Stage 2 resizes them
    ax = axes[0, -1]
    for name, color in zip(CLASSES, ("#2a9d8f", "#c0392b")):
        hw = np.array(sizes[name])
        ax.scatter(hw[:, 1], hw[:, 0], s=6, alpha=0.5, color=color, label=name)
    ax.set_title("Image sizes vary", fontsize=10, fontweight="bold")
    ax.set_xlabel("width (px)", fontsize=8)
    ax.set_ylabel("height (px)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    axes[1, -1].axis("off")
    axes[1, -1].text(0, 0.5, "Every image is RGB,\ncell on black background.\n\n"
                     "Sizes differ, so Stage 2\nresizes all to 128 x 128.",
                     fontsize=9, va="center")
    fig.suptitle("Malaria Cell Images Dataset (NIH) - 27,558 cells, perfectly balanced",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pipeline_figure(image_path, path, model=None):
    rgb = load_image(image_path)
    r = run_pipeline(rgb)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.6))
    for ax, (key, title, module, note) in zip(axes.flat, STEPS):
        show(ax, r.steps[key], f"{title}  [{module.replace('Module ', 'M')}]", note)
    headline = f"{len(r.spot_contours)} stained spot(s) found by Stages 6-8"
    if model is not None:
        label, p = predict(model, r.cnn_input)
        conf = p if label == "Parasitized" else 1 - p
        headline = f"Stage 9 CNN: {label} ({conf:.1%})   |   " + headline
    fig.suptitle(f"MalariaVision pipeline - {image_path.name}\n{headline}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def histogram_figure(image_path, path):
    rgb = load_image(image_path)
    v_before = to_hsv(rgb)[:, :, 2]
    v_after = to_hsv(enhance_contrast(rgb))[:, :, 2]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4),
                             gridspec_kw={"width_ratios": [1, 1.6, 1, 1.6]})
    for (ax_img, ax_hist), v, img, title in (
            (axes[:2], v_before, rgb, "Before"),
            (axes[2:], v_after, enhance_contrast(rgb), "After equalisation")):
        show(ax_img, img, title)
        cell = v[v > 10]  # ignore the black background
        ax_hist.hist(cell, bins=64, range=(0, 255), color="#5b4b8a")
        ax_hist.set_title(f"{title}: brightness histogram (cell pixels)", fontsize=10)
        ax_hist.set_xlim(0, 255)
        ax_hist.set_xlabel("brightness (V channel)")
    fig.suptitle("Stage 4 - Histogram equalisation spreads brightness over the full range",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pick_example(name, index=1):
    sample = SAMPLES / f"{name.lower()}_{index}.png"
    if sample.exists():
        return sample
    return sorted((DATA / name).glob("*.png"))[0]


def main():
    OUT.mkdir(exist_ok=True)
    model = load_trained() if MODEL_PATH.exists() else None
    if DATA.exists():
        dataset_overview(OUT / "dataset_overview.png")
    for name in CLASSES:
        pipeline_figure(pick_example(name), OUT / f"pipeline_{name.lower()}.png", model)
    histogram_figure(pick_example("Parasitized"), OUT / "histogram_equalisation.png")
    print("Figures saved to", OUT)


if __name__ == "__main__":
    main()
