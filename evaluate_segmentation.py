"""
MalariaVision - evaluate the classical stages (6-8) on the test set.

    python evaluate_segmentation.py

Stages 6-8 (Otsu thresholding, erosion + dilation, contours) outline the
dark-stained spots inside a cell. This script measures, on the same test
images used for the CNN scores (outputs/test_split.csv), how often a spot
is outlined in infected cells and how often one is wrongly outlined in
healthy cells. It also measures how often the cell outline itself is found.
Results go to outputs/segmentation_eval.json.
"""
import csv
import json
import time
from pathlib import Path

import numpy as np

from pipeline import MIN_STAIN_GAP, load_image, run_pipeline

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "cell_images"
OUT = ROOT / "outputs"


def main():
    rows = list(csv.DictReader(open(OUT / "test_split.csv")))
    t0 = time.time()
    stats = {"Parasitized": [], "Uninfected": []}
    cell_found = 0
    for row in rows:
        r = run_pipeline(load_image(DATA / row["label"] / row["file"]))
        cell_found += r.cell_contour is not None
        stats[row["label"]].append((len(r.spot_contours), r.spot_area, r.stain_gap))

    def summary(values):
        spots = np.array([v[0] for v in values])
        gaps = np.array([v[2] for v in values])
        return {"cells": len(values),
                "with_spot": int((spots > 0).sum()),
                "with_spot_rate": float((spots > 0).mean()),
                "mean_spots_when_found": float(spots[spots > 0].mean()) if (spots > 0).any() else 0.0,
                "median_stain_contrast": float(np.median(gaps))}

    p, u = summary(stats["Parasitized"]), summary(stats["Uninfected"])
    result = {
        "test_images": len(rows),
        "cell_outline_found_rate": cell_found / len(rows),
        "parasitized": p,
        "uninfected": u,
        "spot_detection_rate_infected": p["with_spot_rate"],
        "false_spot_rate_healthy": u["with_spot_rate"],
        "min_stain_contrast": MIN_STAIN_GAP,
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "segmentation_eval.json").write_text(json.dumps(result, indent=2))
    print(f"Test images: {len(rows)}  (cell outline found in {result['cell_outline_found_rate']:.1%})")
    print(f"Infected cells with an outlined spot: {p['with_spot']}/{p['cells']} = {p['with_spot_rate']:.1%}")
    print(f"Healthy cells with a false spot:      {u['with_spot']}/{u['cells']} = {u['with_spot_rate']:.1%}")


if __name__ == "__main__":
    main()
