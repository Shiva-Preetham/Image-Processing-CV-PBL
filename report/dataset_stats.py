"""Image properties of the whole dataset (Module 1), saved to outputs/dataset_stats.json."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cell_images"

stats = {}
for name in ("Parasitized", "Uninfected"):
    sizes = np.array([Image.open(f).size for f in (DATA / name).glob("*.png")])  # (width, height)
    stats[name] = {"images": int(len(sizes)),
                   "width": {"min": int(sizes[:, 0].min()), "median": int(np.median(sizes[:, 0])),
                             "max": int(sizes[:, 0].max())},
                   "height": {"min": int(sizes[:, 1].min()), "median": int(np.median(sizes[:, 1])),
                              "max": int(sizes[:, 1].max())}}
all_sizes = [s for s in stats.values()]
stats["all"] = {k: {"min": min(s[k]["min"] for s in all_sizes), "max": max(s[k]["max"] for s in all_sizes)}
                for k in ("width", "height")}
(ROOT / "outputs" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
