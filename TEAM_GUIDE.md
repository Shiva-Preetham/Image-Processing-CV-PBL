# MalariaVision: Team Guide

A quick tour of the code for the whole group. Read this first, then open the files in the order below. You do not need to understand every line; you need to be able to explain **what each stage does, why, and which syllabus module it comes from**.

- **Live demo:** https://malariavision.vercel.app (open it a minute early; the free server takes up to a minute to wake up)
- **Report:** [`report/MalariaVision_Project_Report.pdf`](report/MalariaVision_Project_Report.pdf)
- **Viva talk track:** [`FACULTY_GUIDE.md`](FACULTY_GUIDE.md)
- **Code on GitHub:** https://github.com/Shiva-Preetham/Image-Processing-CV-PBL

---

## 1. The project in three sentences

We take a microscope image of **one red blood cell** and decide whether it is **infected with malaria**. Classical image processing (Modules 1-5) cleans the cell and outlines any dark-purple stained spot, and a pre-trained CNN (MobileNetV2, transfer learning) gives the final answer. On 4,144 test images the model never saw, it is right **96.6%** of the time and catches **96.3%** of infected cells.

---

## 2. Run it on your laptop (10 minutes)

You need **Python 3.11**. Node.js is only needed for the React website.

```bash
python -m venv .venv
.venv\Scripts\activate              # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Easiest demo (Python only, opens in the browser):

```bash
streamlit run app.py
```

Full website, as deployed (two terminals):

```bash
uvicorn backend.main:app --port 8000
```

```bash
cd frontend
npm install
npm run dev                          # then open http://localhost:5173
```

The trained model and 12 demo cells are already included, so **you do not need the dataset to run the demo**. The dataset (337 MB) is only needed to re-train; the download commands are in the README.

---

## 3. What is in the folder

| Path | What it is | Read it? |
|------|-------------|----------|
| [`pipeline.py`](pipeline.py) | **Stages 1-8**: all the classical image processing | **Yes, first** |
| [`classifier.py`](classifier.py) | **Stage 9**: MobileNetV2, data augmentation, prediction | **Yes** |
| [`train.py`](train.py) | Training + **Stage 10** evaluation (metrics, confusion matrix) | **Yes** |
| [`evaluate_segmentation.py`](evaluate_segmentation.py) | Measures Stages 6-8 on the test set | Skim |
| [`export_onnx.py`](export_onnx.py) | Converts the model for the web server and checks it | Skim |
| [`make_figures.py`](make_figures.py) | Makes the figures used in the report | Skim |
| [`app.py`](app.py) | Streamlit demo (offline backup) | Optional |
| [`backend/`](backend/main.py) | FastAPI web server: runs `pipeline.py` + the model | Skim |
| [`frontend/src/`](frontend/src/App.jsx) | React website (only displays results) | Optional |
| [`tests/`](tests/test_api.py) | Automated tests of the web server | Optional |
| `models/` | Trained model: `.pt` (PyTorch) and `.onnx` (for the server) | No |
| `samples/` | The 12 demo cells (all from the test set) | No |
| `outputs/` | Results: `metrics.json`, figures, training log, test split | Look at the figures |
| [`report/`](report/) | The report (PDF + Word) and the scripts that build it | Read the PDF |

---

## 4. Reading order

### Step 1: `pipeline.py` (most important for the viva)

Every function does **one syllabus operation** and is labelled with its module. Read them top to bottom:

| Stage | Function | Module | What it does in one line |
|:-----:|----------|:------:|--------------------------|
| 1 | `load_image` | M1 | Reads the file; OpenCV gives BGR, so convert to RGB |
| 2 | `crop_and_resize` | M2 | Cuts off the black border and resizes to 128 × 128 |
| 3 | `to_gray`, `to_hsv` | M2 | Grayscale for brightness; HSV saturation makes the purple stain stand out |
| 4 | `enhance_contrast` | M3 | **CLAHE** histogram equalisation on the brightness (V) channel |
| 5 | `denoise` | M3 | **Median** filter 3 × 3 (Gaussian and Average also available) |
| 6 | `otsu_threshold`, `otsu_inside` | M4 | **Otsu** picks the threshold automatically: cell vs. background, then stain vs. normal |
| 7 | `clean_mask`, `fill_holes` | M2 | **Erosion + dilation** (opening) and dilation + erosion (closing) |
| 8 | `find_contours` | M5 | **Contours** of the cell (green) and of the stained spots (red) |

Then read **`run_pipeline()`** at the bottom: it calls the stages in order and keeps every intermediate image in `result.steps` (that is what the website and the report figures show). Two details worth knowing:

- **Convex hull** (Stage 8): a dark parasite on the cell edge used to be cut out of the cell outline; wrapping the outline in its convex hull keeps it inside. This raised spot detection from 64% to 89% during development.
- **Stain-contrast rule** (Stage 6): Otsu always splits pixels into two groups, even in a healthy cell, so a spot only counts if it is clearly more saturated than the rest of the cell (gap of at least 20).

`preprocess()` = Stages 2 to 5. Its output is exactly what the CNN sees.

### Step 2: `classifier.py`

- `build_model()`: MobileNetV2 pre-trained on ImageNet, with the last layer replaced by a 2-class layer (Uninfected, Parasitized).
- `augment()`: random flips and 90° rotations (Module 2). A cell looks the same upside-down, so this gives more training examples for free.
- `CellDataset`: loads an image, runs `preprocess()` (Stages 1-5), returns a tensor.
- `predict()`: returns the label and the probability of infection.

### Step 3: `train.py`

`main()` does everything in order:

1. `list_dataset()` + `split_dataset()`: 70% train / 15% validation / 15% test. The 12 demo cells are forced into the test set, so the demo never shows a cell the model trained on.
2. **Phase A**: only the new last layer learns (pre-trained layers frozen), 2 epochs.
3. **Phase B**: the last 8 blocks are unfrozen and fine-tuned with a 10× smaller learning rate, 5 epochs.
4. The epoch with the best validation accuracy is kept, then tested **once** on the test set (Stage 10): accuracy, precision, recall, F1, confusion matrix. Everything is saved in `outputs/`.

### Step 4: the web app (optional)

`backend/main.py` receives an image from the website, runs `run_pipeline()` and the model, and sends back the verdict and the 10 step images. The model runs with **ONNX Runtime** instead of PyTorch because the free server has only 512 MB of memory. `export_onnx.py` proves it gives the same answers. `frontend/` is the React website and contains **no image processing**; it only displays the server's results.

---

## 5. How one image flows through the code

```
cell image (PNG)
   │  pipeline.load_image                      Stage 1  (M1)
   ▼
   │  crop_and_resize → 128 × 128              Stage 2  (M2)
   │  to_gray / to_hsv                         Stage 3  (M2)
   │  enhance_contrast (CLAHE)                 Stage 4  (M3)
   │  denoise (median 3 × 3)                   Stage 5  (M3)
   ├──────────────────────────────┐
   │                              ▼
   │                Otsu on gray → cell mask          Stage 6  (M4)
   │                erosion/dilation, convex hull     Stage 7  (M2) / 8 (M5)
   │                Otsu on saturation → stain mask   Stage 6  (M4)
   │                erosion + dilation → contours     Stage 7-8
   │                → "2 stained spots" (red outline)
   ▼
MobileNetV2 (classifier.predict)                    Stage 9  (M5)
   ▼
"Parasitized, 98.6%"   →   compared with the true label on the test set = Stage 10
```

---

## 6. Numbers to remember for the viva

| | |
|---|---|
| Dataset | 27,558 cell images (13,779 infected, 13,779 healthy), NIH |
| Split | 19,282 train / 4,132 validation / 4,144 test |
| Model | MobileNetV2 (≈3.5 M parameters), ImageNet pre-trained, 128 × 128 input |
| Training | 2 + 5 epochs, Adam, batch 64, 37 minutes on a laptop CPU |
| Accuracy / Precision / Recall / F1 | **96.6% / 96.9% / 96.3% / 96.6%** |
| Missed infections | 76 of 2,072 infected test cells (3.7%) |
| False alarms | 64 of 2,072 healthy test cells (3.1%) |
| Stages 6-8 alone | spot outlined in 86.3% of infected cells, false spot in 3.0% of healthy |
| Improvement story | first run 92.7% → CLAHE + longer fine-tuning 96.3% → full dataset 96.6% |

---

## 7. Suggested split for the presentation

| Part | Covers | Files |
|------|--------|-------|
| 1. Problem + dataset | Malaria, why automate, NIH dataset, Stage 1 | `pipeline.py` (load), report ch. 1 and 7 |
| 2. Pre-processing | Stages 2-5: crop/resize, HSV, CLAHE, median filter | `pipeline.py`, report ch. 8.1-8.5 |
| 3. Segmentation | Stages 6-8: Otsu, morphology, contours, convex hull | `pipeline.py`, report ch. 8.6-8.8 |
| 4. CNN + results + demo | Stage 9-10: transfer learning, metrics, live demo | `classifier.py`, `train.py`, report ch. 9-11 |

Each person should also be ready for the common questions in [`FACULTY_GUIDE.md`](FACULTY_GUIDE.md) (section 7).

---

## 8. Common tasks

| I want to... | Command |
|--------------|---------|
| Run the demo without Node.js | `streamlit run app.py` |
| Run the tests | `python -m pytest tests -q` |
| Re-train (needs the dataset in `data/cell_images/`) | `python train.py` (about 40 min) |
| Measure Stages 6-8 | `python evaluate_segmentation.py` |
| Redo the figures | `python make_figures.py` |
| Rebuild the report (needs Node.js + LibreOffice) | `python report/build.py` |
| Update the live website | push to `main` on GitHub; Render and Vercel redeploy automatically |

**Troubleshooting**

- *The live site says "Waking up the analysis server"*: normal on the free plan; wait up to a minute.
- *`pip install` fails*: use Python 3.11 (`py -3.11 -m venv .venv` on Windows).
- *Port 8000 already in use*: run `uvicorn backend.main:app --port 8001` and start the website with `set VITE_API_URL=http://localhost:8001` before `npm run dev` (PowerShell: `$env:VITE_API_URL="http://localhost:8001"`).
