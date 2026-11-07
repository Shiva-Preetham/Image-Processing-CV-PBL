<div align="center">

# 🔬 MalariaVision

### Malaria parasite detection from blood-smear images using image processing and transfer learning

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV2-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/status-prototype%20v0.1-orange)

**Project-based learning · Fundamental of Image Processing and Computer Vision Lab (26B16CS316) · 2026-27**

<img src="outputs/pipeline_parasitized.png" alt="Every pipeline stage on an infected cell" width="100%">

</div>

---

## 🩸 The problem in 30 seconds

Malaria is confirmed by a technician who looks at a stained blood smear under a microscope and checks the cells one by one. That is slow and tiring, and a tired eye can miss an infected cell.

The parasite shows up as a **dark purple spot** inside the red blood cell (the Giemsa stain colours it). MalariaVision looks at a single cell image and:

1. **cleans** it with classical image processing,
2. **points to** the suspicious stained spot (Otsu thresholding, morphology, contours), and
3. **decides** *Parasitized* or *Uninfected* with a pre-trained CNN (transfer learning).

Every technique comes from the lab syllabus (Modules 1 to 5).

---

## 🧭 How it works

```mermaid
flowchart LR
    A["1. Load image<br/>M1"] --> B["2. Crop + resize<br/>M2"]
    B --> C["3. Gray / HSV<br/>M2"]
    C --> D["4. Histogram<br/>equalisation · M3"]
    D --> E["5. Median filter<br/>M3"]
    E --> F["6. Otsu threshold<br/>M4"]
    F --> G["7. Erosion + dilation<br/>M2"]
    G --> H["8. Contours<br/>M5"]
    E --> I["9. MobileNetV2<br/>transfer learning · M5"]
    H --> J(["📍 Where is the stain?"])
    I --> K(["✅ Parasitized / Uninfected"])
    K --> L["10. Evaluation<br/>CO4"]
```

| # | Stage | What it does, in plain words | Technique | Module |
|:-:|-------|------------------------------|-----------|:------:|
| 1 | Load | Read the microscope image and display it | `cv2.imread`, Matplotlib | **M1** |
| 2 | Crop + resize | Cut off the black border, make every image 128 × 128 | Bounding box, `cv2.resize` | **M2** |
| 3 | Colour spaces | Grayscale for brightness; HSV saturation makes the purple stain glow | `cv2.cvtColor` | **M2** |
| 4 | Enhancement | Spread brightness out so every slide has similar contrast | CLAHE histogram equalisation (V channel) | **M3** |
| 5 | Noise removal | Remove tiny specks while keeping edges sharp | Median filter (Gaussian / Average selectable) | **M3** |
| 6 | Segmentation | Pick thresholds automatically: cell vs. background, then stained vs. normal | Otsu thresholding | **M4** |
| 7 | Mask cleanup | Delete small noisy dots, keep the real regions | Erosion + dilation | **M2** |
| 8 | Region detection | Outline the cell (green) and the stained spots (red) | Contour detection, convex hull | **M5** |
| 9 | Classification | Re-use a network trained on 1.2 M photos; retrain its last layers for our task | MobileNetV2 transfer learning, flip/rotate augmentation | **M5** (+M2) |
| 10 | Evaluation | Test on images the model has never seen | Accuracy, precision, recall, F1, confusion matrix | **CO4** |

---

## 📊 Results (prototype v0.1)

**Baseline run** (balanced subset of 6,000 images, 7 minutes on a laptop CPU), tested on **900 images the model never saw**:

| Accuracy | Precision | Recall | F1-score | Missed infections |
|:--------:|:---------:|:------:|:--------:|:-----------------:|
| **92.7%** | **95.5%** | **89.6%** | **92.4%** | 47 of 450 |

<img src="outputs/confusion_matrix.png" alt="Confusion matrix" width="38%"> <img src="outputs/training_curves.png" alt="Training curves" width="60%">

The baseline showed that plain histogram equalisation over-darkens cells and that the model was still improving, so v0.1 switches to CLAHE and fine-tunes longer. Updated numbers follow when that run finishes.

**Stain-spot highlighting (Stages 6-8)**, measured on 800 random cells: a spot is outlined in **87%** of infected cells and wrongly outlined in only **3%** of healthy ones.

---

## 🖥️ Live demo

```bash
streamlit run app.py
```

The demo opens in the browser with three tabs:

- **🔍 Live diagnosis**: pick one of 12 test-set cells (or upload your own) and see all 10 stages, the stained-spot outline and the CNN's verdict with its confidence. The sidebar lets you switch the noise filter (Median / Gaussian / Average) and kernel sizes to see how each stage reacts.
- **📊 Model performance**: accuracy, precision, recall, F1, confusion matrix and training curves.
- **🗺️ Syllabus map**: every stage mapped to its module and course outcome.

---

## 🚀 Quick start

**1. Clone and install** (Python 3.11 recommended)

```bash
git clone https://github.com/Shiva-Preetham/Image-Processing-CV-PBL.git
cd Image-Processing-CV-PBL
python -m venv .venv
```

```bash
.venv\Scripts\activate            # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

**2. Try the demo right away.** The trained model and 12 sample cells are already in the repo.

```bash
streamlit run app.py
```

**3. (Optional) Download the dataset to re-train** (337 MB, official NIH source)

```powershell
# Windows PowerShell
Invoke-WebRequest https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip -OutFile data/cell_images.zip
Expand-Archive data/cell_images.zip -DestinationPath data
```

```bash
# macOS / Linux
mkdir -p data && curl -L -o data/cell_images.zip https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip
unzip -q data/cell_images.zip -d data
```

**4. Train, evaluate and regenerate the figures**

```bash
python train.py                  # prototype: 3,000 images per class (~10 min on a laptop CPU)
python train.py --per-class 0    # full dataset: all 27,558 images
python make_figures.py           # dataset overview + per-stage pipeline figures
```

---

## 📁 Project structure

```
├── pipeline.py          Stages 1-8  classical image processing (each function tagged with its module)
├── classifier.py        Stage 9     MobileNetV2 transfer learning + flip/rotate augmentation
├── train.py             Stages 9-10 two-phase training and test-set evaluation
├── make_figures.py      figures for the report and viva
├── app.py               Streamlit demo for the faculty review
├── FACULTY_GUIDE.md     5-minute talk track, demo script and viva Q&A
├── models/              trained weights (~9 MB)
├── samples/             12 test-set cells used by the demo
├── outputs/             metrics.json, confusion matrix, curves, pipeline figures, training log
└── requirements.txt
```

---

## 🗂️ Dataset

**Malaria Cell Images Dataset** from the U.S. National Library of Medicine (NIH / LHNCBC): 27,558 single-cell images from Giemsa-stained thin blood smears, **13,779 parasitized** and **13,779 uninfected**.

<img src="outputs/dataset_overview.png" alt="Sample cells from both classes" width="100%">

Split used: 70% train, 15% validation, 15% test (stratified, seed 42). Test images are never used during training.

---

## 🎯 Course outcome mapping

| CO | What the project shows |
|----|------------------------|
| **CO1** | Image loading, display, saving, resizing, cropping and colour-space conversion |
| **CO2** | Histogram enhancement, spatial filtering, thresholding and morphology |
| **CO3** | Contour detection and CNN-based recognition with transfer learning |
| **CO4** | A healthcare application, built end to end and evaluated with standard metrics |

---

## 🛣️ Roadmap

- [x] **v0.1 prototype**: full 10-stage pipeline, trained model, demo app, report figures
- [ ] Faculty review and feedback
- [ ] **v0.2**: improvements from the feedback (candidates: train on the full dataset, adaptive thresholding for unevenly stained cells, a Jupyter notebook version for submission)
- [ ] Final report and viva

---

## 👥 Team

| Name | Enrollment No. |
|------|----------------|
| Lakshita Gautam | 23103343 |
| Ranjeet Kaur | 2401030065 |
| Navya Banga | 2401030070 |
| Shiva Preetham Chelimila | 2401030078 |

**Faculty:** Ms. Neha · Dr. Pawan Kumar Upadhyay

---

## 📚 References

- Rajaraman S., Antani S. K., Poostchi M., *et al.* "Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images." *PeerJ* 6:e4568, 2018. (Source of the dataset.)
- Sandler M., Howard A., Zhu M., Zhmoginov A., Chen L.-C. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR*, 2018.
- Otsu N. "A Threshold Selection Method from Gray-Level Histograms." *IEEE Trans. SMC* 9(1), 1979.
- Gonzalez R. C., Woods R. E. *Digital Image Processing*, 4th ed., Pearson, 2018.
