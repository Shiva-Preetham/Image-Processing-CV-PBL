"""
MalariaVision - live demo for the faculty review.

    streamlit run app.py

Pick (or upload) a blood-cell image and watch every pipeline stage,
from loading the image to the CNN's final answer.
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from classifier import MODEL_PATH, load_trained, predict
from pipeline import FILTERS, STEPS, decode_image, load_image, run_pipeline, to_hsv

ROOT = Path(__file__).parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "outputs"

st.set_page_config(page_title="MalariaVision", page_icon="🔬", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.verdict {border-radius: 14px; padding: 18px 22px; color: white; margin-bottom: 10px;}
.verdict h2 {color: white; margin: 0 0 4px 0; font-size: 1.9rem;}
.verdict p {margin: 0; opacity: .95;}
.infected {background: linear-gradient(135deg, #b42318, #e8590c);}
.healthy {background: linear-gradient(135deg, #087f5b, #2b8a3e);}
.badge {display: inline-block; background: #ede7f6; color: #4a2c8f; border-radius: 6px;
        padding: 1px 8px; font-size: .75rem; font-weight: 600;}
.step-note {font-size: .8rem; color: #666; margin-top: -6px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_trained() if MODEL_PATH.exists() else None


def big(img, size=256):
    """Enlarge small images (keeping their shape) so they are easy to see on a projector."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_NEAREST)


def sample_files():
    return sorted(SAMPLES.glob("*.png")) if SAMPLES.exists() else []


# ------------------------------- sidebar -------------------------------
st.sidebar.title("🔬 MalariaVision")
st.sidebar.caption("Malaria parasite detection from blood-smear cell images")

samples = sample_files()
source = st.sidebar.radio("Cell image", ["Test-set sample", "Upload an image"],
                          index=0 if samples else 1)
rgb, true_label, image_name = None, None, None
if source == "Test-set sample" and samples:
    names = {f"{p.stem.split('_')[0].title()}  #{p.stem.split('_')[1]}": p for p in samples}
    labels = list(names)
    # Open on a typical infected cell; Parasitized #1 is a known hard case (see FACULTY_GUIDE.md)
    default = labels.index("Parasitized  #2") if "Parasitized  #2" in labels else 0
    choice = st.sidebar.selectbox("Choose a sample (never seen in training)", labels,
                                  index=default)
    rgb = load_image(names[choice])
    true_label, image_name = choice.split()[0], names[choice].name
else:
    up = st.sidebar.file_uploader("Single-cell image (PNG / JPG)", type=["png", "jpg", "jpeg"])
    if up is not None:
        rgb, image_name = decode_image(up.getvalue()), up.name

st.sidebar.markdown("---")
st.sidebar.subheader("Pipeline settings")
filter_method = st.sidebar.selectbox("Noise filter  (Module 3)", FILTERS)
ksize = st.sidebar.select_slider("Filter kernel size", [3, 5, 7], value=3)
morph_k = st.sidebar.select_slider("Morphology kernel  (Module 2)", [3, 5], value=3)
st.sidebar.caption("The CNN was trained with Median 3x3. Change the settings to see "
                   "how each stage reacts.")

model = get_model()

# ------------------------------- header --------------------------------
st.title("MalariaVision")
st.markdown("**Is this red blood cell infected with the malaria parasite?** "
            "Classical image processing cleans and segments the cell, then a "
            "pre-trained CNN gives the final answer.")

tab_demo, tab_perf, tab_map = st.tabs(["🔍 Live diagnosis", "📊 Model performance",
                                       "🗺️ Syllabus map"])

# ---------------------------- live diagnosis ---------------------------
with tab_demo:
    if rgb is None:
        st.info("Upload a single-cell image in the sidebar to begin.")
    else:
        r = run_pipeline(rgb, filter_method, ksize, morph_k)
        c1, c2, c3 = st.columns([1, 1, 1.3])
        c1.image(big(r.steps["2_resized"]), caption=f"Input: {image_name}")
        c2.image(big(r.steps["8_contours"]),
                 caption="Detected regions (green = cell, red = stained spots)")
        with c3:
            if model is None:
                st.warning("No trained model found. Run `python train.py` first.")
            else:
                label, p = predict(model, r.cnn_input)
                conf = p if label == "Parasitized" else 1 - p
                css = "infected" if label == "Parasitized" else "healthy"
                icon = "🦠" if label == "Parasitized" else "✅"
                st.markdown(f"<div class='verdict {css}'><p>CNN verdict (Stage 9)</p>"
                            f"<h2>{icon} {label}</h2><p>Confidence {conf:.1%}</p></div>",
                            unsafe_allow_html=True)
                st.progress(p, text=f"Probability of infection: {p:.1%}")
            st.metric("Stained spots found (Stages 6-8)", len(r.spot_contours),
                      help="Dark purple regions inside the cell, found by Otsu "
                           "thresholding + morphology + contours.")
            if true_label:
                st.caption(f"Ground truth from the dataset: **{true_label}**")

        st.markdown("### How the computer got there - step by step")
        for row in (STEPS[:5], STEPS[5:]):
            cols = st.columns(5)
            for col, (key, title, module, note) in zip(cols, row):
                with col:
                    st.markdown(f"**{title}** <span class='badge'>{module}</span>",
                                unsafe_allow_html=True)
                    st.image(big(r.steps[key], 192))
                    st.markdown(f"<div class='step-note'>{note}</div>",
                                unsafe_allow_html=True)

        with st.expander("Stage 4 in detail - brightness histogram before and after"):
            fig, ax = plt.subplots(1, 2, figsize=(9, 2.6))
            for a, img, t in ((ax[0], r.steps["2_resized"], "Before"),
                              (ax[1], r.steps["4_enhanced"], "After equalisation")):
                v = to_hsv(img)[:, :, 2]
                a.hist(v[v > 10], bins=48, range=(0, 255), color="#5b4b8a")
                a.set_title(t)
                a.set_xlim(0, 255)
            fig.tight_layout()
            st.pyplot(fig)
            st.caption("Equalisation spreads the brightness values across the full 0-255 "
                       "range, so faint details inside the cell become visible.")

        with st.expander("Stage 6 in detail - the thresholds Otsu chose"):
            st.write(f"- Cell vs. background (grayscale): **{r.otsu_cell:.0f}**")
            st.write(f"- Stained vs. normal cell pixels (saturation): **{r.otsu_stain:.0f}**")
            st.write(f"- Stain contrast (spot vs. cell body): **{r.stain_gap:.0f}** - spots "
                     "are only reported when this gap is large enough.")

# --------------------------- model performance -------------------------
with tab_perf:
    mpath = OUT / "metrics.json"
    if not mpath.exists():
        st.info("Run `python train.py` to produce the evaluation results.")
    else:
        m = json.loads(mpath.read_text())
        d = m["dataset"]
        st.markdown(f"Evaluated on **{d['test']:,} test images** that the model never saw "
                    f"during training (trained on {d['train']:,}, validated on {d['val']:,}).")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{m['accuracy']:.1%}", help="Correct answers out of all cells")
        k2.metric("Precision", f"{m['precision']:.1%}",
                  help="When the model says 'infected', how often is it right?")
        k3.metric("Recall", f"{m['recall']:.1%}",
                  help="Out of all truly infected cells, how many did it catch?")
        k4.metric("F1-score", f"{m['f1']:.1%}", help="Balance of precision and recall")
        cm = m["confusion_matrix"]
        st.markdown(f"Missed infections (false negatives): **{cm['FN']}**  ·  "
                    f"False alarms (false positives): **{cm['FP']}**. In diagnosis a missed "
                    "infection is the costlier mistake, which is why we report recall, "
                    "not just accuracy.")
        a, b = st.columns([1, 1.6])
        a.image(str(OUT / "confusion_matrix.png"))
        b.image(str(OUT / "training_curves.png"))
        if (OUT / "sample_predictions.png").exists():
            st.image(str(OUT / "sample_predictions.png"))

# ------------------------------ syllabus map ---------------------------
with tab_map:
    st.markdown("Every stage uses a technique from the lab syllabus (Modules 1-5).")
    rows = [
        ("Load, display, save", "cv2.imread / Matplotlib", "M1", "CO1"),
        ("Crop + resize", "Bounding box crop, cv2.resize", "M2", "CO1"),
        ("Colour spaces", "RGB -> Grayscale, RGB -> HSV", "M2", "CO1"),
        ("Enhancement", "CLAHE histogram equalisation (V channel)", "M3", "CO2"),
        ("Noise removal", "Median / Gaussian / Average filter", "M3", "CO2"),
        ("Segmentation", "Otsu thresholding", "M4", "CO2"),
        ("Mask cleanup", "Erosion + dilation", "M2", "CO2"),
        ("Region detection", "Contour detection, convex hull", "M5", "CO3"),
        ("Classification", "MobileNetV2 transfer learning (+ flip/rotate augmentation)",
         "M5 (+M2)", "CO3"),
        ("Evaluation", "Accuracy, precision, recall, F1, confusion matrix", "Project", "CO4"),
    ]
    st.table([{"Stage": i, "What it does": s, "Technique": t, "Module": m, "CO": c}
              for i, (s, t, m, c) in enumerate(rows, 1)])
