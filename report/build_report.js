// Builds report/MalariaVision_Project_Report.docx from the project's real results.
// Run through `python report/build.py`, which also makes the PDF and fills in the
// page numbers of the contents and the list of figures (pages.json).
const fs = require("fs");
const path = require("path");
const docx = require("docx");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, PageNumber, Footer, NumberFormat, BorderStyle, LevelFormat, Tab,
  TabStopType, VerticalAlign, PageBreak,
} = docx;

const HERE = __dirname;
const ROOT = path.join(HERE, "..");
const OUT = path.join(ROOT, "outputs");
const readJSON = (p, fallback = null) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : fallback);

const M = readJSON(path.join(OUT, "metrics.json"));
const S = readJSON(path.join(OUT, "segmentation_eval.json"));
const ONNX = readJSON(path.join(OUT, "onnx_check.json"), { max_probability_difference: 0, samples_checked: 12 });
const DS = readJSON(path.join(OUT, "dataset_stats.json"));
const PAGES = readJSON(path.join(HERE, "pages.json"), {});

const FONT = "Times New Roman";
const TEXT_W = 9026; // A4 width 11906 minus 1-inch margins (DXA)
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const n = (x) => Number(x).toLocaleString("en-US");
const CM = M.confusion_matrix;
const D = M.dataset;
const H = M.history;
const headEpochs = H.filter((e) => e.phase.startsWith("Phase A")).length;
const ftEpochs = H.length - headEpochs;

// Earlier runs on the 6,000-image subset (recorded in outputs/ history of the repository)
const VERSIONS = [
  { name: "Baseline", data: "6,000-image subset", enh: "Plain equalizeHist", ft: "3 epochs, last 5 blocks",
    test: 900, acc: 0.9267, prec: 0.955, rec: 0.8956, f1: 0.9243, fn: 47, pos: 450 },
  { name: "v0.1", data: "6,000-image subset", enh: "CLAHE", ft: "5 epochs, last 8 blocks",
    test: 900, acc: 0.9633, prec: 0.9749, rec: 0.9511, f1: 0.9629, fn: 22, pos: 450 },
  { name: "v1.0 (final)", data: "Full dataset, 27,558 images", enh: "CLAHE", ft: `${ftEpochs} epochs, last 8 blocks`,
    test: D.test, acc: M.accuracy, prec: M.precision, rec: M.recall, f1: M.f1, fn: CM.FN, pos: CM.FN + CM.TP },
];

// ------------------------------------------------------------------ text helpers
// Inline markup: **bold**, *italic*, `code`, _{subscript}, ^{superscript}
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), ...base, bold: true }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), ...base, font: "Courier New", size: (base.size || 24) - 3 }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), ...base, subScript: true }));
    else if (t.startsWith("^{")) out.push(new TextRun({ text: t.slice(2, -1), ...base, superScript: true }));
    else out.push(new TextRun({ text: t.slice(1, -1), ...base, italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const body = [];
const tocEntries = []; // { level, text }
const figures = []; // { num, caption }
let tableNo = 0;
let figureNo = 0;
let listNo = 0;
let firstChapter = true;

// Empty spacer paragraphs after tables; removed again before a new chapter so a
// spacer can never spill onto an otherwise blank page.
const spacers = new Set();
function spacer(after = 120) {
  const sp = new Paragraph({ spacing: { after, line: 240 }, children: [] });
  spacers.add(sp);
  body.push(sp);
}
function h1(text) {
  while (body.length && spacers.has(body[body.length - 1])) body.pop();
  tocEntries.push({ level: 1, text });
  body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: !firstChapter,
    children: [new TextRun({ text, font: FONT, size: 28, bold: true })] }));
  firstChapter = false;
}
function h2(text) {
  tocEntries.push({ level: 2, text });
  body.push(new Paragraph({ heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: FONT, size: 24, bold: true })] }));
}
function p(text, opts = {}) {
  body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 140, line: 276 },
    ...opts, children: runs(text, { font: FONT, size: 24 }) }));
}
function list(items, numbered = false, alignment = AlignmentType.JUSTIFIED) {
  listNo += 1;
  items.forEach((item, i) => {
    body.push(new Paragraph({
      alignment, spacing: { after: i === items.length - 1 ? 200 : 70, line: 264 },
      numbering: { reference: numbered ? "numbers" : "bullets", level: 0, instance: listNo },
      children: runs(item, { font: FONT, size: 24 }),
    }));
  });
}
const bullets = (items) => list(items, false);
const numbered = (items) => list(items, true);
function eq(text) {
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 },
    children: runs(text, { font: "Cambria Math", size: 23 }) }));
}

const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };
// keep = "keep with next", set on every row but the last so a table never splits across pages
function cell(text, width, header = false, keep = false) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    shading: header ? { fill: "D9D9D9", type: ShadingType.CLEAR, color: "auto" } : undefined,
    children: [new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 0 }, keepNext: keep,
      children: runs(String(text), { font: FONT, size: 20, bold: header }) })],
  });
}
function table(caption, headers, rows, fractions) {
  tableNo += 1;
  const widths = fractions.map((f) => Math.floor(f * TEXT_W));
  widths[widths.length - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 80 },
    children: [new TextRun({ text: `Table ${tableNo}: ${caption}`, font: FONT, size: 20, bold: true })] }));
  body.push(new Table({
    width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((h, i) => cell(h, widths[i], true, true)) }),
      ...rows.map((r, k) => new TableRow({ cantSplit: true,
        children: r.map((c, i) => cell(c, widths[i], false, k < rows.length - 1)) })),
    ],
  }));
  spacer();
}
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(file, caption, widthInches = 6.2) {
  figureNo += 1;
  figures.push({ num: figureNo, caption });
  const { w, h } = pngSize(file);
  let width = widthInches * 96;
  let height = (width * h) / w;
  const maxH = 8.3 * 96; // keep the figure and its caption on one page
  if (height > maxH) { width = (width * maxH) / height; height = maxH; }
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
      transformation: { width: Math.round(width), height: Math.round(height) } })] }));
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: `Figure ${figureNo}: ${caption}`, font: FONT, size: 20, italics: true })] }));
}
function code(lines) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  body.push(new Table({
    width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: TEXT_W, type: WidthType.DXA },
      borders: { top: border, bottom: border, left: border, right: border },
      shading: { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" },
      margins: { top: 80, bottom: 80, left: 140, right: 140 },
      children: lines.map((l) => new Paragraph({ spacing: { after: 0 },
        children: [new TextRun({ text: l || " ", font: "Courier New", size: 17 })] })),
    })] })],
  }));
  spacer(140);
}
const shot = (name) => path.join(HERE, "screenshots", name);
const asset = (name) => path.join(HERE, "assets", name);
const outFig = (name) => path.join(OUT, name);

// ================================================================== CONTENT
h1("1. INTRODUCTION");
h2("1.1 Background");
p("Malaria is a life-threatening disease caused by *Plasmodium* parasites, which are spread to people through the bites of infected female *Anopheles* mosquitoes. The World Health Organization estimated about 263 million malaria cases and 597,000 malaria deaths worldwide in 2023 [1]. Early and correct diagnosis is essential, because it allows prompt treatment and limits further transmission.");
p("The standard way to confirm malaria is to examine a Giemsa-stained thin blood smear under a light microscope. The stain colours the chromatin of the parasite a dark purple, so an infected red blood cell shows one or more dark purple dots or rings, while a healthy cell is evenly pink. A technician must inspect hundreds of cells on every slide. The work is slow and tiring, and its accuracy depends on the experience of the person and on the quality of the slide.");
p("**MalariaVision** is a computer vision system that automates this first check for single-cell images. It applies classical image processing (enhancement, filtering, thresholding, morphology and contour detection) to clean each cell and outline the dark-stained region, and then uses a pre-trained convolutional neural network (CNN), adapted through transfer learning, to classify the cell as *Parasitized* or *Uninfected*. The system is trained and evaluated on the public Malaria Cell Images Dataset of the U.S. National Library of Medicine [2], [3], and it is available as a web application that shows every processing stage step by step.");
p("Every technique used in the project belongs to Modules 1 to 5 of the syllabus of the Fundamental of Image Processing and Computer Vision Lab (26B16CS316), and every stage in the source code is labelled with the module it comes from (Chapter 13).");
h2("1.2 Problem Statement");
p("Manual identification of malaria parasites from microscopic blood-smear images is time-consuming and requires careful examination of a large number of cells. Image quality also varies because of differences in brightness, contrast, staining and noise, which makes the task harder and the results less consistent.");
p("The aim of the project is to develop an automated system that processes microscopic cell images and classifies them as parasitized or uninfected using image processing and deep-learning techniques, and that also shows where the suspicious region is, so that a person can check the result. This addresses the course requirement of applying image-based or AI-based techniques to a meaningful healthcare problem.");
h2("1.3 Objectives");
bullets([
  "To develop a computer vision system for detecting malaria-infected cells from microscopic blood-smear images.",
  "To apply image preprocessing, enhancement, filtering, thresholding, morphological operations and contour detection techniques from the syllabus.",
  "To classify cells as infected or uninfected using transfer learning with a pre-trained CNN model (MobileNetV2).",
  "To evaluate the classification performance using accuracy, precision, recall, F1-score and a confusion matrix on images that were never used for training.",
  "To measure how reliably the classical stages outline the stained parasite region.",
  "To demonstrate the practical application of the system through a web application that shows every stage, and to deploy it online.",
]);

h1("2. FEATURES OF THE SYSTEM");
h2("2.1 Ten-Stage Processing Pipeline");
p("Every input cell passes through ten stages: loading (Stage 1), cropping and resizing (Stage 2), colour-space conversion (Stage 3), histogram equalisation (Stage 4), noise filtering (Stage 5), Otsu thresholding (Stage 6), erosion and dilation (Stage 7), contour detection (Stage 8), CNN classification (Stage 9) and evaluation (Stage 10). Stages 1 to 8 are implemented in `pipeline.py` with OpenCV and NumPy, and each function is labelled with its syllabus module.");
h2("2.2 Stained-Spot Highlighting");
p(`Stages 6 to 8 find the outline of the cell, drawn in green, and the dark-stained spots inside it, drawn in red. The number of spots and their total area are reported next to the answer of the CNN. This makes the result explainable: the user can see where the suspicious region is and judge whether the classification is reasonable. On the test set a spot is outlined in ${pct(S.spot_detection_rate_infected)} of infected cells and in only ${pct(S.false_spot_rate_healthy)} of healthy cells (Section 10.6).`);
h2("2.3 CNN Classification with Confidence");
p("The cleaned 128 × 128 image produced by Stages 2 to 5 is passed to a MobileNetV2 network fine-tuned on the dataset. The network returns the probability that the cell is parasitized. The class with the higher probability is the verdict, and its probability is shown as the confidence.");
h2("2.4 Live Diagnosis Page");
p("The web application opens with a gallery of twelve test cells (six parasitized and six uninfected) that the model never saw during training. Selecting a cell, or uploading a PNG or JPG image of up to 5 MB by clicking or by drag and drop, shows the input, the detected regions, a verdict card with the probability of infection, the number of stained spots and, for gallery cells, whether the verdict agrees with the dataset label.");
h2("2.5 Step-by-Step View");
p("Below the verdict, all ten intermediate images are shown in order, each with a title, a one-line explanation and a coloured tag that names its syllabus module. Two detail panels show the brightness histogram before and after equalisation and the thresholds selected by Otsu's method.");
h2("2.6 Pipeline Tuning");
p("The noise filter can be switched between Median, Gaussian and Average, the filter kernel size between 3, 5 and 7, and the morphology kernel size between 3 and 5. Every stage is recomputed immediately, which demonstrates the effect of each technique on the segmentation and on the confidence of the CNN.");
h2("2.7 Model Performance Page");
p("A second page shows the accuracy, precision, recall and F1-score on the test set, the confusion matrix, the training curves, a comparison with earlier versions of the model and a grid of sample predictions.");
h2("2.8 Syllabus Map");
p("A third page lists all stages with the technique used, the syllabus module and the course outcome each one supports.");
h2("2.9 Offline Demonstration");
p("A Streamlit version of the same demonstration (`app.py`) runs with a single command and needs no internet connection. It is kept as a backup for presentations.");

h1("3. SYSTEM REQUIREMENTS");
h2("3.1 Hardware Requirements");
table("Hardware requirements", ["Component", "Minimum / Recommended Configuration"], [
  ["Processor", "Any 64-bit dual-core processor; quad-core or better recommended for training. Developed on an Intel Core i5-11400H (6 cores, 12 threads)"],
  ["Memory", "4 GB RAM to run the application; 8 GB recommended for training (8 GB used)"],
  ["Storage", "About 3 GB free: Python environment, dataset (337 MB compressed), models and outputs"],
  ["Graphics card", "Not required. The final model was trained on the CPU"],
  ["Display", "1280 × 720 or higher; the web pages also adapt to phone screens"],
], [0.25, 0.75]);
h2("3.2 Software Requirements");
table("Software requirements", ["Software", "Version / Notes"], [
  ["Operating system", "Windows 10/11, Linux or macOS (developed on Windows 11)"],
  ["Python", "Python 3.11 (3.11.9 used)"],
  ["OpenCV", "opencv-python 5.0 (opencv-python-headless on the server)"],
  ["NumPy, Matplotlib", "NumPy 2.4, Matplotlib 3.11"],
  ["PyTorch", "PyTorch 2.15 and torchvision 0.29, CPU build (training only)"],
  ["scikit-learn", "1.9 (data split and evaluation metrics)"],
  ["ONNX Runtime", "1.31 (model inference on the server)"],
  ["FastAPI, Uvicorn", "FastAPI 0.162 and Uvicorn 0.54 (web API)"],
  ["Node.js", "20.19 or later (24 used), for the React web application"],
  ["React, Vite", "React 19 and Vite 8"],
  ["Streamlit", "1.67 (offline demonstration)"],
  ["Web browser", "Any current version of Chrome, Edge or Firefox"],
  ["Editor", "Visual Studio Code or any Python editor"],
], [0.25, 0.75]);

h1("4. SYSTEM ARCHITECTURE");
p("MalariaVision has two parts (Figure 1). The offline part trains and evaluates the classifier on a laptop. The online part is a web application: a React front end hosted on Vercel sends the image and the chosen settings to a FastAPI server hosted on Render, which runs the image-processing pipeline and the trained network and returns all results as JSON. All image processing is done in Python with OpenCV, using the same `pipeline.py` that prepares the training images, so the web application shows exactly what the model sees.");
figure(asset("architecture.png"), "System architecture of MalariaVision");
h2("4.1 Training Flow");
numbered([
  `\`train.py\` lists the ${n(D.train + D.val + D.test)} images and splits them into training (70%), validation (15%) and test (15%) sets, keeping the twelve demonstration cells in the test set.`,
  "Each image passes through Stages 1 to 5 of `pipeline.py`; training images are also flipped and rotated at random.",
  "MobileNetV2 is trained in two phases (Section 9.4), and the weights with the best validation accuracy are kept.",
  "The kept model is evaluated once on the test set. The metrics, the confusion matrix, the training curves and sample predictions are saved in `outputs/`.",
  "`evaluate_segmentation.py` measures Stages 6 to 8 on the same test set, and `export_onnx.py` converts the model to ONNX and checks that its answers match.",
]);
h2("4.2 Request Flow for Analysing a Cell");
numbered([
  "The user selects a sample cell or uploads an image in the browser.",
  "The React application sends `GET /api/samples/{name}/analyze` or `POST /api/analyze` with the image and the filter settings.",
  "The server checks the settings and the image (file type and a size of at most 5 MB) and runs Stages 1 to 8 with OpenCV.",
  "The cleaned image from Stage 5 is classified by the ONNX model (Stage 9).",
  "The server returns the verdict, the probability, the Otsu thresholds, the spot count, the brightness histograms and the ten step images as PNG data.",
  "The browser shows the verdict card and the step-by-step view.",
]);

h1("5. TECHNOLOGY STACK AND PROJECT STRUCTURE");
h2("5.1 Technology Stack");
table("Technologies used and their role", ["Technology", "Role in MalariaVision"], [
  ["Python", "Programming language for image processing, training, evaluation and the web API"],
  ["OpenCV", "All image-processing stages: loading, resizing, colour conversion, CLAHE, filtering, Otsu thresholding, morphology, contours, flips and rotations"],
  ["NumPy", "Images as arrays, masks and pixel statistics"],
  ["Matplotlib", "Pipeline figures, histograms, confusion matrix and training curves"],
  ["PyTorch, torchvision", "MobileNetV2 with ImageNet weights and the training loop"],
  ["scikit-learn", "Stratified train/validation/test split; accuracy, precision, recall, F1-score and confusion matrix"],
  ["ONNX, ONNX Runtime", "Exported model and lightweight inference on the server"],
  ["FastAPI, Uvicorn", "Web API that runs the pipeline and the model"],
  ["React, Vite", "Web user interface (single-page application)"],
  ["Streamlit", "Offline one-command demonstration"],
  ["pytest", "Automated tests of the web API"],
  ["Git, GitHub", "Version control and sharing of the code"],
  ["Render, Vercel", "Cloud hosting of the API and of the web application"],
], [0.27, 0.73]);
h2("5.2 Project Directory Structure");
code([
  "Image-Processing-CV-PBL/",
  "├── pipeline.py               Stages 1-8: classical image processing",
  "├── classifier.py             Stage 9: MobileNetV2 and data augmentation",
  "├── train.py                  Stages 9-10: training and test-set evaluation",
  "├── evaluate_segmentation.py  Stages 6-8 measured on the test set",
  "├── export_onnx.py            exports the model to ONNX and checks it",
  "├── make_figures.py           figures for the report",
  "├── app.py                    Streamlit demonstration (offline backup)",
  "├── backend/                  FastAPI server (main.py, model.py)",
  "├── frontend/                 React web application (src/, components/)",
  "├── tests/                    automated API tests (pytest)",
  "├── models/                   trained weights (.pt and .onnx)",
  "├── samples/                  twelve test-set cells for the demonstration",
  "├── outputs/                  metrics, figures, logs and the test split",
  "├── report/                   this report and the script that builds it",
  "├── render.yaml               Render deployment configuration",
  "└── requirements.txt          Python packages",
]);

h1("6. DESCRIPTION OF MODULES");
h2("6.1 Image Processing Module (pipeline.py)");
p("This module contains Stages 1 to 8. Each function is short and does one syllabus operation, and `run_pipeline()` runs them in order and keeps every intermediate image, which the web application and the report figures display.");
table("Functions of pipeline.py", ["Function", "Stage", "Module", "Purpose"], [
  ["load_image, decode_image", "1", "M1", "Read an image file or uploaded bytes; convert BGR to RGB"],
  ["crop_and_resize", "2", "M2", "Crop to the cell's bounding box and resize to 128 × 128"],
  ["to_gray, to_hsv", "3", "M2", "Colour-space conversion"],
  ["enhance_contrast", "4", "M3", "CLAHE histogram equalisation of the V channel"],
  ["denoise", "5", "M3", "Median, Gaussian or average filter"],
  ["otsu_threshold, otsu_inside", "6", "M4", "Otsu threshold of a whole image or of a masked region"],
  ["clean_mask, fill_holes", "7", "M2", "Erosion then dilation (opening); dilation then erosion (closing)"],
  ["find_contours", "8", "M5", "External contours above a minimum area"],
  ["run_pipeline", "1-8", "-", "Runs all stages and returns all intermediate images"],
], [0.30, 0.09, 0.10, 0.51]);
h2("6.2 Classifier and Training Modules (classifier.py, train.py)");
p("`classifier.py` builds MobileNetV2 with ImageNet weights and a new two-class output layer, freezes or unfreezes its blocks, applies the random flips and rotations, and provides the `CellDataset` class that loads an image, runs Stages 1 to 5 and returns a normalised tensor. `train.py` splits the data, runs the two training phases, keeps the best model, evaluates it on the test set and saves the metrics, the figures, the log and the list of test images.");
h2("6.3 Web API (backend/main.py)");
table("API endpoints", ["Endpoint", "Method", "Description"], [
  ["/api/health", "GET", "Reports that the server and the model are ready"],
  ["/api/samples", "GET", "Lists the twelve demonstration cells with their labels"],
  ["/api/samples/{name}/analyze", "GET", "Runs all stages on a demonstration cell; settings as query parameters"],
  ["/api/analyze", "POST", "Runs all stages on an uploaded image (multipart form)"],
  ["/api/metrics", "GET", "Returns the test-set metrics and the training history"],
  ["/static/samples, /static/outputs", "GET", "Demonstration images and result figures"],
], [0.37, 0.12, 0.51]);
p("The server rejects invalid input: an unknown sample gives HTTP 404, a filter other than Median, Gaussian or Average or a kernel size other than the allowed values gives HTTP 422, a file that is not an image gives HTTP 400 and a file larger than 5 MB gives HTTP 413. The model runs with ONNX Runtime (`backend/model.py`), so the server does not need PyTorch.");
h2("6.4 Web Front End (frontend/src)");
table("Main components of the web application", ["Component", "Purpose"], [
  ["App.jsx", "Page layout, tabs and the waiting screen while the free server wakes up"],
  ["Diagnosis.jsx", "Sample gallery, upload, pipeline settings, verdict card, step grid and detail panels"],
  ["Histogram.jsx", "Brightness histograms before and after equalisation"],
  ["Performance.jsx", "Metric cards, confusion matrix, version comparison and sample predictions"],
  ["LineChart.jsx", "Training curves"],
  ["SyllabusMap.jsx", "Stage flow, module table and course outcomes"],
  ["api.js", "Requests to the web API"],
], [0.27, 0.73]);

h1("7. DATASET");
h2("7.1 Malaria Cell Images Dataset");
p(`The project uses the Malaria Cell Images Dataset published by the Lister Hill National Center for Biomedical Communications of the U.S. National Library of Medicine [2], [3]. Giemsa-stained thin blood smears were photographed through a microscope, individual red blood cells were segmented and saved as separate images, and each cell was labelled by an expert. The dataset contains ${n(DS.Parasitized.images + DS.Uninfected.images)} cell images, ${n(DS.Parasitized.images)} parasitized and ${n(DS.Uninfected.images)} uninfected, so the two classes are perfectly balanced.`);
table("Dataset properties (Module 1: image properties)", ["Property", "Value"], [
  ["Source", "U.S. National Library of Medicine (LHNCBC), public download"],
  ["Images", `${n(DS.Parasitized.images + DS.Uninfected.images)} single-cell images`],
  ["Classes", `Parasitized (${n(DS.Parasitized.images)}) and Uninfected (${n(DS.Uninfected.images)})`],
  ["Format", "RGB PNG, one cell on a black background"],
  ["Image size", `Varies from ${DS.all.width.min} to ${DS.all.width.max} pixels wide and ${DS.all.height.min} to ${DS.all.height.max} pixels high; median about ${DS.Uninfected.width.median} × ${DS.Uninfected.height.median}`],
], [0.25, 0.75]);
h2("7.2 Dataset Exploration");
p("Figure 2 shows random cells of both classes. Infected cells contain dark purple spots, rings or crescents, while healthy cells are evenly coloured. Staining varies from pink to violet between slides, and the image sizes differ, which is why every image is enhanced and resized to a fixed size in the first stages of the pipeline.");
figure(outFig("dataset_overview.png"), "Sample cells from both classes and the spread of image sizes");
h2("7.3 Training, Validation and Test Split");
p("The images are divided with a stratified random split (seed 42), so each set has the same balance of classes. The test set is used only once, for the final scores. The twelve cells shown in the web application are always placed in the test set, so the demonstration only uses cells that the model has not seen.");
table("Data split", ["Set", "Parasitized", "Uninfected", "Total", "Used for"], [
  ["Training", n(D.train / 2), n(D.train / 2), n(D.train), "Learning the weights"],
  ["Validation", n(D.val / 2), n(D.val / 2), n(D.val), "Choosing the best epoch"],
  ["Test", n(S.parasitized.cells), n(S.uninfected.cells), n(D.test), "Final, unbiased evaluation"],
  ["Total", n((D.train + D.val + D.test) / 2), n((D.train + D.val + D.test) / 2), n(D.train + D.val + D.test), ""],
], [0.18, 0.17, 0.17, 0.16, 0.32]);

h1("8. IMAGE PROCESSING METHODOLOGY");
p("This chapter describes Stages 1 to 8. Figure 3 shows the overall workflow planned in the project synopsis, and Figures 4 and 5 show the actual output of every stage for an infected and a healthy test cell.");
figure(asset("project_workflow.png"), "Project workflow (from the project synopsis)");
figure(outFig("pipeline_parasitized.png"), "All pipeline stages on an infected test cell");
figure(outFig("pipeline_uninfected.png"), "All pipeline stages on a healthy test cell");
h2("8.1 Stage 1: Image Acquisition and Display (Module 1)");
p("Images are read with `cv2.imread`, which returns the colour channels in BGR order, and are converted to RGB with `cv2.cvtColor` so that they are displayed correctly by Matplotlib. Uploaded files are decoded from memory with `cv2.imdecode`. Result images are saved with `cv2.imwrite` and figures with Matplotlib.");
h2("8.2 Stage 2: Cropping and Resizing (Module 2)");
p("Each cell is surrounded by black padding. The bounding box of all pixels with a grey level above 10 is found with `cv2.findNonZero` and `cv2.boundingRect`, the image is cropped to that box, and the crop is resized to 128 × 128 pixels with area interpolation (`cv2.INTER_AREA`), which averages pixels and avoids aliasing when an image is reduced. A fixed size is required by the CNN and makes the parameters of the later stages comparable between images.");
h2("8.3 Stage 3: Colour-Space Conversion (Module 2)");
p("The grayscale image is used to separate the cell from the background. The HSV image separates the colour (hue H), the strength of the colour (saturation S) and the brightness (value V). OpenCV computes them as:");
eq("Y = 0.299 R + 0.587 G + 0.114 B");
eq("V = max(R, G, B),     S = (V − min(R, G, B)) / V");
p("The purple Giemsa stain on the parasite has a much higher saturation than the pale pink cytoplasm, so the S channel shows the parasite as a bright spot, more clearly than any single RGB channel. Stage 6 therefore thresholds the S channel to find stained pixels.");
h2("8.4 Stage 4: Histogram Equalisation with CLAHE (Module 3)");
p("Histogram equalisation maps every grey level r_{k} to a new level s_{k} using the cumulative histogram, where p(r_{j}) is the fraction of pixels with level r_{j} and L = 256:");
eq("s_{k} = (L − 1) · Σ_{j = 0}^{k} p(r_{j}),     k = 0, 1, … , L − 1");
p("This spreads the brightness values over the full range and gives slides with different illumination a similar contrast. It is applied only to the V channel, so the colour of the stain (H and S) is not changed. The first version of the project used global equalisation (`cv2.equalizeHist`), which stretched the darkest pixels of each cell almost to black and caused parts of the cell to be mistaken for background in Stage 6. The final version uses **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) [6]: the image is divided into 8 × 8 tiles, each tile is equalised separately, and each tile's histogram is clipped at a limit of 2.0 before equalisation, which caps the gain in contrast. This change also improved the accuracy of the CNN (Section 10.5).");
figure(outFig("histogram_equalisation.png"), "Brightness histogram of the cell before and after CLAHE");
h2("8.5 Stage 5: Noise Removal by Spatial Filtering (Module 3)");
p("Three spatial filters are implemented. The average filter replaces each pixel by the mean of its k × k neighbourhood, the Gaussian filter by a weighted mean whose weights follow a Gaussian function, and the median filter by the median of the neighbourhood. The median filter removes isolated noisy pixels without blurring edges, which keeps the boundary of the small parasite sharp, so a 3 × 3 median filter is used for training and by default. The other filters can be selected in the web application to compare their effect. The output of this stage is the input of the CNN.");
h2("8.6 Stage 6: Segmentation by Otsu Thresholding (Module 4)");
p("Otsu's method [5] chooses the threshold t that maximises the between-class variance, where ω_{0} and ω_{1} are the fractions of pixels below and above t and μ_{0} and μ_{1} are their mean values:");
eq("σ_{B}^{2}(t) = ω_{0}(t) · ω_{1}(t) · [ μ_{0}(t) − μ_{1}(t) ]^{2}");
p("The method needs no manual threshold and adapts to every image. It is applied twice: first on the grayscale image, to separate the bright cell from the black background, and then on the saturation channel, using only the pixels inside the cell, to separate stained pixels from normal cytoplasm. Because Otsu's method always divides the pixels into two groups, even in a healthy cell, a spot is accepted only if the mean saturation of the stained group exceeds the median saturation of the cell by at least 20 levels (the stain-contrast rule).");
h2("8.7 Stage 7: Morphological Operations (Module 2)");
p("Erosion with a structuring element B shrinks bright regions and deletes regions smaller than B; dilation grows them. Erosion followed by dilation (opening) removes isolated specks while keeping the size of the real regions, and dilation followed by erosion (closing) fills small gaps:");
eq("A ⊖ B = { z | (B)_{z} ⊆ A },     A ⊕ B = { z | (B)_{z} ∩ A ≠ ∅ }");
eq("A ∘ B = (A ⊖ B) ⊕ B  (opening),     A • B = (A ⊕ B) ⊖ B  (closing)");
p("Elliptical structuring elements are used. A 3 × 3 opening and a 7 × 7 closing clean the cell mask, the cell mask is then eroded with a 5 × 5 element so that the thin dark membrane at the edge of the cell is ignored, and a 3 × 3 opening cleans the stain mask.");
h2("8.8 Stage 8: Contour Detection (Module 5)");
p("`cv2.findContours` traces the outer boundaries of the white regions in a binary mask. The largest contour of the cell mask is the outline of the cell. A dark parasite on the edge of the cell can fall below the threshold in Stage 6 and leave a notch in this outline, so the outline is replaced by its convex hull (the smallest convex shape that contains it, like a rubber band stretched around the cell), after which the pure black padding is removed again. Contours of the stain mask with an area of at least 12 pixels are reported as stained spots and drawn in red, and the cell outline is drawn in green. During development, measured on 800 random cells, the convex-hull step raised the share of infected cells with an outlined spot from 64% to 89%.");
table("Parameters of the image-processing stages", ["Stage", "Parameter", "Value"], [
  ["2", "Output size", "128 × 128 pixels"],
  ["2", "Background cut-off", "Grey level 10"],
  ["4", "CLAHE clip limit and tile grid", "2.0 and 8 × 8 tiles"],
  ["5", "Filter and kernel", "Median, 3 × 3"],
  ["6", "Thresholds", "Otsu's method (automatic)"],
  ["6", "Minimum stain contrast", "20 saturation levels"],
  ["7", "Opening / closing of the cell mask", "3 × 3 / 7 × 7 ellipse"],
  ["7", "Erosion of the cell rim", "5 × 5 ellipse"],
  ["8", "Minimum spot area", "12 pixels"],
], [0.12, 0.48, 0.40]);

h1("9. CLASSIFICATION USING TRANSFER LEARNING");
h2("9.1 MobileNetV2");
p("MobileNetV2 [4] is a convolutional neural network designed for mobile devices. It is built from inverted residual blocks with depthwise separable convolutions, which keep the number of parameters and computations low (about 3.5 million parameters). The torchvision implementation pre-trained on ImageNet [9] (about 1.28 million photographs in 1,000 classes) is used. Its early layers already respond to edges, colours, blobs and textures, which are also the features that distinguish a stained parasite. The final 1,000-class layer is replaced by a new fully connected layer with two outputs, Uninfected and Parasitized.");
p(`Figure ${figureNo + 1} gives a simplified view of the network as used in this project. The 128 × 128 × 3 input image passes through 52 convolution layers (a first convolution, 17 inverted residual blocks and a final 1 × 1 convolution), global average pooling and the new output layer. Every node computes a weighted sum of its inputs plus a bias, z = Σ w_{i}x_{i} + b, and applies an activation function: ReLU6 in the hidden layers and Softmax in the output layer, which turns the two scores into the probabilities of Parasitized and Uninfected. After the output layer is replaced, the network has about 2.2 million parameters.`);
figure(asset("neural_network.png"), "Neural network architecture of the classifier, with one node shown in detail");
h2("9.2 Input Preparation");
p("The input of the network is the 128 × 128 RGB image produced by Stages 2 to 5. The pixel values are scaled to the range 0 to 1 and normalised with the ImageNet channel means (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225), as expected by the pre-trained network.");
h2("9.3 Data Augmentation (Module 2)");
p("During training every image is flipped horizontally and vertically at random and rotated by 0°, 90°, 180° or 270° (`cv2.flip`, `cv2.rotate`). A cell has no fixed orientation, so these operations create new valid examples and reduce over-fitting.");
h2("9.4 Two-Phase Training");
p(`Phase A trains only the new output layer (2,562 parameters) while all pre-trained layers are frozen. Phase B then unfreezes the last eight blocks of the network (1,987,074 trainable parameters) and fine-tunes them with a ten times smaller learning rate, so that the pre-trained features adapt to blood-cell images without being destroyed. Batch-normalisation layers of frozen blocks keep their ImageNet statistics. After every epoch the model is evaluated on the validation set, and the weights with the highest validation accuracy (${pct(M.best_val_acc)}) are kept.`);
table("Training settings", ["Setting", "Value"], [
  ["Input", "128 × 128 × 3 image after Stages 2 to 5"],
  ["Loss function", "Cross-entropy"],
  ["Optimiser", "Adam"],
  ["Batch size", "64"],
  ["Phase A", `Output layer only, learning rate 0.001, ${headEpochs} epochs`],
  ["Phase B", `Last 8 blocks + output layer, learning rate 0.0001, ${ftEpochs} epochs`],
  ["Augmentation", "Random horizontal and vertical flips, rotations by multiples of 90°"],
  ["Model selection", "Best validation accuracy"],
  ["Hardware and time", `Laptop CPU (Intel Core i5-11400H), ${M.train_minutes} minutes for all ${H.length} epochs`],
], [0.28, 0.72]);
h2("9.5 Model Export for Deployment");
p(`The trained network is exported to the ONNX format by \`export_onnx.py\` and run on the server with ONNX Runtime, which needs much less memory than PyTorch and fits the 512 MB free cloud instance. On all ${ONNX.samples_checked} demonstration cells the exported model gives the same probabilities as the PyTorch model; the largest difference is ${ONNX.max_probability_difference.toExponential(1)}, which is only rounding.`);

h1("10. RESULTS AND EVALUATION");
h2("10.1 Evaluation Metrics");
p("The positive class is Parasitized. With TP, TN, FP and FN the numbers of true positives, true negatives, false positives and false negatives, the metrics are:");
eq("Accuracy = (TP + TN) / (TP + TN + FP + FN)");
eq("Precision = TP / (TP + FP),     Recall = TP / (TP + FN)");
eq("F1-score = 2 · Precision · Recall / (Precision + Recall)");
p("Recall is the most important metric for screening, because every false negative is an infected cell that is missed. Precision shows how many of the cells reported as infected really are infected.");
h2("10.2 Test-Set Results");
p(`The final model was evaluated once on the ${n(D.test)} test images, which were not used for training or for choosing the model.`);
table("Results of the final model on the test set", ["Metric", "Value", "Meaning"], [
  ["Accuracy", pct(M.accuracy, 2), `${n(CM.TN + CM.TP)} of ${n(D.test)} cells classified correctly`],
  ["Precision", pct(M.precision, 2), "Share of cells reported as infected that are infected"],
  ["Recall (sensitivity)", pct(M.recall, 2), `${n(CM.TP)} of ${n(CM.TP + CM.FN)} infected cells detected`],
  ["Specificity", pct(CM.TN / (CM.TN + CM.FP), 2), `${n(CM.TN)} of ${n(CM.TN + CM.FP)} healthy cells recognised`],
  ["F1-score", pct(M.f1, 2), "Balance of precision and recall"],
], [0.25, 0.15, 0.60]);
h2("10.3 Confusion Matrix");
p(`Figure ${figureNo + 1} shows the confusion matrix. Of the ${n(CM.TP + CM.FN)} infected cells, ${n(CM.TP)} were detected and ${n(CM.FN)} were missed (${pct(CM.FN / (CM.TP + CM.FN))}). Of the ${n(CM.TN + CM.FP)} healthy cells, ${n(CM.TN)} were recognised and ${n(CM.FP)} were reported as infected (${pct(CM.FP / (CM.TN + CM.FP))}). The two kinds of error are of similar size, so the model is not biased towards either class.`);
figure(outFig("confusion_matrix.png"), "Confusion matrix of the final model on the test set", 3.6);
h2("10.4 Training Curves");
p(`Figure ${figureNo + 1} shows the loss and accuracy after each epoch. In Phase A, with only the output layer trained, the validation accuracy reaches ${pct(H[headEpochs - 1].val_acc)}. Fine-tuning in Phase B raises it to ${pct(M.best_val_acc)}. The training and validation curves stay close to each other, which shows that the model has not over-fitted the training images.`);
figure(outFig("training_curves.png"), "Training and validation loss and accuracy (dashed line: start of fine-tuning)");
h2("10.5 Improvement Across Versions");
p("The project was developed in three versions. The baseline used plain histogram equalisation. Version 0.1 replaced it with CLAHE and fine-tuned longer, which reduced the missed infections by more than half. The final version 1.0 was trained on the full dataset and evaluated on a test set more than four times larger.");
table("Comparison of the model versions", ["Version", "Training data", "Stage 4", "Test images", "Accuracy", "Recall", "F1", "Missed"], VERSIONS.map((v) => [
  v.name, v.data, v.enh, n(v.test), pct(v.acc), pct(v.rec), pct(v.f1), `${v.fn} (${pct(v.fn / v.pos)})`,
]), [0.12, 0.18, 0.14, 0.09, 0.13, 0.10, 0.09, 0.15]);
h2("10.6 Evaluation of Stained-Spot Detection (Stages 6 to 8)");
p("The classical stages were measured on the same test set to check whether the outlined spots are meaningful.");
table("Stained-spot detection on the test set", ["Measure", "Result"], [
  ["Test images", n(S.test_images)],
  ["Cells whose outline was found", pct(S.cell_outline_found_rate)],
  ["Infected cells with at least one outlined spot", `${n(S.parasitized.with_spot)} of ${n(S.parasitized.cells)} (${pct(S.spot_detection_rate_infected)})`],
  ["Healthy cells with a false spot", `${n(S.uninfected.with_spot)} of ${n(S.uninfected.cells)} (${pct(S.false_spot_rate_healthy)})`],
  ["Median stain contrast: infected / healthy", `${S.parasitized.median_stain_contrast.toFixed(0)} / ${S.uninfected.median_stain_contrast.toFixed(0)} saturation levels`],
], [0.55, 0.45]);
p(`Using only thresholding, morphology and contours, a stained spot is outlined in ${pct(S.spot_detection_rate_infected)} of infected cells, while only ${pct(S.false_spot_rate_healthy)} of healthy cells receive a false outline. The classical stages therefore point to the right region in most cases and make the decision of the CNN easy to check, while the CNN, with a recall of ${pct(M.recall)}, is the more reliable classifier.`);
h2("10.7 Sample Predictions and Error Analysis");
figure(outFig("sample_predictions.png"), "Sample test predictions (green: correct, red: wrong)");
p("Inspection of the wrongly classified cells shows four main causes:");
bullets([
  "**Faint or diffuse staining.** Some infected cells show only a pale, spread-out stain without a dark dot. Parasitized demonstration cell #1 is such a case; both the spot detector and the CNN classify it as uninfected.",
  "**Uneven staining of the whole cell.** When the rim of the cell is darker than its centre, a single Otsu threshold separates rim and centre instead of parasite and cytoplasm.",
  "**Debris and platelets.** Dark particles touching a healthy cell can look like a parasite and cause false positives.",
  "**Label noise.** A few cells labelled as parasitized show no visible parasite, which has also been noted in studies of this dataset [3].",
]);

h1("11. OUTPUT SCREENS");
p("The web application is available at https://malariavision.vercel.app. The following screens were captured from the final version.");
h2("11.1 Live Diagnosis Page");
p("The page opens on an infected demonstration cell. The left panel contains the gallery of test cells, the upload area and the pipeline settings; the right side shows the input, the outlined regions and the verdict.");
figure(shot("01_live_diagnosis.png"), "Live diagnosis page with the verdict for an infected cell");
h2("11.2 Step-by-Step Pipeline View");
figure(shot("02_pipeline_steps.png"), "The ten stage images with their syllabus modules");
h2("11.3 Histogram and Otsu Details");
figure(shot("03_histogram_otsu.png"), "Brightness histograms and the thresholds chosen by Otsu's method");
h2("11.4 Result for a Healthy Cell");
figure(shot("04_uninfected_verdict.png"), "Verdict for a healthy cell: no stained spot found");
h2("11.5 Pipeline Tuning");
p("Here the noise filter is changed to Gaussian with a 7 × 7 kernel. The images become smoother and the outlines change, which shows why a small median filter is used by default.");
figure(shot("05_tuning_gaussian7.png"), "Pipeline recomputed with a 7 × 7 Gaussian filter");
h2("11.6 Uploaded Image");
figure(shot("06_upload_result.png"), "Result for an uploaded image");
h2("11.7 Model Performance Page");
figure(shot("07_metrics.png"), "Metric cards compared with the first version");
figure(shot("08_confusion_curves.png"), "Confusion matrix and training curves in the web application");
h2("11.8 Syllabus Map");
figure(shot("09_syllabus_map.png"), "Syllabus map page");
h2("11.9 Mobile View");
figure(shot("10_mobile.png"), "The application on a phone screen", 2.6);

h1("12. TESTING AND VALIDATION");
h2("12.1 Automated API Tests");
p("The web API is tested automatically with pytest (`tests/test_api.py`). All eight tests pass with the final model.");
table("Automated tests", ["Test", "What is checked", "Result"], [
  ["test_health", "The server reports that it is ready", "Pass"],
  ["test_samples_listed", "Twelve samples of both classes are listed and their images load", "Pass"],
  ["test_analyze_sample (2 cases)", "An infected and a healthy sample get the right verdict, 10 step images and both histograms", "Pass"],
  ["test_server_matches_pytorch", "The server's probabilities equal the PyTorch model's on all 12 samples", "Pass"],
  ["test_upload", "An uploaded image is analysed with the chosen settings", "Pass"],
  ["test_rejects_bad_input", "Unknown sample, invalid filter or kernel size and a non-image file are rejected", "Pass"],
  ["test_metrics", "The metrics endpoint returns the test results", "Pass"],
], [0.30, 0.58, 0.12]);
h2("12.2 Model and Deployment Checks");
bullets([
  `The ONNX model gives the same probabilities as the PyTorch model on all ${ONNX.samples_checked} demonstration cells (largest difference ${ONNX.max_probability_difference.toExponential(1)}).`,
  "The server was installed in a clean Python environment with only `backend/requirements.txt` (without PyTorch) to reproduce the cloud installation; seven tests pass and the PyTorch comparison is skipped, as expected.",
  "The demonstration cells are forced into the test set by comparing file contents, and the training, validation and test sets were checked to have no image in common.",
  "On the live site the health check, sample analysis, image upload, metrics page and cross-origin access were checked, with no errors in the browser console.",
  "The web pages were checked at phone width (375 pixels) with no horizontal scrolling.",
]);

h1("13. SYLLABUS COVERAGE AND COURSE OUTCOME MAPPING");
h2("13.1 Syllabus Coverage");
table("Syllabus topics used in the project", ["Module", "Syllabus topic", "Use in the project", "Code"], [
  ["M1", "Image acquisition, loading, displaying and saving; Python, OpenCV, NumPy, Matplotlib", "Stage 1 and all figures", "load_image, make_figures.py"],
  ["M2", "Image resizing and cropping", "Stage 2: standard 128 × 128 size", "crop_and_resize"],
  ["M2", "Colour-space conversion (RGB, HSV, Grayscale)", "Stage 3", "to_gray, to_hsv"],
  ["M2", "Image morphology", "Stage 7: erosion, dilation, opening, closing", "clean_mask, fill_holes"],
  ["M2", "Image rotation and flipping", "Data augmentation during training", "augment"],
  ["M3", "Histogram-based enhancement", "Stage 4: CLAHE", "enhance_contrast"],
  ["M3", "Basic image filtering (Average, Gaussian, Median)", "Stage 5", "denoise"],
  ["M4", "Thresholding", "Stage 6: Otsu's method", "otsu_threshold, otsu_inside"],
  ["M5", "Contour detection", "Stage 8: cell outline and stained spots", "find_contours"],
  ["M5", "Transfer learning with pre-trained CNN models", "Stage 9: MobileNetV2", "build_model, train.py"],
], [0.12, 0.36, 0.28, 0.24]);
p("Topics of Module 6 (U-Net, Segment Anything Model and Vision Transformers) are outside the scope of this project, as planned in the synopsis; they are listed as future work.");
h2("13.2 Course Outcome Mapping");
table("Course outcomes and the project's contribution", ["CO", "Course outcome", "Contribution of the project"], [
  ["CO1", "Perform image acquisition, image manipulation, and basic image processing operations using Python", "Image loading, display and saving, cropping, resizing, colour conversion, flipping and rotation"],
  ["CO2", "Execute image enhancement, image filtering, edge detection, and feature extraction techniques", "CLAHE, median, Gaussian and average filters, Otsu thresholding, morphology and contour features"],
  ["CO3", "Investigate the performance of object detection, identification and recognition using ML and DL models", "Transfer learning with MobileNetV2, studied with metrics, a confusion matrix, three model versions and an error analysis"],
  ["CO4", "Assess and evaluate image processing and computer vision solutions for real-world applications", `A healthcare application built end to end, evaluated on ${n(D.test)} unseen images and deployed online`],
], [0.08, 0.44, 0.48]);

h1("14. INSTALLATION AND EXECUTION");
h2("14.1 Using the Live Application");
p("The application runs online and needs only a web browser: https://malariavision.vercel.app (web application) and https://malariavision-api.onrender.com/docs (API documentation). The free server sleeps after 15 minutes without visitors, so the first visit can take up to a minute while it starts; the page shows a waiting message during this time.");
h2("14.2 Local Setup");
code([
  "git clone https://github.com/Shiva-Preetham/Image-Processing-CV-PBL.git",
  "cd Image-Processing-CV-PBL",
  "python -m venv .venv",
  ".venv\\Scripts\\activate          (Linux / macOS: source .venv/bin/activate)",
  "pip install -r requirements.txt",
]);
h2("14.3 Running the Web Application Locally");
p("The trained model and the demonstration cells are included in the repository. The API is started in one terminal and the React application in a second terminal, and the application is then opened at http://localhost:5173:");
code([
  "uvicorn backend.main:app --port 8000",
  "",
  "cd frontend",
  "npm install",
  "npm run dev",
]);
p("Without Node.js, the offline demonstration is started with `streamlit run app.py`.");
h2("14.4 Training and Evaluation");
p("The dataset (337 MB) is downloaded from the National Library of Medicine and extracted into the `data` folder. The remaining commands train the model, evaluate it, export it and recreate all figures:");
code([
  "Invoke-WebRequest https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip -OutFile data/cell_images.zip",
  "Expand-Archive data/cell_images.zip -DestinationPath data",
  "",
  "python train.py                    # full dataset, about 40 minutes on a laptop CPU",
  "python evaluate_segmentation.py    # Stages 6-8 on the test set",
  "python export_onnx.py              # ONNX model for the server",
  "python make_figures.py             # report figures",
  "python -m pytest tests -q          # API tests",
]);
h2("14.5 Deployment");
numbered([
  "**Render (API):** New, Blueprint, select the GitHub repository and apply. Render reads `render.yaml`, installs `backend/requirements.txt` and starts the server.",
  "**Vercel (web application):** Add New, Project, import the repository, set the root directory to `frontend`, keep the Vite preset and deploy.",
  "Both services deploy again automatically after every push to the main branch.",
]);

h1("15. CONCLUSION");
p(`MalariaVision shows that the techniques of the Image Processing and Computer Vision Lab can be combined into a complete and useful healthcare application. Classical stages (cropping, colour-space conversion, CLAHE, median filtering, Otsu thresholding, morphology and contour detection) clean every cell and outline the stained parasite, and a MobileNetV2 network adapted by transfer learning classifies the cell. On ${n(D.test)} test images that were never used for training, the final model reaches an accuracy of ${pct(M.accuracy)}, a precision of ${pct(M.precision)}, a recall of ${pct(M.recall)} and an F1-score of ${pct(M.f1)}.`);
p(`The two approaches complement each other. The CNN gives the more accurate decision, while the classical stages outline a stained spot in ${pct(S.spot_detection_rate_infected)} of infected cells and in only ${pct(S.false_spot_rate_healthy)} of healthy cells, which shows the user where to look and makes the result easy to verify. Measuring each change also showed its value: replacing plain histogram equalisation with CLAHE and fine-tuning longer more than halved the missed infections of the first version.`);
p("The system is reproducible and openly available: the code, the trained model, the results and this report are generated from scripts in the GitHub repository, the web API is covered by automated tests, and the application is deployed online, where every stage can be inspected for any cell.");
h2("15.1 Learning Outcomes");
bullets([
  "Applying image enhancement, filtering, thresholding, morphology and contour detection to real microscope images and understanding the effect of each parameter.",
  "Using transfer learning to train an accurate classifier on a laptop CPU in less than an hour.",
  "Evaluating a model correctly with a held-out test set, and choosing metrics such as recall that suit a medical screening task.",
  "Finding and fixing problems by measurement, for example the convex-hull fix in Stage 8 and the change to CLAHE in Stage 4.",
  "Building, testing and deploying a complete web application around a computer vision model.",
]);

h1("16. LIMITATIONS AND FUTURE SCOPE");
h2("16.1 Limitations");
bullets([
  "The system works on images of single cells. A photograph of a whole blood smear would first need its cells to be separated.",
  "The model was trained on one dataset; other staining protocols, microscopes or cameras may need re-training or colour normalisation.",
  "A single Otsu threshold can split an unevenly stained cell into rim and centre and miss the parasite (Section 10.7).",
  "Some labels in the dataset are noisy, which limits the accuracy that can be measured.",
  "The free cloud server sleeps after 15 minutes without visitors, so the first request can take up to a minute.",
  "The system is an educational project and not a certified medical device; its results must be confirmed by a trained person.",
]);
h2("16.2 Future Scope");
numbered([
  "Adaptive thresholding (Module 4) as an alternative to Otsu's method for unevenly stained cells.",
  "Detection of individual cells in whole smear images with contour-based segmentation or YOLO (Module 5), followed by per-cell classification and an estimate of the parasite density.",
  "Comparison with other pre-trained networks such as ResNet18, and training on a GPU.",
  "Pixel-level segmentation of parasites with U-Net or the Segment Anything Model (Module 6).",
  "Identification of the parasite species and of its life-cycle stage.",
  "A mobile version that works with a smartphone attached to a microscope.",
]);

h1("17. REFERENCES");
list([
  "World Health Organization, *World Malaria Report 2024*, Geneva: WHO, 2024.",
  "U.S. National Library of Medicine, Lister Hill National Center for Biomedical Communications, Malaria Cell Images Dataset, https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip",
  "S. Rajaraman, S. K. Antani, M. Poostchi, K. Silamut, M. A. Hossain, R. J. Maude, S. Jaeger and G. R. Thoma, \"Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images,\" *PeerJ*, vol. 6, e4568, 2018.",
  "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov and L.-C. Chen, \"MobileNetV2: Inverted Residuals and Linear Bottlenecks,\" in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 4510-4520.",
  "N. Otsu, \"A Threshold Selection Method from Gray-Level Histograms,\" *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62-66, 1979.",
  "K. Zuiderveld, \"Contrast Limited Adaptive Histogram Equalization,\" in *Graphics Gems IV*, Academic Press, 1994, pp. 474-485.",
  "R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018.",
  "R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., Springer, 2022.",
  "J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, \"ImageNet: A Large-Scale Hierarchical Image Database,\" in *Proc. IEEE CVPR*, 2009.",
  "OpenCV Documentation, https://docs.opencv.org/",
  "PyTorch and torchvision Documentation, https://pytorch.org/docs/",
  "ONNX Runtime Documentation, https://onnxruntime.ai/docs/",
  "FastAPI Documentation, https://fastapi.tiangolo.com/",
  "React Documentation, https://react.dev/",
  "scikit-learn Documentation, Metrics and scoring, https://scikit-learn.org/stable/modules/model_evaluation.html",
  "Syllabus of the Fundamental of Image Processing and Computer Vision Lab (26B16CS316), and the MalariaVision project synopsis, JIIT Noida, 2026-27.",
], true, AlignmentType.LEFT);

// ================================================================== FRONT MATTER
const center = (text, size, bold = true, extra = {}) => new Paragraph({ alignment: AlignmentType.CENTER,
  spacing: { after: 60 }, ...extra, children: runs(text, { font: FONT, size, bold }) });
const left = (text, size, opts = {}, runOpts = {}) => new Paragraph({ indent: { left: 1560 }, spacing: { after: 40 }, ...opts,
  children: [new TextRun({ text, font: FONT, size, ...runOpts })] });

const logo = asset("jiit_logo.png");
const lg = pngSize(logo);
const titlePage = [
  new Paragraph({ spacing: { before: 600 }, children: [] }),
  center("BACHELOR OF TECHNOLOGY", 26),
  center("Computer Science Engineering", 26),
  center("PROJECT-BASED LEARNING", 26),
  center("PROJECT REPORT", 26),
  center("IMAGE PROCESSING AND COMPUTER VISION LAB", 26),
  center("(26B16CS316)", 26, true, { spacing: { after: 300 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [
    new TextRun({ text: "PROJECT TITLE: - MalariaVision : ", font: FONT, size: 26, bold: true }),
    new TextRun({ text: "Malaria Parasite Detection from Blood Smear Images Using Image Processing", font: FONT, size: 23, bold: true }),
  ] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 }, children: [
    new TextRun({ text: "Academic Year: ", font: FONT, size: 24, bold: true }),
    new TextRun({ text: "2026-27", font: FONT, size: 24 }),
  ] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 }, children: [
    new ImageRun({ type: "png", data: fs.readFileSync(logo), transformation: { width: 150, height: Math.round(150 * lg.h / lg.w) } }),
  ] }),
  left("Group Members:", 24, { spacing: { after: 80 } }, { bold: true, underline: {} }),
  left("Lakshita Gautam 23103343", 24),
  left("Ranjeet Kaur 2401030065", 24),
  left("Navya Banga 2401030070", 24),
  left("Shiva Preetham Chelimila 2401030078", 24, { spacing: { after: 360 } }),
  left("Submitted To:", 24, { spacing: { after: 60 } }, { bold: true }),
  left("Ms. Neha", 22, {}, { bold: true }),
  left("Dr. Pawan Kumar Upadhyay", 22, {}, { bold: true }),
];

const dotTabs = [{ type: TabStopType.RIGHT, position: TEXT_W, leader: "dot" }];
const page = (key, fallback = "0") => String(PAGES[key] ?? fallback);
const contents = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
    children: [new TextRun({ text: "CONTENTS", font: FONT, size: 28, bold: true })] }),
  new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W }], spacing: { after: 120 },
    children: [new TextRun({ text: "Chapter / Section", font: FONT, size: 22, bold: true }),
      new TextRun({ children: [new Tab(), "Page No."], font: FONT, size: 22, bold: true })] }),
  new Paragraph({ tabStops: dotTabs, spacing: { after: 60 },
    children: [new TextRun({ text: "List of Figures", font: FONT, size: 22, bold: true }),
      new TextRun({ children: [new Tab(), page("__lof__", "iii")], font: FONT, size: 22, bold: true })] }),
  ...tocEntries.map((e) => new Paragraph({
    tabStops: dotTabs, spacing: { before: e.level === 1 ? 80 : 0, after: e.level === 1 ? 30 : 15 },
    indent: { left: e.level === 1 ? 0 : 360 },
    children: [new TextRun({ text: e.text, font: FONT, size: e.level === 1 ? 22 : 21, bold: e.level === 1 }),
      new TextRun({ children: [new Tab(), page(e.text)], font: FONT, size: e.level === 1 ? 22 : 21, bold: e.level === 1 })],
  })),
];
const lof = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, pageBreakBefore: true,
    children: [new TextRun({ text: "LIST OF FIGURES", font: FONT, size: 28, bold: true })] }),
  ...figures.map((f) => new Paragraph({ tabStops: dotTabs, spacing: { after: 50 },
    children: [new TextRun({ text: `Figure ${f.num}: ${f.caption}`, font: FONT, size: 22 }),
      new TextRun({ children: [new Tab(), page(`Figure ${f.num}:`)], font: FONT, size: 22 })] })),
];

// ================================================================== DOCUMENT
const edge = { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 24 };
const pageProps = (pageNumbers) => ({
  page: {
    size: { width: 11906, height: 16838 },
    margin: { top: 1300, bottom: 1300, left: 1440, right: 1440, footer: 620 },
    borders: {
      pageBorders: { display: "allPages", offsetFrom: "page", zOrder: "front" },
      pageBorderTop: edge, pageBorderBottom: edge, pageBorderLeft: edge, pageBorderRight: edge,
    },
    ...(pageNumbers ? { pageNumbers } : {}),
  },
});
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] });

const doc = new Document({
  creator: "MalariaVision project group",
  title: "MalariaVision: Malaria Parasite Detection from Blood Smear Images - Project Report",
  styles: {
    default: {
      document: { run: { font: FONT, size: 24 } },
      heading1: { run: { font: FONT, size: 28, bold: true, color: "000000" },
        paragraph: { spacing: { before: 0, after: 220 }, keepNext: true } },
      heading2: { run: { font: FONT, size: 24, bold: true, color: "000000" },
        paragraph: { spacing: { before: 220, after: 100 }, keepNext: true } },
    },
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
  ] },
  sections: [
    { properties: pageProps(), children: titlePage },
    { properties: pageProps({ start: 1, formatType: NumberFormat.LOWER_ROMAN }), footers: { default: footer },
      children: [...contents, ...lof] },
    { properties: pageProps({ start: 1, formatType: NumberFormat.DECIMAL }), footers: { default: footer },
      children: body },
  ],
});

const outFile = path.join(HERE, "MalariaVision_Project_Report.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outFile, buf);
  fs.writeFileSync(path.join(HERE, "toc_entries.json"), JSON.stringify({
    headings: tocEntries.map((e) => e.text), figures: figures.map((f) => `Figure ${f.num}:`),
  }, null, 2));
  console.log(`Wrote ${outFile} (${tocEntries.length} headings, ${figures.length} figures, ${tableNo} tables)`);
});
