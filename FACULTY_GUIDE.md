# Presenting MalariaVision to the Lab Faculty

A 5-minute talk track for the prototype review, written in plain words.
Read it once, then present from the live demo (`streamlit run app.py`).

---

## 1. The one-line pitch

> "We built a system that looks at a single red blood cell under the microscope and tells you whether a malaria parasite is inside it. Classical image processing cleans the image and points to the suspicious spot, then a pre-trained CNN makes the final call."

## 2. The problem (30 seconds)

- Malaria is confirmed by a technician looking at a stained blood smear under a microscope and checking cell after cell.
- That is slow and tiring, and a tired eye can miss an infected cell.
- The parasite shows up as a **dark purple dot** inside the cell, because the Giemsa stain colours it purple.
- Our goal: let the computer do the first check automatically and show **where** it saw the parasite.

## 3. The idea in one picture

```
 Load  ->  Clean up  ->  Find the cell & the dark spot  ->  CNN decides  ->  Measure how good it is
 (M1)      (M2, M3)          (M4, M2, M5)                     (M5)            (CO4)
```

## 4. Walking through the 10 stages (what to say)

| # | Stage | What we say | Module |
|---|-------|-------------|--------|
| 1 | Load the image | "We read the microscope image with OpenCV and show it with Matplotlib." | M1 |
| 2 | Crop + resize | "Images come in different sizes, so we cut off the empty black border and resize every cell to 128 x 128." | M2 |
| 3 | Colour spaces | "Grayscale tells us brightness, so we use it to find the cell. HSV separates colour strength (saturation), and the purple stain glows brightly there." | M2 |
| 4 | Histogram equalisation (CLAHE) | "Some slides are darker than others. Equalising the brightness histogram spreads the values out, so every image gets similar contrast. We use the contrast-limited version, CLAHE, which works on small tiles and caps how much contrast it adds." | M3 |
| 5 | Median filter | "A median filter removes tiny specks of noise but keeps edges sharp. The Gaussian and average filters are also selectable in the demo." | M3 |
| 6 | Otsu thresholding | "Otsu chooses the threshold automatically from the histogram. We use it twice: first to separate the cell from the background, then to separate stained pixels from normal ones." | M4 |
| 7 | Erosion + dilation | "Thresholding leaves small noisy dots. Erosion removes them and dilation grows the real regions back to size." | M2 |
| 8 | Contours | "We trace the outlines: green for the cell, red for any stained spot. This shows the doctor **where** to look. A dark parasite on the cell's edge used to be cut out of the cell outline, so we wrap the outline in its convex hull, like a rubber band, to keep the parasite inside." | M5 |
| 9 | Transfer learning | "MobileNetV2 already learned edges and textures from 1.2 million photos. We keep that knowledge and retrain only the last layers to answer one question: infected or not. We also flip and rotate the training images (Module 2), because a cell looks the same upside-down." | M5 |
| 10 | Evaluation | "We test on images the model never saw and report accuracy, precision, recall, F1 and a confusion matrix." | CO4 |

## 5. Results

_Filled in after training: see section below._

## 6. Live demo script (2 minutes)

1. Open the **Live diagnosis** tab and pick **Parasitized #1**.
   Point at the red outline: "Stages 6 to 8 found the stained spot." Then point at the verdict card: "The CNN agrees: Parasitized, with this confidence."
2. Switch to **Uninfected #1**: no red spot, and a green verdict.
3. Scroll through the 10 step images: "Each picture is one syllabus topic, labelled with its module."
4. In the sidebar, change the filter from **Median** to **Gaussian** and raise the kernel size to 7. The images get blurrier and the spot outline changes. "This is why we chose a small median filter."
5. Open **Model performance**: read out recall and the number of missed infections.
6. Open **Syllabus map**: "Every stage maps to Modules 1 to 5."

## 7. Questions the faculty may ask

- **Why HSV and not RGB?** In RGB, brightness and colour are mixed into all three channels. HSV keeps colour strength (S) separate from brightness (V), so the purple stain stands out in one channel.
- **Why CLAHE and not plain histogram equalisation?** We tried plain `equalizeHist` first. It stretched each cell's darkest pixels to almost black, so Otsu then mistook parts of the cell for background. CLAHE equalises small tiles and limits the contrast boost, which keeps the cell intact. Both are histogram-based enhancement (Module 3).
- **Why Otsu?** It picks the threshold from each image's own histogram, so it adapts to darker or lighter slides instead of using one fixed number.
- **Why median and not Gaussian?** The median removes isolated noisy pixels without blurring the edge of the parasite. Gaussian smooths everything, including the edges we need.
- **Why transfer learning and not a CNN from scratch?** A CNN trained from scratch needs far more images and time. A pre-trained network already knows basic shapes and textures, so it learns our task quickly and accurately on a laptop.
- **Why not just accuracy?** In diagnosis, missing an infected cell (false negative) is worse than a false alarm, so recall and the confusion matrix matter more than accuracy alone.
- **What is the role of Stages 6-8 if the CNN decides?** They make the system explainable: the red contour shows where the suspicious spot is, so a human can check the CNN's answer.
- **How did you tune Stages 6-8?** We measured them on 800 random cells. The first version highlighted a spot in only 64% of infected cells, because the dark parasite was cut out of the cell outline. After the convex-hull fix it is 89%, while only 4% of healthy cells get a false highlight.
- **Limitations?** The dataset has one cell per image. A full slide would first need the cells to be separated (more segmentation). When the whole cell is unevenly stained, a single Otsu threshold splits the cell's rim from its centre and misses the parasite; adaptive thresholding (also Module 4) is our planned fix. A few dataset images labelled "Parasitized" show no visible parasite, so some label noise exists. Different stains or microscopes may need re-training.

## 8. What we want feedback on

Ask the faculty these questions, so the next version follows their guidance:

1. Is the 10-stage pipeline the right depth, or should one stage be shown in more detail (for example, comparing Average vs. Gaussian vs. Median with numbers)?
2. Should we train on the full 27,558 images for the final version (the prototype uses a subset to train quickly on a laptop)?
3. Do you prefer a Jupyter notebook submission alongside the demo app?
4. Should the stained-spot detection from Stages 6-8 also be measured on the test set?
5. For unevenly stained cells, should we add adaptive thresholding next to Otsu (both are Module 4)?
