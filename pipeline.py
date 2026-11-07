"""
MalariaVision - classical image-processing pipeline (Stages 1 to 8).

Each function is labelled with the syllabus module it comes from, so this
file is also the "syllabus map" of the project:

    Stage 1  Load / display / save image ........ Module 1
    Stage 2  Crop + resize ........................ Module 2
    Stage 3  Colour spaces (RGB, Gray, HSV) ....... Module 2
    Stage 4  Histogram equalisation ............... Module 3
    Stage 5  Noise removal (Median / Gaussian) .... Module 3
    Stage 6  Otsu thresholding .................... Module 4
    Stage 7  Morphology (erosion + dilation) ...... Module 2
    Stage 8  Contour detection .................... Module 5

Stages 2-5 produce the cleaned image that the CNN classifier sees.
Stages 6-8 find the cell and highlight the dark-stained (parasite-like)
spots inside it, so a human can see WHERE the suspicious region is.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

IMG_SIZE = 128  # every image is standardised to 128 x 128 pixels


# --------------------------------------------------------------------------
# Stage 1 (Module 1): image acquisition - loading and saving
# --------------------------------------------------------------------------
def load_image(path):
    """Read an image file. OpenCV reads colours as BGR, so convert to RGB."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_image(file_bytes):
    """Same as load_image, but for an uploaded file held in memory."""
    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("The uploaded file is not a readable image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------
# Stage 2 (Module 2): cropping and resizing
# --------------------------------------------------------------------------
def crop_and_resize(rgb, size=IMG_SIZE):
    """Crop away the empty black border around the cell, then resize so that
    every image has the same size (the CNN needs a fixed input size)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    non_black = cv2.findNonZero((gray > 10).astype(np.uint8))
    if non_black is not None:
        x, y, w, h = cv2.boundingRect(non_black)
        rgb = rgb[y:y + h, x:x + w]
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# Stage 3 (Module 2): colour-space conversion
# --------------------------------------------------------------------------
def to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def to_hsv(rgb):
    """HSV = Hue (colour), Saturation (how strong the colour is), Value
    (brightness). The purple Giemsa stain on parasites shows up as strong
    saturation, which is easier to separate than in plain RGB."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


# --------------------------------------------------------------------------
# Stage 4 (Module 3): histogram-based enhancement
# --------------------------------------------------------------------------
def enhance_contrast(rgb):
    """Histogram equalisation on the V (brightness) channel only.
    Stretching V spreads out the brightness levels (better contrast) while
    H and S are untouched, so the stain colour is preserved."""
    h, s, v = cv2.split(to_hsv(rgb))
    v = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)


# --------------------------------------------------------------------------
# Stage 5 (Module 3): spatial filtering for noise removal
# --------------------------------------------------------------------------
FILTERS = ("Median", "Gaussian", "Average")


def denoise(rgb, method="Median", ksize=3):
    """Median: replaces each pixel by the middle value of its neighbours
    (removes specks, keeps edges sharp). Gaussian / Average: weighted /
    plain mean of the neighbours (smooths, but blurs edges more)."""
    if method == "Median":
        return cv2.medianBlur(rgb, ksize)
    if method == "Gaussian":
        return cv2.GaussianBlur(rgb, (ksize, ksize), 0)
    if method == "Average":
        return cv2.blur(rgb, (ksize, ksize))
    raise ValueError(f"Unknown filter: {method}")


def preprocess(rgb, filter_method="Median", ksize=3):
    """Stages 2 -> 5. This cleaned 128x128 RGB image is the CNN's input."""
    return denoise(enhance_contrast(crop_and_resize(rgb)), filter_method, ksize)


# --------------------------------------------------------------------------
# Stage 6 (Module 4): Otsu thresholding
# --------------------------------------------------------------------------
def otsu_threshold(gray):
    """Otsu picks the threshold automatically from the image histogram:
    the value that best splits pixels into two groups."""
    t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t, mask


def otsu_inside(channel, region_mask):
    """Otsu threshold computed only from the pixels inside region_mask."""
    pixels = channel[region_mask > 0].reshape(-1, 1)
    if pixels.size < 2:
        return 255.0
    t, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


# --------------------------------------------------------------------------
# Stage 7 (Module 2): morphological operations
# --------------------------------------------------------------------------
def clean_mask(mask, ksize=3):
    """Erosion then dilation (= 'opening'): erosion deletes tiny noise specks,
    dilation grows the surviving regions back to their original size."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.erode(mask, kernel, iterations=1)
    return cv2.dilate(mask, kernel, iterations=1)


def fill_holes(mask, ksize=5):
    """Dilation then erosion (= 'closing'): fills small gaps inside a region."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.erode(mask, kernel, iterations=1)


# --------------------------------------------------------------------------
# Stage 8 (Module 5): contour detection
# --------------------------------------------------------------------------
def find_contours(mask, min_area=0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


# --------------------------------------------------------------------------
# Full analysis: runs Stages 1-8 and keeps every intermediate image
# --------------------------------------------------------------------------
# Stain-spot settings (for a 128 x 128 cell image)
RIM_ERODE = 9          # ignore the dark cell membrane at the edge
MIN_SPOT_AREA = 12     # px: smaller blobs are treated as noise
MIN_STAIN_GAP = 35     # spot must be clearly more stained than the cell body


@dataclass
class PipelineResult:
    steps: dict = field(default_factory=dict)  # name -> image
    cell_contour: object = None
    spot_contours: list = field(default_factory=list)
    otsu_cell: float = 0.0
    otsu_stain: float = 0.0
    stain_gap: float = 0.0

    @property
    def cnn_input(self):
        return self.steps["5_filtered"]

    @property
    def spot_area(self):
        return int(sum(cv2.contourArea(c) for c in self.spot_contours))


def run_pipeline(rgb, filter_method="Median", ksize=3, morph_ksize=3):
    r = PipelineResult()
    r.steps["1_original"] = rgb

    resized = crop_and_resize(rgb)
    r.steps["2_resized"] = resized

    r.steps["3_gray"] = to_gray(resized)
    r.steps["3_saturation"] = to_hsv(resized)[:, :, 1]

    enhanced = enhance_contrast(resized)
    r.steps["4_enhanced"] = enhanced

    filtered = denoise(enhanced, filter_method, ksize)
    r.steps["5_filtered"] = filtered

    # Stage 6a - Otsu on grayscale: bright cell vs. black background
    gray_f = to_gray(filtered)
    r.otsu_cell, cell_mask = otsu_threshold(gray_f)
    r.steps["6_cell_mask_raw"] = cell_mask

    # Stage 7a - morphology to clean the cell mask, keep the largest blob
    cell_mask = fill_holes(clean_mask(cell_mask, morph_ksize), 7)
    cells = find_contours(cell_mask)
    solid_cell = np.zeros_like(cell_mask)
    if cells:
        r.cell_contour = max(cells, key=cv2.contourArea)
        cv2.drawContours(solid_cell, [r.cell_contour], -1, 255, thickness=cv2.FILLED)
    r.steps["7_cell_mask"] = solid_cell

    # Stage 6b - Otsu on the Saturation channel, inside the cell only
    rim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RIM_ERODE, RIM_ERODE))
    inner = cv2.erode(solid_cell, rim)
    saturation = to_hsv(filtered)[:, :, 1]
    r.steps["6_saturation"] = saturation
    r.otsu_stain = otsu_inside(saturation, inner)
    stain_mask = ((saturation > r.otsu_stain) & (inner > 0)).astype(np.uint8) * 255
    r.steps["6_stain_mask_raw"] = stain_mask

    # Is the "stained" group really darker-stained than the rest of the cell?
    inside = saturation[inner > 0].astype(np.float32)
    if inside.size and (inside > r.otsu_stain).any():
        r.stain_gap = float(inside[inside > r.otsu_stain].mean() - np.median(inside))

    # Stage 7b - morphology to remove specks from the stain mask
    stain_mask = clean_mask(stain_mask, morph_ksize)
    r.steps["7_stain_mask"] = stain_mask
    both = (solid_cell // 255 * 90).astype(np.uint8)
    both[stain_mask > 0] = 255
    r.steps["7_masks"] = both  # grey = cell, white = stained spots

    # Stage 8 - contours of the remaining stained spots
    if r.stain_gap >= MIN_STAIN_GAP:
        r.spot_contours = find_contours(stain_mask, MIN_SPOT_AREA)

    overlay = resized.copy()
    if r.cell_contour is not None:
        cv2.drawContours(overlay, [r.cell_contour], -1, (40, 200, 70), 1)
    cv2.drawContours(overlay, r.spot_contours, -1, (255, 30, 30), 2)
    r.steps["8_contours"] = overlay
    return r


# The steps shown in the report figures and the demo app:
# (image key, title, syllabus module, one-line plain-English explanation)
STEPS = [
    ("1_original", "1. Original", "Module 1", "The cell image as loaded from the dataset."),
    ("2_resized", "2. Crop + resize", "Module 2", "Black border cropped, resized to 128 x 128."),
    ("3_gray", "3a. Grayscale", "Module 2", "Brightness only - used to find the cell."),
    ("3_saturation", "3b. HSV: saturation", "Module 2", "Colour strength - purple stain glows here."),
    ("4_enhanced", "4. Histogram equalised", "Module 3", "Brightness spread out = better contrast."),
    ("5_filtered", "5. Noise filter", "Module 3", "Small specks of noise smoothed away."),
    ("6_cell_mask_raw", "6a. Otsu: cell", "Module 4", "Auto threshold: cell vs. black background."),
    ("6_stain_mask_raw", "6b. Otsu: stain", "Module 4", "Auto threshold: stained vs. normal pixels."),
    ("7_masks", "7. Erosion + dilation", "Module 2", "Specks removed. Grey = cell, white = stain."),
    ("8_contours", "8. Contours", "Module 5", "Green = cell outline, red = stained spots."),
]
