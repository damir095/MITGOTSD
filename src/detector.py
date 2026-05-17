"""
Traffic-sign region detector (OpenCV, no ML).

Design philosophy: **recall-first**.  This detector feeds a two-stage pipeline
(`video_pipeline.py`) that already rejects junk hard downstream — confidence
threshold 0.92, entropy gate, NMS and 6-frame track persistence.  So the
detector's job is to *not miss signs*; deciding "is this actually a sign" is
the classifier's job.  Gates here are deliberately loose compared to a
single-stage detector.

Three complementary candidate sources, merged + de-duplicated:

1. Colour path  — saturated red / blue / yellow blobs (relaxed S/V floors so
   distant, faded or poorly-lit signs still register).  Catches prohibitory,
   mandatory, warning-triangle and priority-diamond signs.
2. Circle path  — Hough circles on the CLAHE-equalised grayscale.  Votes on
   *partial / weak* edge support, so it recovers (a) distant faint red rings
   the colour mask fragments, and (b) **achromatic** round signs that have no
   saturated colour at all — GTSRB classes 6, 32, 41, 42 (white disk, thin
   black ring + diagonal stroke).  Pure colour detection can never see these.
3. Polygon path — edge-based contour approximation for triangles / diamond /
   octagon, independent of colour saturation (helps the same distance/light
   cases as #2 for non-round signs).

Every candidate then passes a *lenient* ROI verification (`_verify_roi`) that
only kills obvious non-signs (flat sky / wall: bright but no dark structure
and no edges).  Each ROI is (x, y, w, h) in input-frame pixel coordinates.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# HSV colour ranges.  Hue in [0, 180] (OpenCV convention).
# S/V floors were lowered to S>=70 for recall-first, but that put skin tone and
# pale/beige fabric (low-hue, S≈70-120, bright) squarely inside the red & yellow
# masks → heavy false positives indoors.  Raised back to S>=110 (red) / S>=140
# (yellow): real sign colour is highly saturated, skin/beige is not.  Costs some
# recall on very distant/shaded signs; acceptable — downstream still gates.
# ---------------------------------------------------------------------------
_COLOUR_RANGES = [
    # Red — vivid red borders (speed-limit, prohibition, warning triangles).
    (np.array([0,   110,  70]), np.array([10,  255, 255])),
    (np.array([168, 110,  70]), np.array([180, 255, 255])),
    # Blue — mandatory / regulatory signs (round, solid blue fill).
    # Skin/beige is never blue, so its floor stays low (no FP cost here).
    (np.array([95,   70,  55]), np.array([130, 255, 255])),
    # Yellow — warning triangles and "Главная дорога" diamond.
    # Narrowed hue + raised S/V: beige curtains/skin highlights live at the
    # low-S edge of this band and were the main indoor false trigger.
    (np.array([20, 140, 130]), np.array([32,  255, 255])),
]

# Larger close kernel + more iterations than the old 3×3/2: at distance the
# red ring is a thin broken arc; we *want* to bridge it into one blob even at
# the cost of occasionally merging neighbours (downstream NMS/classifier sorts
# it out).  Open kernel stays small to drop speckle without eroding far signs.
_CLOSE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_OPEN_K  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def _apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Enhance local contrast via CLAHE on the LAB L-channel."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _build_mask(hsv: np.ndarray) -> np.ndarray:
    """Saturated-colour mask (red ∪ blue ∪ yellow).  Used by the debug overlay."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _COLOUR_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    # Close bridges the thin/broken ring of distant signs into a solid blob;
    # open removes isolated noise pixels.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE_K, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _OPEN_K,  iterations=1)
    return mask


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with median-based auto thresholds (robust to global lighting)."""
    v = float(np.median(gray))
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lo, max(hi, lo + 1))


# ---------------------------------------------------------------------------
# Lenient ROI verification
# ---------------------------------------------------------------------------

def _verify_roi(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """
    Cheap sanity check — recall-first, only rejects *obvious* non-signs.

    A real sign is either (a) noticeably colour-saturated, or (b) an
    achromatic disk: a bright field with dark structure (ring + glyph) and
    real edges inside it.  Flat sky / plaster / road is bright but has almost
    no dark pixels and no internal edges → rejected.  Everything borderline is
    kept and left to the classifier's confidence / entropy gate.
    """
    x, y, w, h = box
    crop = frame_bgr[y:y + h, x:x + w]
    if crop.size == 0 or w < 6 or h < 6:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    n = s.size

    # S>=110 (was 70) so a face/beige patch is no longer accepted on the
    # "colour cue" shortcut — matches the tightened _COLOUR_RANGES floors.
    sat_frac = np.count_nonzero((s >= 110) & (v >= 50)) / n
    if sat_frac >= 0.12:                       # colour cue → accept
        return True

    bright_frac = np.count_nonzero(v >= 150) / n
    dark_frac   = np.count_nonzero(v <= 90) / n
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edge_density = np.count_nonzero(_auto_canny(gray)) / n

    # Achromatic sign cue: bright disk + some dark structure + real edges.
    return bright_frac >= 0.18 and dark_frac >= 0.05 and edge_density >= 0.06


# ---------------------------------------------------------------------------
# Candidate sources
# ---------------------------------------------------------------------------

def _shape_ok(cnt: np.ndarray, bbox_w: int, bbox_h: int) -> bool:
    """Loosened version of the old 4-gate quality check (recall-first)."""
    area = cv2.contourArea(cnt)
    if area < 1:
        return False

    perimeter = cv2.arcLength(cnt, True)
    if perimeter <= 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
    is_shape = circularity >= 0.40 or (3 <= len(approx) <= 10)
    if not is_shape:
        return False

    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area > 0 and area / hull_area < 0.45:      # was 0.55
        return False

    bbox_area = bbox_w * bbox_h
    if bbox_area > 0 and area / bbox_area < 0.20:      # was 0.28
        return False
    return True


def _detect_colour(enhanced: np.ndarray, min_side: int, max_side: int
                    ) -> list[tuple[int, int, int, int]]:
    hsv  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    mask = _build_mask(hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if min(cw, ch) < min_side or max(cw, ch) > max_side:
            continue
        if not (0.55 <= cw / max(ch, 1) <= 1.80):
            continue
        if not _shape_ok(cnt, cw, ch):
            continue
        out.append((x, y, cw, ch))
    return out


def _detect_circles(gray: np.ndarray, min_side: int, max_side: int
                     ) -> list[tuple[int, int, int, int]]:
    """
    Hough circles — colour-independent, recovers achromatic round signs and
    distant faint rings.  param2 kept low (accumulator) for recall; the
    false circles this lets through are killed by _verify_roi + downstream.
    """
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(20, min_side),
        param1=120,                 # Canny high threshold
        param2=45,                  # accumulator votes — raised 32→45: a face
                                    # oval triggered weak circles at 32
        minRadius=max(8, min_side // 2),
        maxRadius=max_side // 2,
    )
    if circles is None:
        return []
    out = []
    for cx, cy, r in np.round(circles[0]).astype(int):
        side = int(2 * r * 1.02)                       # ~tight to the rim
                                                       # (GTSRB ROI is tight;
                                                       # was 1.15 → too loose)
        x = cx - side // 2
        y = cy - side // 2
        out.append((x, y, side, side))
    return out


def _detect_polygons(edges: np.ndarray, min_side: int, max_side: int
                      ) -> list[tuple[int, int, int, int]]:
    """Triangles / diamond / octagon from the edge map (colour-independent)."""
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _CLOSE_K, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.045 * peri, True)
        v = len(approx)
        # 3 = triangle, 4 = diamond/square, 7-8 ≈ octagon (≈ circle too).
        if v not in (3, 4, 7, 8):
            continue
        if not cv2.isContourConvex(cv2.convexHull(approx)):
            continue
        x, y, cw, ch = cv2.boundingRect(approx)
        if min(cw, ch) < min_side or max(cw, ch) > max_side:
            continue
        if not (0.55 <= cw / max(ch, 1) <= 1.85):
            continue
        area = cv2.contourArea(approx)
        if area / max(cw * ch, 1) < 0.30:              # fill the bbox enough
            continue
        out.append((x, y, cw, ch))
    return out


# ---------------------------------------------------------------------------
# Merge / de-duplicate
# ---------------------------------------------------------------------------

def _iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


def _dedupe(boxes: list[tuple[int, int, int, int]],
            iou_thresh: float = 0.45) -> list[tuple[int, int, int, int]]:
    """Drop near-duplicate boxes coming from different paths (keep the larger)."""
    order = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for box in order:
        if all(_iou(box, k) < iou_thresh for k in kept):
            kept.append(box)
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(frame_bgr: np.ndarray,
           min_side: int = 18,
           max_side: int = 320) -> list[tuple[int, int, int, int]]:
    """
    Return (x, y, w, h) bounding boxes of candidate sign regions.

    Args:
        frame_bgr:  Input frame, BGR.
        min_side:   Minimum pixel length of the shorter bbox side
                    (lowered from 25 → 18 for distant signs).
        max_side:   Maximum pixel length of the longer bbox side.
    """
    fh, fw = frame_bgr.shape[:2]

    enhanced = _apply_clahe(frame_bgr)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    edges = _auto_canny(gray)

    raw: list[tuple[int, int, int, int]] = []
    raw += _detect_colour(enhanced, min_side, max_side)
    raw += _detect_circles(gray, min_side, max_side)
    raw += _detect_polygons(edges, min_side, max_side)

    rois: list[tuple[int, int, int, int]] = []
    for (x, y, cw, ch) in raw:
        # No expansion: training crops tightly to the GTSRB ROI (sign fills
        # ~90% of the frame), and the Flatten-based CNN is scale/centre
        # sensitive. The 12% pad fed it sign-on-background crops it never saw
        # → confidently-wrong predictions. Keep the box tight to the contour.
        pad = 0
        x  = max(0, x - pad)
        y  = max(0, y - pad)
        cw = min(fw - x, cw + 2 * pad)
        ch = min(fh - y, ch + 2 * pad)
        if x < 0 or y < 0 or cw < min_side or ch < min_side:
            continue
        box = (x, y, cw, ch)
        if _verify_roi(frame_bgr, box):
            rois.append(box)

    return _dedupe(rois)


def draw_detections(frame: np.ndarray,
                    rois: list[tuple[int, int, int, int]],
                    labels: list[str] | None = None,
                    color: tuple[int, int, int] = (0, 200, 0)) -> np.ndarray:
    """Draw bounding boxes (and optional labels) on a copy of the frame."""
    out = frame.copy()
    for i, (x, y, w, h) in enumerate(rois):
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        if labels and i < len(labels):
            cv2.putText(out, labels[i], (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out
