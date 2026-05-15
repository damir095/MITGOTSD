"""
Colour-based traffic sign region detector (OpenCV, no ML).

Pipeline:
    frame (BGR) → CLAHE → HSV masks → contours → shape/quality filter → ROI list

Key design decisions vs. the naive approach:
- HSV ranges require HIGH saturation (S > 130) so pale/dark non-sign colours are
  rejected before any shape check (removes most false positives from cars, sky,
  road markings, clothing).
- Small morphological kernel (3×3) with minimal iterations prevents separate
  coloured objects from merging into one large fake blob.
- Four quality gates per contour: size, aspect ratio, fill ratio (coloured pixels
  / bbox area), convex-hull ratio.  All four must pass.
- Circularity threshold raised to 0.50; polygon capped at 3–8 vertices — tighter
  than "accept almost anything".

Each ROI is a tuple (x, y, w, h) in pixel coordinates of the input frame.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# HSV colour ranges.  Hue in [0, 180] (OpenCV convention).
# S and V floors are deliberately high to reject faded / dark / pale objects.
# ---------------------------------------------------------------------------
_COLOUR_RANGES = [
    # Red — vivid red borders (speed-limit, prohibition, warning triangles).
    # S > 130 excludes dark maroon, rusty surfaces.
    # V > 100 excludes shadows on red objects.
    (np.array([0,   130, 100]), np.array([8,   255, 255])),
    (np.array([172, 130, 100]), np.array([180, 255, 255])),
    # Blue — mandatory / regulatory signs (round, solid blue fill).
    # S > 130 excludes sky, faded denim, light-blue walls.
    (np.array([100, 130,  80]), np.array([125, 255, 255])),
    # Yellow — warning triangles and "Главная дорога" diamond.
    # Both S and V > 150 rejects yellow road paint, pale amber, dim lamps.
    (np.array([18,  150, 150]), np.array([32,  255, 255])),
]

# Small kernel — fills narrow gaps inside sign ring without merging
# separate objects.  5×5 was causing red cars & stop-lights to fuse.
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def _apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Enhance local contrast via CLAHE on the LAB L-channel."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _build_mask(hsv: np.ndarray) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _COLOUR_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    # Close: fills thin gaps inside circular/triangular sign borders.
    # Open:  removes isolated noise pixels.
    # Fewer iterations + smaller kernel vs. old version → less merging.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL, iterations=1)
    return mask


def _contour_quality(cnt: np.ndarray,
                     bbox_w: int, bbox_h: int) -> bool:
    """
    Four shape-quality gates — all must pass.

    1. Circularity ≥ 0.50  OR  polygon with 3–8 vertices
       (raised from 0.40 / 3–12 to reduce false positives on blobs).
    2. Convex-hull fill ≥ 0.55
       (sign shapes are convex or nearly so; broken blobs have low hull fill).
    3. Bounding-box fill ≥ 0.28
       (sign colour ring covers most of the bbox; scattered pixels fail this).
    4. Aspect ratio already checked in caller (0.60–1.65).
    """
    area = cv2.contourArea(cnt)
    if area < 1:
        return False

    # --- gate 1: shape class ---
    perimeter = cv2.arcLength(cnt, True)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity >= 0.50:
            is_shape = True
        else:
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            is_shape = (3 <= len(approx) <= 8)
    else:
        is_shape = False
    if not is_shape:
        return False

    # --- gate 2: convex hull fill ---
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0 and (area / hull_area) < 0.55:
        return False

    # --- gate 3: bounding-box fill ---
    bbox_area = bbox_w * bbox_h
    if bbox_area > 0 and (area / bbox_area) < 0.28:
        return False

    return True


def detect(frame_bgr: np.ndarray,
           min_side: int = 25,
           max_side: int = 280) -> list[tuple[int, int, int, int]]:
    """
    Return list of (x, y, w, h) bounding boxes of candidate sign regions.

    Args:
        frame_bgr:  Input frame in BGR colour space.
        min_side:   Minimum pixel length of the shorter bbox side.
        max_side:   Maximum pixel length of the longer bbox side.
                    Keeps sizes in the 25–280 px range (Rudov et al. use 35–200 px
                    for MSER; slightly wider here for colour-based approach).
    """
    fh, fw = frame_bgr.shape[:2]

    enhanced = _apply_clahe(frame_bgr)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    mask = _build_mask(hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        # --- size gate (by side length, not area) ---
        short_side = min(cw, ch)
        long_side  = max(cw, ch)
        if short_side < min_side or long_side > max_side:
            continue

        # --- aspect ratio gate ---
        aspect = cw / max(ch, 1)
        if not (0.60 <= aspect <= 1.65):
            continue

        # --- shape / quality gates ---
        if not _contour_quality(cnt, cw, ch):
            continue

        # Padding: 12% of the larger side, clamped to frame
        pad = int(long_side * 0.12)
        x  = max(0, x - pad)
        y  = max(0, y - pad)
        cw = min(fw - x, cw + 2 * pad)
        ch = min(fh - y, ch + 2 * pad)

        rois.append((x, y, cw, ch))

    return rois


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
