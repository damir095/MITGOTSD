"""
Full two-stage pipeline: detect sign regions → classify each ROI.

Usage:
    python -m src.video_pipeline --model path/to/model.h5 [--cam 0] [--input-size 48]

Keys:
    q  — quit
    s  — save current frame as JPEG
    d  — toggle colour-mask debug overlay (top-left corner)
    +  — raise confidence threshold by 0.02
    -  — lower confidence threshold by 0.02
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.detector import detect, _apply_clahe

# ── 43 GTSRB class names (index = ClassId) ────────────────────────────────
CLASS_NAMES = [
    "20 км/ч",          "30 км/ч",          "50 км/ч",
    "60 км/ч",          "70 км/ч",          "80 км/ч",
    "Конец 80 км/ч",    "100 км/ч",         "120 км/ч",
    "Обгон запрещён",   "Обгон >3.5т запр.","Приоритет на пересеч.",
    "Главная дорога",   "Уступи дорогу",    "Стоп",
    "Движение запрещ.", "Движ. >3.5т запр.","Въезд запрещён",
    "Осторожно",        "Опасный поворот Л","Опасный поворот П",
    "Двойной поворот",  "Неровная дорога",  "Скользкая дорога",
    "Сужение дороги П", "Дорожные работы",  "Светофор",
    "Пешеходы",         "Дети",             "Велосипеды",
    "Лёд/снег",         "Дикие животные",   "Конец ограничений",
    "Поворот направо",  "Поворот налево",   "Прямо",
    "Прямо или право",  "Прямо или лево",   "Держаться правее",
    "Держаться левее",  "Круговое движение","Конец запрета обгона",
    "Конец запрета обг.>3.5т",
]
_NUM_CLASSES = len(CLASS_NAMES)   # 43
_MAX_ENTROPY = np.log(_NUM_CLASSES)  # ≈ 3.76 — entropy of uniform distribution

# ── Tunable constants ─────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.96   # raised 0.92→0.96 — model trained only on signs, so
                         # any patch gets *some* class; higher bar = fewer false alarms
ENTROPY_MAX     = 0.40   # fraction of max entropy; reject if distribution too flat
                         # (lowered 0.55→0.40 to drop more "spread" non-sign patches)
MARGIN_MIN      = 0.30   # require top1 − top2 ≥ this. A real sign wins by a clear
                         # margin; "confidently-wrong" non-sign patches often hover
                         # between two similar classes even at high top1 confidence.
PERSIST_FRAMES  = 6      # frames to keep a track alive without a new detection
GRID_CELL       = 50     # px — position grid for track identity
DETECT_WIDTH    = 640    # px — detector runs on frame scaled to this width
NMS_IOU_THRESH  = 0.35   # IoU above which two boxes are considered the same sign

# ── Cyrillic font ─────────────────────────────────────────────────────────
_FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
]
_PIL_FONT = None
for _fp in _FONT_PATHS:
    try:
        _PIL_FONT = ImageFont.truetype(_fp, 18)
        break
    except OSError:
        pass
if _PIL_FONT is None:
    _PIL_FONT = ImageFont.load_default()


# ── Helpers ───────────────────────────────────────────────────────────────

def _put_text(img_bgr: np.ndarray, text: str, xy: tuple[int, int],
              color: tuple[int, int, int] = (0, 220, 0)) -> np.ndarray:
    """Draw Cyrillic-safe text via PIL; return modified BGR ndarray."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    draw.text((xy[0] + 1, xy[1] + 1), text, font=_PIL_FONT, fill=(0, 0, 0))
    draw.text(xy, text, font=_PIL_FONT, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _iou(a: tuple, b: tuple) -> float:
    """IoU for two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax1+aw, bx1+bw); iy2 = min(ay1+ah, by1+bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


def _nms(rois: list[tuple], scores: list[float],
         iou_thresh: float = NMS_IOU_THRESH) -> list[int]:
    """Non-maximum suppression. Returns indices of surviving boxes."""
    if not rois:
        return []
    order = sorted(range(len(rois)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    while order:
        best = order.pop(0)
        kept.append(best)
        order = [i for i in order if _iou(rois[best], rois[i]) < iou_thresh]
    return kept


def _is_uncertain(probs: np.ndarray, entropy_max_frac: float) -> bool:
    """
    Return True if the probability distribution is too flat (model uncertain).

    A model trained only on traffic signs will assign *some* class to any patch,
    even random noise.  Entropy check catches these: a confident prediction is
    peaked (low entropy); a confused prediction is spread across many classes.
    """
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))
    return entropy > entropy_max_frac * _MAX_ENTROPY


# ── Model loading & inference ─────────────────────────────────────────────

def load_keras_model(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


def classify_batch(model, patches_bgr: list[np.ndarray],
                   input_size: int) -> np.ndarray:
    """
    Classify all patches in one model.predict() call (much faster than N calls).

    Applies CLAHE + resize + normalise — identical to training preprocessing.
    Returns array of shape (N, 43) with softmax probabilities.
    """
    processed = []
    for patch in patches_bgr:
        p = _apply_clahe(patch)
        p = cv2.resize(p, (input_size, input_size))
        processed.append(p.astype(np.float32) / 255.0)
    batch = np.stack(processed, axis=0)          # (N, H, W, 3)
    return model.predict(batch, verbose=0)        # (N, 43)


# ── Main loop ─────────────────────────────────────────────────────────────

def run(model_path: str, cam_id: int | str = 0, input_size: int = 48):
    model = load_keras_model(model_path)
    print(f"Модель загружена: {model_path}  (вход {input_size}×{input_size})")
    print(f"Порог уверенности: {CONF_THRESHOLD:.2f}  "
          f"| энтропийный порог: {ENTROPY_MAX:.2f} × max"
          f"  | мин. разрыв top1−top2: {MARGIN_MIN:.2f}")
    print("Клавиши: [q] выход  [s] снимок  [d] маска  [+/-] порог")

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_id}")

    save_dir = Path("experiments/screenshots")
    save_dir.mkdir(parents=True, exist_ok=True)

    # tracked[(gx, gy)] = [cls1, conf1, cls2, conf2, x, y, w, h, ttl]
    tracked: dict = {}
    show_debug = False
    conf_threshold = CONF_THRESHOLD

    prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_orig, w_orig = frame.shape[:2]

        # ── Detection on scaled frame ──────────────────────────────────
        scale      = DETECT_WIDTH / w_orig
        det_frame  = cv2.resize(frame, (DETECT_WIDTH, int(h_orig * scale)))
        rois_sc    = detect(det_frame)

        inv  = 1.0 / scale
        rois = [(int(x*inv), int(y*inv), int(w*inv), int(h*inv))
                for (x, y, w, h) in rois_sc]

        # ── Batch classification ───────────────────────────────────────
        patches, valid_rois = [], []
        for (x, y, w, h) in rois:
            patch = frame[y:y+h, x:x+w]
            if patch.size == 0:
                continue
            patches.append(patch)
            valid_rois.append((x, y, w, h))

        candidates: list[tuple] = []
        scores:     list[float] = []

        if patches:
            all_probs = classify_batch(model, patches, input_size)  # one call
            for probs, (x, y, w, h) in zip(all_probs, valid_rois):
                # Entropy gate — skip if model is uncertain
                if _is_uncertain(probs, ENTROPY_MAX):
                    continue
                top2 = np.argsort(probs)[::-1][:2]
                cls1, conf1 = int(top2[0]), float(probs[top2[0]])
                cls2, conf2 = int(top2[1]), float(probs[top2[1]])
                # Margin gate — reject ambiguous top1/top2 (likely a non-sign
                # patch the sign-only classifier can't decide on).
                if conf1 - conf2 < MARGIN_MIN:
                    continue
                candidates.append((cls1, conf1, cls2, conf2, x, y, w, h))
                scores.append(conf1)

        # ── NMS ────────────────────────────────────────────────────────
        boxes_for_nms = [(c[4], c[5], c[6], c[7]) for c in candidates]
        keep_idx = _nms(boxes_for_nms, scores)

        # ── Age tracks ─────────────────────────────────────────────────
        for key in list(tracked):
            tracked[key][-1] -= 1
            if tracked[key][-1] <= 0:
                del tracked[key]

        # ── Update tracks with confident, surviving detections ─────────
        for i in keep_idx:
            cls1, conf1, cls2, conf2, x, y, w, h = candidates[i]
            if conf1 >= conf_threshold:
                gx = (x + w // 2) // GRID_CELL
                gy = (y + h // 2) // GRID_CELL
                tracked[(gx, gy)] = [cls1, conf1, cls2, conf2,
                                     x, y, w, h, PERSIST_FRAMES]

        # ── FPS ────────────────────────────────────────────────────────
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        # ── Render ─────────────────────────────────────────────────────
        out = frame.copy()

        if show_debug:
            # Colour mask preview in top-left corner
            from src.detector import _build_mask as _mk
            _enh = _apply_clahe(det_frame)
            _hsv = cv2.cvtColor(_enh, cv2.COLOR_BGR2HSV)
            _m   = _mk(_hsv)
            _m3  = cv2.cvtColor(_m, cv2.COLOR_GRAY2BGR)
            ph, pw = h_orig // 4, w_orig // 4
            _m3  = cv2.resize(_m3, (pw, ph))
            out[0:ph, 0:pw] = _m3
            cv2.putText(out, "mask", (4, ph - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        for cls1, conf1, cls2, conf2, x, y, w, h, _ in tracked.values():
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 220, 0), 2)
            label = f"{CLASS_NAMES[cls1]}  {conf1*100:.0f}%"
            if conf2 >= 0.08:
                label += f"  |  {CLASS_NAMES[cls2]} {conf2*100:.0f}%"
            out = _put_text(out, label, (x, max(y - 22, 2)))

        thr_color = (80, 220, 80) if conf_threshold >= 0.90 else (80, 180, 220)
        out = _put_text(out, f"FPS {fps:.1f}", (10, 8),
                        color=(200, 200, 200))
        out = _put_text(out,
                        f"порог {conf_threshold:.2f}  [+/-]  "
                        f"{'маска вкл' if show_debug else '[d] маска'}  [q] выход",
                        (10, h_orig - 10), color=(160, 160, 160))

        # Candidate count (before conf gate) — useful for tuning
        n_cand = len(candidates)
        n_show = len(tracked)
        out = _put_text(out, f"кандидатов: {n_cand}  показываем: {n_show}",
                        (10, 30), color=thr_color)

        cv2.imshow("Traffic Sign Detection", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            p = save_dir / f"frame_{int(time.time())}.jpg"
            cv2.imwrite(str(p), out)
            print(f"Saved {p}")
        elif key == ord("d"):
            show_debug = not show_debug
        elif key == ord("+") or key == ord("="):
            conf_threshold = min(0.99, conf_threshold + 0.02)
            print(f"Порог → {conf_threshold:.2f}")
        elif key == ord("-"):
            conf_threshold = max(0.50, conf_threshold - 0.02)
            print(f"Порог → {conf_threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Детектор дорожных знаков (цвет + CNN, батчевый инференс)"
    )
    parser.add_argument("--model",      required=True,
                        help="Путь к .h5 модели (gtsrb_vgg_v2.h5)")
    parser.add_argument("--cam",        default="0",
                        help="Индекс камеры (0,1,...) или путь к видеофайлу")
    parser.add_argument("--input-size", type=int, default=48,
                        help="Размер входа модели (48 для v2, 32 для v1)")
    args = parser.parse_args()
    cam = int(args.cam) if args.cam.isdigit() else args.cam
    run(args.model, cam, args.input_size)
