"""
Cyrillic text overlay on OpenCV (BGR) frames.

`cv2.putText` only renders ASCII (default font has no Cyrillic) — Russian
labels come out as `???`. This helper draws via PIL/ImageFont (uses a real
TTF), converts back to BGR. Batched per frame to keep the convert overhead
to one round-trip even when several labels are drawn.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_fonts: dict[int, ImageFont.ImageFont] = {}


def _font(size: int) -> ImageFont.ImageFont:
    if size in _fonts:
        return _fonts[size]
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            _fonts[size] = ImageFont.truetype(p, size)
            return _fonts[size]
    _fonts[size] = ImageFont.load_default()
    return _fonts[size]


def draw_texts(bgr: np.ndarray, items) -> None:
    """In-place: draw a batch of texts on a BGR frame in one PIL round-trip.

    items: iterable of (text, (x, y), color_bgr, size_px).
    """
    items = list(items)
    if not items:
        return
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, xy, bgr_color, size in items:
        b, g, r = bgr_color
        draw.text(xy, text, fill=(r, g, b), font=_font(size))
    bgr[...] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
