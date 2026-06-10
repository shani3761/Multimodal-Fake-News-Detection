"""
utils/image_utils.py — Image Preprocessing & Visualization Helpers
====================================================================
Utility functions used by both the image model and the results display.
"""

from __future__ import annotations
import io
import base64
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AppConfig


# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image,
                     size: tuple[int, int] = AppConfig.IMAGE_SIZE) -> Image.Image:
    """Resize and convert to RGB, ready for model input."""
    return image.convert("RGB").resize(size, Image.LANCZOS)


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Encode a PIL image to raw bytes."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode image as Base64 data-URI (for HTML embedding)."""
    raw = image_to_bytes(image, fmt)
    b64 = base64.b64encode(raw).decode()
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def bytes_to_image(data: bytes) -> Image.Image:
    """Decode bytes → PIL Image."""
    return Image.open(io.BytesIO(data))


def pil_from_upload(uploaded_file) -> Image.Image:
    """Convert a Streamlit UploadedFile to a PIL Image."""
    return Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")


# ── ELA helper ───────────────────────────────────────────────────────────────
def ela_visualization(image: Image.Image,
                      quality: int = AppConfig.ELA_QUALITY,
                      scale: int = AppConfig.ELA_SCALE) -> tuple[Image.Image, float]:
    """
    Compute and scale the Error-Level-Analysis difference image.

    Returns
    -------
    (ela_pil, mean_error)
    """
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")

    orig = np.array(image.convert("RGB"), dtype=np.float32)
    comp = np.array(compressed,            dtype=np.float32)
    diff = np.abs(orig - comp)

    mean_error = float(diff.mean() / 255.0)
    scaled = np.clip(diff * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(scaled), mean_error


# ── Heatmap overlay ──────────────────────────────────────────────────────────
def overlay_heatmap(image: Image.Image,
                    heatmap: np.ndarray,
                    alpha: float = 0.45) -> Image.Image:
    """
    Blend a 2-D numpy heatmap (values 0–1) onto the original image using a
    jet colour map.
    """
    try:
        import cv2

        h, w = image.height, image.width
        hm_resized = cv2.resize(heatmap.astype(np.float32), (w, h))
        hm_uint8   = (hm_resized * 255).astype(np.uint8)
        jet        = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        jet_rgb    = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
        orig_np    = np.array(image.convert("RGB"), dtype=np.uint8)
        blended    = cv2.addWeighted(orig_np, 1 - alpha, jet_rgb, alpha, 0)
        return Image.fromarray(blended)
    except Exception:
        return image


def add_label_badge(image: Image.Image, label: str, score: float) -> Image.Image:
    """Stamp a coloured label badge onto a copy of the image."""
    colour_map = {
        "FAKE":       (220, 50,  50,  200),
        "SUSPICIOUS": (255, 165,  0,  200),
        "REAL":       ( 50, 200, 80,  200),
        "UNCERTAIN":  (150, 150, 150, 200),
    }
    rgba   = colour_map.get(label, (150, 150, 150, 200))
    copy   = image.convert("RGBA").copy()
    draw   = ImageDraw.Draw(copy)
    text   = f"  {label}  {score:.0%}  "
    x, y   = 10, 10
    draw.rectangle([x, y, x + len(text) * 9, y + 28],
                   fill=rgba)
    draw.text((x + 4, y + 5), text, fill=(255, 255, 255, 255))
    return copy.convert("RGB")


# ── Thumbnail ────────────────────────────────────────────────────────────────
def thumbnail(image: Image.Image,
              max_size: tuple[int, int] = (400, 400)) -> Image.Image:
    """Return a proportionally resized copy for display."""
    img = image.copy()
    img.thumbnail(max_size, Image.LANCZOS)
    return img


# ── Collage helper ───────────────────────────────────────────────────────────
def make_frame_collage(frames: list[Image.Image],
                       scores: list[float],
                       cols: int = 4) -> Image.Image:
    """
    Arrange frame thumbnails in a grid with a coloured border that
    encodes the manipulation score (red = high, green = low).
    """
    cell = (160, 120)
    rows = (len(frames) + cols - 1) // cols
    canvas = Image.new("RGB", (cell[0] * cols + (cols - 1) * 4,
                               cell[1] * rows + (rows - 1) * 4 + 20),
                       (30, 30, 40))
    draw = ImageDraw.Draw(canvas)

    for i, (frame, score) in enumerate(zip(frames, scores)):
        r, c = divmod(i, cols)
        x = c * (cell[0] + 4)
        y = r * (cell[1] + 4)

        thumb = frame.resize(cell, Image.LANCZOS)

        # Border colour: green→red as score increases
        g = int(200 * (1 - score))
        rd = int(200 * score)
        border_colour = (rd, g, 40)

        # Draw border
        draw.rectangle([x - 2, y - 2, x + cell[0] + 2, y + cell[1] + 2],
                       outline=border_colour, width=2)
        canvas.paste(thumb, (x, y))

        # Score text
        draw.text((x + 4, y + cell[1] - 16),
                  f"{score:.0%}", fill=border_colour)

    return canvas
