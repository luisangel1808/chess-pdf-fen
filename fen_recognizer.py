"""
fen_recognizer.py
-----------------
Converts a cropped chess board image (PIL) to a FEN string.

Recognition priority:
  1. Trained local CNN  (model/chess_classifier.pt)  ← best accuracy
  2. Heuristic fallback (no model needed)             ← rough, offline

Train the CNN with:
    python extract_samples.py
    python label_tool.py
    python train.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_DIR  = Path("model")
MODEL_PATH = MODEL_DIR / "chess_classifier.pt"
CLASS_PATH = MODEL_DIR / "classes.json"

BOARD_PX = 512
SQ_PX    = BOARD_PX // 8   # 64

CLASSES = ["empty", "wK", "wQ", "wR", "wB", "wN", "wP",
           "bK", "bQ", "bR", "bB", "bN", "bP"]

PIECE_CHARS = {
    "wK": "K", "wQ": "Q", "wR": "R", "wB": "B", "wN": "N", "wP": "P",
    "bK": "k", "bQ": "q", "bR": "r", "bB": "b", "bN": "n", "bP": "p",
    "empty": "1",
}

# ---------------------------------------------------------------------------
# Model loader (lazy, singleton)
# ---------------------------------------------------------------------------

_model = None
_class_list: list[str] = CLASSES


def _load_model():
    global _model, _class_list
    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        return None

    try:
        import torch
        import torch.nn as nn

        # Load class list
        if CLASS_PATH.exists():
            with open(CLASS_PATH) as f:
                data = json.load(f)
            _class_list = data["classes"]

        # Rebuild the same architecture as train.py
        class ChessCNN(nn.Module):
            def __init__(self, num_classes=13):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.MaxPool2d(2), nn.Dropout2d(0.1),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(2), nn.Dropout2d(0.1),
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.MaxPool2d(2), nn.Dropout2d(0.2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.4),
                    nn.Linear(256, num_classes),
                )
            def forward(self, x):
                return self.classifier(self.features(x))

        model = ChessCNN(num_classes=len(_class_list))
        state = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _model = model
        logger.info("Loaded trained CNN from %s", MODEL_PATH)
        return _model

    except Exception as exc:
        logger.warning("Could not load model: %s — using heuristic fallback", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def board_to_fen(board_img: Image.Image, api_key: Optional[str] = None) -> str:
    """Convert a PIL chess board image to a FEN piece-placement string."""
    model = _load_model()
    if model is not None:
        return _cnn_fen(board_img, model)
    return _heuristic_fen(board_img)


# ---------------------------------------------------------------------------
# CNN path
# ---------------------------------------------------------------------------

def _cnn_fen(board_img: Image.Image, model) -> str:
    import torch
    import torchvision.transforms as T

    gray = _preprocess(board_img)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])

    rows = []
    for rank in range(8):
        empty_run = 0
        row = ""
        for file in range(8):
            sq = _get_sq(gray, rank, file)
            sq_pil = Image.fromarray(sq)
            tensor = transform(sq_pil).unsqueeze(0)  # 1x1x64x64

            with torch.no_grad():
                logits = model(tensor)
                pred_idx = logits.argmax(1).item()

            label = _class_list[pred_idx]
            piece_char = PIECE_CHARS.get(label, "1")

            if piece_char == "1":
                empty_run += 1
            else:
                if empty_run:
                    row += str(empty_run)
                    empty_run = 0
                row += piece_char

        if empty_run:
            row += str(empty_run)
        rows.append(row)

    return "/".join(rows)


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _heuristic_fen(board_img: Image.Image) -> str:
    """
    Rough heuristic for when no trained model is available.
    Works reasonably for simple endgame diagrams; unreliable for complex positions.
    """
    gray = _preprocess(board_img)

    # Otsu binarisation: ink = 255
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Per-square ink density
    densities = np.array([
        [float(np.sum(_get_sq(ink, r, f) > 0)) / (SQ_PX * SQ_PX)
         for f in range(8)]
        for r in range(8)
    ])

    # Separate baselines for light and dark squares
    light = sorted(densities[r, f] for r in range(8) for f in range(8) if (r+f)%2==0)
    dark  = sorted(densities[r, f] for r in range(8) for f in range(8) if (r+f)%2==1)
    n = max(1, 32 * 3 // 10)
    bl = float(np.mean(light[:n]))
    bd = float(np.mean(dark[:n]))
    tl = max(bl * 1.8, bl + 0.05)
    td = max(bd * 1.8, bd + 0.06)

    rows = []
    for rank in range(8):
        empty_run = 0
        row = ""
        for file in range(8):
            is_light = (rank + file) % 2 == 0
            thresh = tl if is_light else td

            if densities[rank, file] < thresh:
                empty_run += 1
                continue

            sq_gray = _get_sq(gray, rank, file)
            sq_ink  = _get_sq(ink,  rank, file)
            if not is_light:
                sq_ink = _clean_crosshatch(sq_ink)

            is_white = _is_white_piece(sq_gray, sq_ink)
            ptype    = _classify_type(sq_ink)
            prefix   = "w" if is_white else "b"
            piece    = PIECE_CHARS.get(prefix + ptype, "1")

            if empty_run:
                row += str(empty_run)
                empty_run = 0
            row += piece

        if empty_run:
            row += str(empty_run)
        rows.append(row)

    return "/".join(rows)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _preprocess(board_img: Image.Image) -> np.ndarray:
    img = board_img.convert("L").resize((BOARD_PX, BOARD_PX), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _get_sq(arr: np.ndarray, rank: int, file: int) -> np.ndarray:
    r0, r1 = rank * SQ_PX, (rank + 1) * SQ_PX
    c0, c1 = file * SQ_PX, (file + 1) * SQ_PX
    return arr[r0:r1, c0:c1].copy()


def _clean_crosshatch(sq_ink: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(sq_ink, cv2.MORPH_OPEN, k)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    result = np.zeros_like(sq_ink)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 20:
            result[labels == i] = 255
    return result


def _is_white_piece(sq_gray: np.ndarray, sq_ink: np.ndarray) -> bool:
    contours, _ = cv2.findContours(sq_ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True
    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    if bw < 4 or bh < 4:
        return True
    roi_gray = sq_gray[y:y+bh, x:x+bw]
    roi_ink  = sq_ink[y:y+bh, x:x+bw]
    non_ink = roi_ink == 0
    if not np.any(non_ink):
        return False
    interior = float(np.mean(roi_gray[non_ink]))
    bg = float(np.mean(sq_gray[sq_ink == 0])) if np.any(sq_ink == 0) else 200.0
    return interior > (bg - 45)


def _classify_type(sq_ink: np.ndarray) -> str:
    h, w = sq_ink.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    sq_ink = cv2.morphologyEx(sq_ink, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(sq_ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "P"
    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    fill = float(np.sum(sq_ink > 0)) / (h * w)
    aspect = bh / max(bw, 1)
    height_frac = bh / h
    width_frac  = bw / w
    ys, _ = np.where(sq_ink > 0)
    top_frac = float(np.sum(sq_ink[:h//2] > 0)) / (float(np.sum(sq_ink > 0)) + 1e-6)
    waist = _waist(sq_ink, y, bh, h, 0.35, 0.65)
    crown_w = _band_w(sq_ink, y, bh, h, 0.0,  0.20)
    base_w  = _band_w(sq_ink, y, bh, h, 0.75, 1.0)
    if fill < 0.07 or height_frac < 0.28: return "P"
    if aspect < 0.85 and width_frac > 0.48: return "R"
    if height_frac > 0.62:
        if waist < 0.75:
            return "B" if crown_w < base_w * 0.85 else "Q"
        if crown_w > base_w * 0.85 and top_frac > 0.46: return "K"
        return "R" if width_frac > 0.55 else "Q"
    if height_frac > 0.42:
        if fill > 0.17 and width_frac > 0.50: return "R"
        return "N" if waist < 0.80 else ("K" if top_frac > 0.52 else "N")
    return "R" if fill > 0.11 and width_frac > 0.50 else "P"


def _waist(mask, y, bh, h, ft, fb):
    yt = min(y + int(bh * ft), h - 1)
    yb = min(y + int(bh * fb), h - 1)
    return (float(np.sum(mask[yt] > 0)) + 1e-6) / (float(np.sum(mask[yb] > 0)) + 1e-6)


def _band_w(mask, y, bh, h, fs, fe):
    y0 = min(y + int(bh * fs), h - 1)
    y1 = min(y + int(bh * fe), h - 1)
    if y1 <= y0: return 0.0
    return float(np.mean(np.sum(mask[y0:y1] > 0, axis=1)))
