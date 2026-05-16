"""
board_detector.py
-----------------
Detects and crops chess board regions from an image using OpenCV.
Returns a list of cropped board images (as numpy arrays).
"""

import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a BGR OpenCV image."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert a BGR OpenCV image to a PIL image."""
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def detect_chessboards(pil_img: Image.Image) -> list[Image.Image]:
    """
    Detect chess board regions in a PIL image.

    Strategy:
    1. Convert to grayscale and apply adaptive thresholding.
    2. Find large square-ish contours that could be chess boards.
    3. Apply a perspective warp to get a clean top-down 512x512 view.

    Returns a list of cropped/warped PIL images, one per detected board.
    """
    img = pil_to_cv2(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    min_area = (min(h, w) * 0.15) ** 2  # board must be at least 15% of image size

    boards = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Approximate the contour to a polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            # Check aspect ratio is roughly square (0.8 – 1.2)
            rect = cv2.boundingRect(approx)
            x, y, bw, bh = rect
            ratio = bw / bh if bh > 0 else 0
            if 0.75 <= ratio <= 1.33:
                warped = _four_point_transform(img, approx.reshape(4, 2))
                boards.append(cv2_to_pil(warped))

    # Fallback: if no 4-point contour found, try bounding-rect heuristic
    if not boards:
        boards = _fallback_square_crop(img, contours, min_area)

    return boards


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp a quadrilateral region to a 512x512 square."""
    rect = _order_points(pts)
    dst = np.array([[0, 0], [511, 0], [511, 511], [0, 511]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (512, 512))


def _fallback_square_crop(
    img: np.ndarray, contours: list, min_area: float
) -> list[Image.Image]:
    """
    Fallback: find the largest roughly-square bounding box among all contours
    and return it as a single board candidate.
    """
    h, w = img.shape[:2]
    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        ratio = bw / bh if bh > 0 else 0
        if 0.75 <= ratio <= 1.33 and area > best_area:
            best_area = area
            best = (x, y, bw, bh)

    if best:
        x, y, bw, bh = best
        side = max(bw, bh)
        # Center the crop
        cx, cy = x + bw // 2, y + bh // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        crop = img[y1:y2, x1:x2]
        resized = cv2.resize(crop, (512, 512))
        return [cv2_to_pil(resized)]

    # Last resort: return the whole image resized
    return [cv2_to_pil(cv2.resize(img, (512, 512)))]
