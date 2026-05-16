"""
extract_samples.py
------------------
Extracts all 64 squares from every saved board PNG in ./boards/
and saves them into ./samples/unlabeled/ for labeling.

Also auto-labels squares that are clearly empty (very low ink density)
and saves them to ./samples/empty/ to save you labeling time.

Usage:
    python extract_samples.py

Output structure:
    samples/
        unlabeled/   <- squares you need to label
        empty/       <- auto-detected empty squares (already labeled)
        wK/ wQ/ wR/ wB/ wN/ wP/
        bK/ bQ/ bR/ bB/ bN/ bP/
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

BOARDS_DIR = Path("boards")
SAMPLES_DIR = Path("samples")
BOARD_PX = 512
SQ_PX = BOARD_PX // 8

CLASSES = ["empty", "wK", "wQ", "wR", "wB", "wN", "wP",
                              "bK", "bQ", "bR", "bB", "bN", "bP"]

def setup_dirs():
    for cls in CLASSES + ["unlabeled"]:
        (SAMPLES_DIR / cls).mkdir(parents=True, exist_ok=True)

def get_ink_density(sq_gray: np.ndarray) -> float:
    """Fraction of pixels that are dark ink (below threshold)."""
    _, binary = cv2.threshold(sq_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(np.sum(binary > 0)) / (SQ_PX * SQ_PX)

def extract_squares(board_path: Path) -> list[tuple[np.ndarray, str]]:
    """Return list of (square_img, square_name) for all 64 squares."""
    img = Image.open(board_path).convert("L").resize((BOARD_PX, BOARD_PX), Image.LANCZOS)
    gray = np.array(img, dtype=np.uint8)
    squares = []
    for rank in range(8):
        for file in range(8):
            sq = gray[rank*SQ_PX:(rank+1)*SQ_PX, file*SQ_PX:(file+1)*SQ_PX]
            name = f"{board_path.stem}_r{rank}_f{file}"
            squares.append((sq, name))
    return squares

def main():
    setup_dirs()
    board_files = sorted(BOARDS_DIR.glob("*.png"))
    print(f"Found {len(board_files)} board images")

    # Compute global Otsu threshold across all boards for consistent empty detection
    # We'll use per-board thresholds instead for robustness

    total = 0
    auto_empty = 0
    unlabeled = 0

    # Track ink densities to find a good empty threshold
    all_densities = []
    for bf in board_files[:20]:  # sample first 20 boards
        img = Image.open(bf).convert("L").resize((BOARD_PX, BOARD_PX), Image.LANCZOS)
        gray = np.array(img, dtype=np.uint8)
        for rank in range(8):
            for file in range(8):
                sq = gray[rank*SQ_PX:(rank+1)*SQ_PX, file*SQ_PX:(file+1)*SQ_PX]
                all_densities.append(get_ink_density(sq))

    all_densities.sort()
    # The bottom 30% are almost certainly empty squares
    empty_threshold = all_densities[int(len(all_densities) * 0.30)]
    # Add a small margin
    empty_threshold = min(empty_threshold * 1.5, 0.08)
    print(f"Auto-empty threshold: {empty_threshold:.3f}")

    for bf in board_files:
        squares = extract_squares(bf)
        for sq_img, name in squares:
            total += 1
            density = get_ink_density(sq_img)

            if density < empty_threshold:
                # Auto-label as empty
                out_path = SAMPLES_DIR / "empty" / f"{name}.png"
                if not out_path.exists():
                    Image.fromarray(sq_img).save(str(out_path))
                auto_empty += 1
            else:
                # Save to unlabeled for manual labeling
                out_path = SAMPLES_DIR / "unlabeled" / f"{name}.png"
                if not out_path.exists():
                    Image.fromarray(sq_img).save(str(out_path))
                unlabeled += 1

    print(f"\nDone!")
    print(f"  Total squares:    {total}")
    print(f"  Auto-labeled empty: {auto_empty}")
    print(f"  Need labeling:    {unlabeled}")
    print(f"\nNext step: run  python label_tool.py")

if __name__ == "__main__":
    main()
