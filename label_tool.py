"""
label_tool.py
-------------
Interactive labeling tool for chess square images.

Shows each unlabeled square image and lets you press a key to assign its label.
Labeled images are moved to samples/<label>/.

Controls:
    e  = empty
    k  = King   (white)   K  = King   (black)
    q  = Queen  (white)   Q  = Queen  (black)
    r  = Rook   (white)   R  = Rook   (black)
    b  = Bishop (white)   B  = Bishop (black)
    n  = Knight (white)   N  = Knight (black)
    p  = Pawn   (white)   P  = Pawn   (black)
    s  = skip (leave in unlabeled)
    z  = undo last label
    ESC / q+Enter = quit

Usage:
    python label_tool.py [--max N]   # label at most N squares (default: all)

Tips:
    - You only need ~5-10 examples per class for a good model.
    - Label the clearest, most representative examples first.
    - The tool shows a 4x zoomed view so small pieces are visible.
    - Progress is saved after every label — you can quit and resume.
"""

import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np

SAMPLES_DIR = Path("samples")
UNLABELED_DIR = SAMPLES_DIR / "unlabeled"

KEY_MAP = {
    ord('e'): "empty",
    ord('k'): "wK", ord('K'): "bK",
    ord('q'): "wQ", ord('Q'): "bQ",
    ord('r'): "wR", ord('R'): "bR",
    ord('b'): "wB", ord('B'): "bB",
    ord('n'): "wN", ord('N'): "bN",
    ord('p'): "wP", ord('P'): "bP",
}

DISPLAY_SIZE = 256   # pixels for the zoomed display window
GRID_COLS = 6        # columns in the reference grid


def make_reference_grid() -> np.ndarray:
    """Build a reference image showing already-labeled examples."""
    classes = ["wK","wQ","wR","wB","wN","wP","bK","bQ","bR","bB","bN","bP","empty"]
    cell = 80
    cols = GRID_COLS
    rows = (len(classes) + cols - 1) // cols
    grid = np.ones((rows * (cell + 20), cols * (cell + 4)), dtype=np.uint8) * 240

    for i, cls in enumerate(classes):
        r, c = divmod(i, cols)
        y0 = r * (cell + 20)
        x0 = c * (cell + 4)

        cls_dir = SAMPLES_DIR / cls
        examples = list(cls_dir.glob("*.png"))
        if examples:
            ex = cv2.imread(str(examples[0]), cv2.IMREAD_GRAYSCALE)
            ex = cv2.resize(ex, (cell, cell))
            grid[y0:y0+cell, x0:x0+cell] = ex
        else:
            grid[y0:y0+cell, x0:x0+cell] = 200

        cv2.putText(grid, cls, (x0, y0+cell+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)

    return grid


def count_labeled() -> dict:
    classes = ["empty","wK","wQ","wR","wB","wN","wP","bK","bQ","bR","bB","bN","bP"]
    return {cls: len(list((SAMPLES_DIR / cls).glob("*.png"))) for cls in classes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=999999,
                        help="Max number of squares to label in this session")
    args = parser.parse_args()

    unlabeled = sorted(UNLABELED_DIR.glob("*.png"))
    if not unlabeled:
        print("No unlabeled squares found. Run extract_samples.py first.")
        return

    print(f"Found {len(unlabeled)} unlabeled squares")
    print()
    print("Controls:")
    print("  e=empty  k/K=King(w/b)  q/Q=Queen  r/R=Rook  b/B=Bishop  n/N=Knight  p/P=Pawn")
    print("  s=skip   z=undo   ESC=quit")
    print()

    history = []   # list of (src_path, dst_path) for undo
    labeled_count = 0

    cv2.namedWindow("Square", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Square", DISPLAY_SIZE, DISPLAY_SIZE)
    cv2.namedWindow("Reference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reference", 500, 300)

    for img_path in unlabeled:
        if labeled_count >= args.max:
            break

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Show zoomed square
        zoomed = cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Square", zoomed)

        # Show reference grid
        ref = make_reference_grid()
        cv2.imshow("Reference", ref)

        # Show counts
        counts = count_labeled()
        status = "  ".join(f"{k}:{v}" for k, v in counts.items())
        print(f"\r[{labeled_count}/{min(len(unlabeled), args.max)}] {img_path.name[:30]:<30}  {status}", end="", flush=True)

        while True:
            key = cv2.waitKey(0)

            if key == 27 or key == -1:  # ESC
                print("\nQuitting.")
                cv2.destroyAllWindows()
                _print_summary()
                return

            if key == ord('s'):  # skip
                break

            if key == ord('z') and history:  # undo
                src, dst = history.pop()
                shutil.move(str(dst), str(src))
                print(f"\n  Undid: moved back to unlabeled")
                labeled_count -= 1
                break

            if key in KEY_MAP:
                label = KEY_MAP[key]
                dst_dir = SAMPLES_DIR / label
                dst_dir.mkdir(exist_ok=True)
                dst_path = dst_dir / img_path.name
                shutil.move(str(img_path), str(dst_path))
                history.append((img_path, dst_path))
                labeled_count += 1
                break

    cv2.destroyAllWindows()
    print()
    _print_summary()


def _print_summary():
    counts = count_labeled()
    total = sum(counts.values())
    print(f"\nLabeling summary ({total} total labeled):")
    for cls, n in counts.items():
        bar = "#" * n
        print(f"  {cls:6s}: {n:4d}  {bar}")
    print()
    min_count = min(v for k, v in counts.items() if k != "empty")
    if min_count < 5:
        print("⚠  Some classes have fewer than 5 examples. Label more for better accuracy.")
    else:
        print("✓  Ready to train! Run:  python train.py")


if __name__ == "__main__":
    main()
