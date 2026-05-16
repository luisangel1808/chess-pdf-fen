"""
correct_tool.py
---------------
Shows each detected board side-by-side with its predicted FEN rendered
as a board. Click a square on the LEFT image to correct its piece.

This serves two purposes:
  1. Lets you fix wrong FEN entries in results.json immediately.
  2. Saves the corrected squares as new labeled training samples so
     retraining improves accuracy on exactly the cases that failed.

Usage:
    python correct_tool.py                  # review all boards
    python correct_tool.py --page 5         # only boards from page 5
    python correct_tool.py --min-conf 0.7   # only low-confidence boards

Controls (when a square is selected):
    e  = empty
    k/K = white/black King
    q/Q = white/black Queen
    r/R = white/black Rook
    b/B = white/black Bishop
    n/N = white/black Knight
    p/P = white/black Pawn
    ESC = quit and save
    → / ← = next / previous board (without changing anything)
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import chess
    import chess.svg
    HAS_CHESS = True
except ImportError:
    HAS_CHESS = False

BOARDS_DIR  = Path("boards")
SAMPLES_DIR = Path("samples")
RESULTS_FILE = Path("results.json")

BOARD_PX = 512
SQ_PX    = BOARD_PX // 8

CLASSES = ["empty", "wK", "wQ", "wR", "wB", "wN", "wP",
           "bK", "bQ", "bR", "bB", "bN", "bP"]

PIECE_CHARS = {
    "wK": "K", "wQ": "Q", "wR": "R", "wB": "B", "wN": "N", "wP": "P",
    "bK": "k", "bQ": "q", "bR": "r", "bB": "b", "bN": "n", "bP": "p",
    "empty": "1",
}
CHAR_TO_CLASS = {v: k for k, v in PIECE_CHARS.items() if v != "1"}
CHAR_TO_CLASS["1"] = "empty"

KEY_MAP = {
    ord('e'): "empty",
    ord('k'): "wK", ord('K'): "bK",
    ord('q'): "wQ", ord('Q'): "bQ",
    ord('r'): "wR", ord('R'): "bR",
    ord('b'): "wB", ord('B'): "bB",
    ord('n'): "wN", ord('N'): "bN",
    ord('p'): "wP", ord('P'): "bP",
}

# Piece symbols for rendering on the board overlay
PIECE_SYMBOLS = {
    "wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
    "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟",
    "empty": "",
}


def fen_to_grid(fen: str) -> list[list[str]]:
    """Parse FEN piece placement into 8x8 grid of piece chars."""
    grid = []
    for rank_str in fen.split("/"):
        row = []
        for ch in rank_str:
            if ch.isdigit():
                row.extend(["1"] * int(ch))
            else:
                row.append(ch)
        grid.append(row)
    return grid


def grid_to_fen(grid: list[list[str]]) -> str:
    rows = []
    for row in grid:
        s = ""
        empty = 0
        for ch in row:
            if ch == "1":
                empty += 1
            else:
                if empty:
                    s += str(empty)
                    empty = 0
                s += ch
        if empty:
            s += str(empty)
        rows.append(s)
    return "/".join(rows)


def render_board(board_img: np.ndarray, grid: list[list[str]],
                 selected: tuple[int, int] | None = None) -> np.ndarray:
    """
    Render the board image with piece labels overlaid on each square.
    Highlights the selected square in yellow.
    """
    vis = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

    for rank in range(8):
        for file in range(8):
            x0, y0 = file * SQ_PX, rank * SQ_PX
            x1, y1 = x0 + SQ_PX, y0 + SQ_PX

            piece = grid[rank][file]

            # Highlight selected square
            if selected == (rank, file):
                overlay = vis.copy()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 255), -1)
                cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

            # Draw piece label
            if piece != "1":
                color = (255, 255, 255) if piece.islower() else (0, 0, 0)
                bg    = (0, 0, 0)       if piece.islower() else (255, 255, 255)
                label = piece
                font  = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.55
                thick = 1
                (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
                tx = x0 + (SQ_PX - tw) // 2
                ty = y0 + (SQ_PX + th) // 2
                # Background rectangle for readability
                cv2.rectangle(vis, (tx-2, ty-th-2), (tx+tw+2, ty+2), bg, -1)
                cv2.putText(vis, label, (tx, ty), font, scale, color, thick)

    # Draw grid lines
    for i in range(9):
        cv2.line(vis, (i*SQ_PX, 0), (i*SQ_PX, BOARD_PX), (100,100,100), 1)
        cv2.line(vis, (0, i*SQ_PX), (BOARD_PX, i*SQ_PX), (100,100,100), 1)

    return vis


def get_square_from_click(x: int, y: int) -> tuple[int, int]:
    file = min(x // SQ_PX, 7)
    rank = min(y // SQ_PX, 7)
    return rank, file


def save_correction_as_sample(board_img: np.ndarray, rank: int, file: int,
                               label: str, board_name: str):
    """Save the corrected square as a new training sample."""
    sq = board_img[rank*SQ_PX:(rank+1)*SQ_PX, file*SQ_PX:(file+1)*SQ_PX]
    dst_dir = SAMPLES_DIR / label
    dst_dir.mkdir(parents=True, exist_ok=True)
    fname = f"correction_{board_name}_r{rank}_f{file}.png"
    Image.fromarray(sq).save(str(dst_dir / fname))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=int, default=None, help="Filter to a specific page")
    args = parser.parse_args()

    if not RESULTS_FILE.exists():
        print("results.json not found. Run main.py first.")
        return

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    if args.page:
        results = [r for r in results if r["page"] == args.page]

    if not results:
        print("No results to review.")
        return

    print(f"Reviewing {len(results)} boards.")
    print("Click a square to select it, then press a key to correct it.")
    print("→/← arrows to navigate, ESC to quit and save.\n")

    selected_sq: tuple[int, int] | None = None
    idx = 0
    modified = set()

    # Load all board images
    board_imgs: dict[str, np.ndarray] = {}

    def get_board_img(result: dict) -> np.ndarray | None:
        p, i, b = result["page"], result["image_index"], result["board_index"]
        key = f"board_p{p}_i{i}_b{b}"
        if key not in board_imgs:
            path = BOARDS_DIR / f"{key}.png"
            if not path.exists():
                return None
            img = Image.open(path).convert("L").resize((BOARD_PX, BOARD_PX), Image.LANCZOS)
            board_imgs[key] = np.array(img, dtype=np.uint8)
        return board_imgs[key]

    cv2.namedWindow("Board", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Board", BOARD_PX, BOARD_PX)

    def mouse_cb(event, x, y, flags, param):
        nonlocal selected_sq
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_sq = get_square_from_click(x, y)

    cv2.setMouseCallback("Board", mouse_cb)

    while True:
        result = results[idx]
        board_img = get_board_img(result)

        if board_img is None:
            print(f"Board image not found for result {result['id']}, skipping.")
            idx = (idx + 1) % len(results)
            continue

        grid = fen_to_grid(result["fen"])
        vis  = render_board(board_img, grid, selected_sq)

        # Status bar
        status = (f"Board {idx+1}/{len(results)}  "
                  f"Page {result['page']}  "
                  f"FEN: {result['fen'][:40]}...")
        cv2.setWindowTitle("Board", status)
        cv2.imshow("Board", vis)

        key = cv2.waitKey(50)

        if key == 27:  # ESC
            break

        elif key == 83 or key == ord('d'):  # → next
            selected_sq = None
            idx = (idx + 1) % len(results)

        elif key == 81 or key == ord('a'):  # ← prev
            selected_sq = None
            idx = (idx - 1) % len(results)

        elif key in KEY_MAP and selected_sq is not None:
            rank, file = selected_sq
            new_label = KEY_MAP[key]
            new_char  = PIECE_CHARS[new_label]

            old_char = grid[rank][file]
            if old_char != new_char:
                grid[rank][file] = new_char
                result["fen"] = grid_to_fen(grid)
                modified.add(idx)

                # Save as training sample
                p, i, b = result["page"], result["image_index"], result["board_index"]
                board_name = f"p{p}_i{i}_b{b}"
                save_correction_as_sample(board_img, rank, file, new_label, board_name)
                print(f"  Corrected rank={rank} file={file}: "
                      f"{old_char} → {new_char}  (saved to samples/{new_label}/)")

            selected_sq = None  # deselect after correction

    cv2.destroyAllWindows()

    # Save updated results
    if modified:
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(modified)} corrected board(s) to results.json")
        print(f"New training samples saved to samples/")
        print(f"\nRetrain to improve accuracy:")
        print(f"  python train.py --epochs 60")
    else:
        print("\nNo corrections made.")


if __name__ == "__main__":
    main()
