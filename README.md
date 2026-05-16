# Chess PDF → FEN Extractor

Reads a PDF file, finds every chess board diagram, and converts each one to a [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) string.

## How it works

```
PDF file
  └─ pdf_extractor.py   → extracts all embedded images (via PyMuPDF)
       └─ board_detector.py  → detects & crops chess board regions (OpenCV)
            └─ fen_recognizer.py  → classifies each square → builds FEN string
```

Two recognition modes are available:

| Mode | How | Accuracy |
|------|-----|----------|
| **Chessvision.ai API** (recommended) | Cloud ML model trained for chess diagrams | Excellent — handles printed books, scans, photos |
| **Heuristic** (offline fallback) | Adaptive threshold + silhouette shape analysis | Limited — works for simple endgame diagrams, unreliable for complex positions |

---

## Installation

```bash
cd chess-pdf-fen
pip install -r requirements.txt
```

> Requires Python 3.10+

---

## Usage

```bash
# Basic (heuristic mode)
python main.py path/to/book.pdf

# With Chessvision.ai API key (recommended for best accuracy)
python main.py path/to/book.pdf --api-key YOUR_KEY

# Save detected board images + verbose output
python main.py path/to/book.pdf --save-boards --verbose

# Custom output file
python main.py path/to/book.pdf --output my_results.json
```

### Environment variable

Instead of `--api-key` you can set:

```bash
set CHESSVISION_API_KEY=your_key_here   # Windows CMD
$env:CHESSVISION_API_KEY="your_key"     # PowerShell
```

---

## Output

A JSON file (default `results.json`) with one entry per detected board:

```json
[
  {
    "id": 1,
    "page": 3,
    "image_index": 0,
    "board_index": 0,
    "fen": "2R4Q/pp1bkp2/4p1r1/qN1p4/Pb6/4B3/1P3PPP/6K1",
    "fen_full": "2R4Q/pp1bkp2/4p1r1/qN1p4/Pb6/4B3/1P3PPP/6K1 w - - 0 1"
  }
]
```

A summary table is also printed to the console:

```
============================================================
  Chess PDF FEN Extractor — chess_book.pdf
============================================================
  #    Page   FEN
  --------------------------------------------------------
  1    3      2R4Q/pp1bkp2/4p1r1/qN1p4/Pb6/4B3/1P3PPP/6K1
  2    7      rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR
============================================================
  Full results saved to: results.json
```

---

## Accuracy notes

- The **Chessvision.ai API** is the recommended path for reliable results. Get a free key at [chessvision.ai](https://chessvision.ai). The free tier supports a limited number of requests per month.
- The **heuristic fallback** uses adaptive thresholding + silhouette shape features. It can handle simple endgame diagrams (few pieces, clean print) but is unreliable for complex positions — distinguishing all 12 piece types from a small grayscale patch requires a trained ML model.
- The FEN output contains only the **piece placement** field. The active colour, castling rights, and move counters are set to placeholder values (`w - - 0 1`).

---

## Project structure

```
chess-pdf-fen/
├── main.py            # CLI entry point
├── pdf_extractor.py   # PDF → PIL images
├── board_detector.py  # PIL image → cropped board images
├── fen_recognizer.py  # board image → FEN string
├── requirements.txt
└── README.md
```
