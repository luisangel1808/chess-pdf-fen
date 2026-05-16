"""
main.py
-------
CLI entry point for the Chess PDF → FEN extractor.

Usage:
    python main.py <path_to_pdf> [--api-key KEY] [--output results.json] [--save-boards]

Options:
    --api-key     Chessvision.ai API key for cloud-based recognition (more accurate).
    --output      Path to write JSON results (default: results.json).
    --save-boards Save each detected board image as a PNG in ./boards/.
    --verbose     Enable debug logging.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from pdf_extractor import extract_images
from board_detector import detect_chessboards
from fen_recognizer import board_to_fen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract chess board images from a PDF and convert them to FEN."
    )
    parser.add_argument("pdf", help="Path to the input PDF file.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CHESSVISION_API_KEY"),
        help="Chessvision.ai API key (or set CHESSVISION_API_KEY env var).",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Output JSON file path (default: results.json).",
    )
    parser.add_argument(
        "--save-boards",
        action="store_true",
        help="Save each detected board image as PNG in ./boards/.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        sys.exit(1)

    boards_dir = Path("boards")
    if args.save_boards:
        boards_dir.mkdir(exist_ok=True)

    results = []
    board_count = 0

    logger.info("Processing PDF: %s", pdf_path)

    for page_num, img_idx, pil_img in extract_images(pdf_path):
        logger.info(
            "Page %d, image %d: %dx%d px — detecting chess boards...",
            page_num, img_idx, pil_img.width, pil_img.height,
        )

        boards = detect_chessboards(pil_img)
        logger.info("  → %d board(s) detected", len(boards))

        for board_idx, board_img in enumerate(boards):
            board_count += 1
            fen = board_to_fen(board_img, api_key=args.api_key)

            entry = {
                "id": board_count,
                "page": page_num,
                "image_index": img_idx,
                "board_index": board_idx,
                "fen": fen,
                "fen_full": f"{fen} w - - 0 1",  # placeholder active/castling/etc.
            }
            results.append(entry)

            logger.info("  Board #%d FEN: %s", board_count, fen)

            if args.save_boards:
                out_path = boards_dir / f"board_p{page_num}_i{img_idx}_b{board_idx}.png"
                board_img.save(str(out_path))
                logger.info("  Saved board image: %s", out_path)

    # Write JSON output
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(
        "Done. %d board(s) found across %d page(s). Results → %s",
        board_count, _count_pages(results), output_path,
    )

    # Also print a summary table to stdout
    print("\n" + "=" * 60)
    print(f"  Chess PDF FEN Extractor — {pdf_path.name}")
    print("=" * 60)
    if not results:
        print("  No chess boards detected.")
    else:
        print(f"  {'#':<4} {'Page':<6} {'FEN'}")
        print("  " + "-" * 56)
        for r in results:
            print(f"  {r['id']:<4} {r['page']:<6} {r['fen']}")
    print("=" * 60)
    print(f"  Full results saved to: {output_path}\n")


def _count_pages(results: list[dict]) -> int:
    return len({r["page"] for r in results})


if __name__ == "__main__":
    main()
