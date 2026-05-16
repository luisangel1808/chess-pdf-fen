"""
pdf_extractor.py
----------------
Extracts all images from a PDF file using PyMuPDF (fitz).
Returns a list of (page_number, image_index, PIL.Image) tuples.
"""

import io
import logging
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

# Minimum image dimensions to consider (ignore tiny icons / decorations)
MIN_WIDTH = 80
MIN_HEIGHT = 80


def extract_images(
    pdf_path: str | Path,
) -> Generator[tuple[int, int, Image.Image], None, None]:
    """
    Yield (page_number, image_index, PIL.Image) for every image in the PDF.

    Args:
        pdf_path: Path to the PDF file.

    Yields:
        Tuples of (1-based page number, 0-based image index on that page, PIL image).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    logger.info("Opened PDF '%s' (%d pages)", pdf_path.name, len(doc))

    for page_num, page in enumerate(doc, start=1):
        image_list = page.get_images(full=True)
        logger.debug("Page %d: %d image(s) found", page_num, len(image_list))

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                if pil_img.width < MIN_WIDTH or pil_img.height < MIN_HEIGHT:
                    logger.debug(
                        "Skipping small image %dx%d on page %d",
                        pil_img.width, pil_img.height, page_num,
                    )
                    continue

                yield page_num, img_idx, pil_img

            except Exception as exc:
                logger.warning(
                    "Could not extract image xref=%d on page %d: %s",
                    xref, page_num, exc,
                )

    doc.close()
