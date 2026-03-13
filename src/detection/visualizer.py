# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Bounding Box Visualizer
==================================
Generates annotated PNG artifacts showing detected layout blocks.
No GPU required — pure Pillow/OpenCV drawing.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from src.models import Block, BlockLabel

# Vibrant, distinct color per block category (RGB)
_LABEL_COLORS: dict[BlockLabel, tuple[int, int, int]] = {
    BlockLabel.TEXT:    (52,  199, 89),    # green
    BlockLabel.TITLE:   (0,   122, 255),   # blue
    BlockLabel.TABLE:   (255, 149, 0),     # orange
    BlockLabel.FIGURE:  (175, 82,  222),   # purple
    BlockLabel.CAPTION: (90,  200, 250),   # cyan
    BlockLabel.LIST:    (255, 59,  48),    # red
    BlockLabel.FORMULA: (255, 204, 0),     # yellow
    BlockLabel.HEADER:  (100, 100, 100),   # grey
    BlockLabel.FOOTER:  (150, 150, 150),   # light grey
    BlockLabel.UNKNOWN: (200, 200, 200),   # very light grey
}


class BlockVisualizer:
    """
    Draws bounding-box overlays on page images and saves them as PNG.

    Args:
        output_dir: Directory where artifact PNGs will be saved.
        font_size:  Point size for the label text overlay.
    """

    def __init__(self, output_dir: str | Path = "artifacts", font_size: int = 14) -> None:
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._font_size = font_size

    def save_page_artifacts(
        self,
        page_image: Image.Image,
        blocks: list[Block],
        filename: str,
    ) -> Path:
        """
        Draw all blocks on a page image and save to artifacts/.

        Args:
            page_image: The raw rasterized page (PIL Image).
            blocks:     List of Block objects to draw (for this page only).
            filename:   Base filename, e.g. 'report_page_01.png'.

        Returns:
            Path to the saved artifact file.
        """
        annotated = self._draw_blocks(page_image.copy(), blocks)
        out_path = self._out / filename
        annotated.save(str(out_path), format="PNG")
        logger.debug(f"Artifact saved: {out_path}")
        return out_path

    def save_all_pages(
        self,
        pages: list[Image.Image],
        blocks: list[Block],
        doc_name: str,
    ) -> list[Path]:
        """
        Convenience method: generate one artifact per page.

        Args:
            pages:    Ordered list of page images (0-indexed).
            blocks:   All blocks from the document (any page).
            doc_name: Document base name for filenames.

        Returns:
            List of paths to saved artifact PNGs.
        """
        saved: list[Path] = []
        for page_num, page_img in enumerate(pages):
            page_blocks = [b for b in blocks if b.page == page_num]
            fname = f"{doc_name}_page_{page_num + 1:03d}.png"
            path = self.save_page_artifacts(page_img, page_blocks, fname)
            saved.append(path)
        logger.success(f"Saved {len(saved)} artifact images for '{doc_name}'.")
        return saved

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _draw_blocks(self, image: Image.Image, blocks: list[Block]) -> Image.Image:
        """Draw colored bounding boxes and labels onto a PIL Image."""
        draw = ImageDraw.Draw(image, "RGBA")

        # Try to load a system font; fall back to PIL default
        try:
            font = ImageFont.truetype("arial.ttf", self._font_size)
        except OSError:
            font = ImageFont.load_default()

        for block in blocks:
            color = _LABEL_COLORS.get(block.label, (200, 200, 200))
            r, g, b = color
            alpha = 50  # semi-transparent fill

            x1, y1, x2, y2 = (
                block.bbox.x1, block.bbox.y1,
                block.bbox.x2, block.bbox.y2,
            )

            # Semi-transparent fill
            draw.rectangle([(x1, y1), (x2, y2)], fill=(r, g, b, alpha))
            # Solid border
            draw.rectangle([(x1, y1), (x2, y2)], outline=(r, g, b, 255), width=2)

            # Label badge
            label_text = f"{block.label.value} {block.confidence:.0%}"
            text_bbox = draw.textbbox((x1 + 4, y1 + 2), label_text, font=font)
            draw.rectangle(text_bbox, fill=(r, g, b, 200))
            draw.text((x1 + 4, y1 + 2), label_text, fill=(255, 255, 255, 255), font=font)

        return image
