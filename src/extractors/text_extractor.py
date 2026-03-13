# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Text Extractor (Stage 2A)
=====================================
Extracts text from TEXT, TITLE, LIST, CAPTION blocks.

Strategy:
  1. If the PDF is DIGITAL: use PyMuPDF native text extraction (fast, lossless).
  2. If the PDF is SCANNED: use PaddleOCR on the cropped block image (same as before).

This eliminates OCR overhead on PDFs that already have embedded text,
which is the single biggest performance improvement in the pipeline.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from loguru import logger
from PIL import Image

from src.models import Block, BlockLabel, DocumentFormat


class NativeTextExtractor:
    """
    Extracts text from a digital PDF using PyMuPDF's native text layer.
    
    For each Block, intersects the block's bounding box with PyMuPDF's
    text spans to retrieve text exactly as embedded in the PDF — no OCR needed.
    """

    def extract_from_page(
        self,
        page: fitz.Page,
        blocks: list[Block],
    ) -> list[Block]:
        """
        Extract native text for all blocks on a single PDF page.

        Args:
            page:   PyMuPDF page object.
            blocks: Blocks on this page (from ContentSplit.text_blocks).

        Returns:
            Same blocks with .text populated.
        """
        # Get all words with their bboxes from PyMuPDF
        words = page.get_text("words")  # (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        
        # Scale factor: blocks are in image pixels (200dpi), page coords are in pts
        # Must match the DPI used in LayoutDetector (v2: 200 DPI)
        scale = 200.0 / 72.0

        for block in blocks:
            # Convert block bbox from image pixels back to PDF points
            bx1 = block.bbox.x1 / scale
            by1 = block.bbox.y1 / scale
            bx2 = block.bbox.x2 / scale
            by2 = block.bbox.y2 / scale

            # Collect words that overlap with this block
            matched_words: list[str] = []
            for wx0, wy0, wx1, wy1, word, *_ in words:
                # Check overlap (allow partial overlap)
                if wx0 < bx2 and wx1 > bx1 and wy0 < by2 and wy1 > by1:
                    matched_words.append(word)

            block.text = " ".join(matched_words).strip() if matched_words else ""
            if block.text:
                logger.debug(
                    f"[TextExtractor:native] Block #{block.id} ({block.label.value}) "
                    f"→ {len(block.text)} chars"
                )

        return blocks


class OcrTextExtractor:
    """
    Fallback text extractor for scanned PDFs using PaddleOCR.
    Wraps the existing PaddleOCREngine for use in the new stream architecture.
    """

    def __init__(self, lang: str = "en") -> None:
        from src.ocr.ocr_engine import PaddleOCREngine
        self._engine = PaddleOCREngine(lang=lang)

    async def extract_blocks(
        self,
        blocks: list[Block],
        page_images: list[Image.Image],
    ) -> list[Block]:
        """
        OCR all text blocks asynchronously.

        Args:
            blocks:      List of text-type blocks.
            page_images: Rendered page images (one per page).

        Returns:
            Same blocks with .text populated.
        """
        loop = asyncio.get_running_loop()

        async def ocr_one(block: Block) -> Block:
            page_img = page_images[block.page]
            crop = self._crop(page_img, block)
            block.text = await loop.run_in_executor(
                None, self._engine.extract_text, crop
            )
            logger.debug(
                f"[TextExtractor:OCR] Block #{block.id} → {len(block.text or '')} chars"
            )
            return block

        return list(await asyncio.gather(*[ocr_one(b) for b in blocks]))

    @staticmethod
    def _crop(page_img: Image.Image, block: Block) -> Image.Image:
        x1 = max(0, int(block.bbox.x1) - 10)
        y1 = max(0, int(block.bbox.y1) - 10)
        x2 = min(page_img.width,  int(block.bbox.x2) + 10)
        y2 = min(page_img.height, int(block.bbox.y2) + 10)
        return page_img.crop((x1, y1, x2, y2))
