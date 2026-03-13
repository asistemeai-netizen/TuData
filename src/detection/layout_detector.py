# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Layout Detector
=========================
Replicates Surya's document layout detection stage.

Uses a YOLOv8 model trained on DocLayNet to identify:
  Text, Title, Table, Figure, Caption, List, Formula, Header, Footer

Input:  PDF path (str)
Output: list[Block] — one Block per detected region across all pages
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from loguru import logger
from PIL import Image

from src.models import Block, BlockLabel, BoundingBox

# DocLayNet label index → BlockLabel mapping (YOLOv8 class order for hantian/yolo-doclaynet)
_DOCLAYNET_LABELS: dict[int, BlockLabel] = {
    0: BlockLabel.CAPTION,
    1: BlockLabel.FOOTER,        # Footnote
    2: BlockLabel.FORMULA,
    3: BlockLabel.LIST,
    4: BlockLabel.FOOTER,        # Page-footer
    5: BlockLabel.HEADER,        # Page-header
    6: BlockLabel.FIGURE,        # Picture
    7: BlockLabel.TITLE,         # Section-header
    8: BlockLabel.TABLE,
    9: BlockLabel.TEXT,
    10: BlockLabel.TITLE,        # Title
}

# Fallback to generic YOLO label strings if a custom model is used
_LABEL_STRING_MAP: dict[str, BlockLabel] = {l.value.lower(): l for l in BlockLabel}


class LayoutDetector:
    """
    Loads a YOLOv8 model and runs per-page layout inference on a PDF.

    Args:
        model_path: Path to a local .pt weight file OR a HuggingFace model ID.
                    Defaults to 'yolov8x-doclaynet.pt' (auto-downloaded by ultralytics).
        confidence: Detection confidence threshold (0–1).
        dpi:        PDF rendering resolution in dots-per-inch.
    """

    DEFAULT_MODEL = "yolov8s-doclaynet.pt"
    MODEL_URL = "https://huggingface.co/hantian/yolo-doclaynet/resolve/main/yolov8s-doclaynet.pt"

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.35,
        dpi: int = 200,       # v2: increased from 150 for better technical diagram detection
    ) -> None:
        # Lazy import to avoid slow startup when module is imported but not used
        from ultralytics import YOLO  # type: ignore

        self._model_path = model_path or self.DEFAULT_MODEL
        if self._model_path == self.DEFAULT_MODEL and not os.path.exists(self._model_path):
            self._download_model()

        self._confidence = confidence
        self._dpi = dpi

        logger.info(f"Loading layout model: {self._model_path}")
        self._model = YOLO(self._model_path)
        logger.success("Layout model loaded.")

    def _download_model(self) -> None:
        import urllib.request
        logger.info(f"Downloading {self.DEFAULT_MODEL} from HuggingFace...")
        try:
            urllib.request.urlretrieve(self.MODEL_URL, self.DEFAULT_MODEL)
            logger.success("Model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_pdf(self, pdf_path: str | Path) -> list[Block]:
        """
        Detect all layout blocks in a PDF.

        Returns a flat list of Block objects sorted by (page, y1, x1).
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Detecting layout in: {pdf_path.name}")
        doc = fitz.open(str(pdf_path))

        all_blocks: list[Block] = []
        block_id = 0

        for page_num, page in enumerate(doc):
            logger.debug(f"  Processing page {page_num + 1}/{len(doc)}")
            image = self._render_page(page)
            page_blocks = self._detect_page(image, page_num, block_id, str(pdf_path.name))
            all_blocks.extend(page_blocks)
            block_id += len(page_blocks)

        page_count = len(doc)
        doc.close()
        logger.success(f"Detected {len(all_blocks)} blocks across {page_count} pages.")
        return sorted(all_blocks, key=lambda b: (b.page, b.bbox.y1, b.bbox.x1))

    def detect_image(self, image: Image.Image, page: int = 0) -> list[Block]:
        """Detect layout blocks in a single PIL Image (useful for testing)."""
        return self._detect_page(image, page, block_id=0, source="<image>")

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _render_page(self, page: fitz.Page) -> Image.Image:
        """Render a PDF page to a PIL Image at the configured DPI."""
        zoom = self._dpi / 72.0  # 72 dpi is PyMuPDF default
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def _detect_page(
        self,
        image: Image.Image,
        page: int,
        block_id: int,
        source: str,
    ) -> list[Block]:
        """Run YOLO inference on a single page image and return Block list."""
        results = self._model(image, conf=self._confidence, verbose=False)
        blocks: list[Block] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                label = _DOCLAYNET_LABELS.get(cls_id)

                # Fall back to string-based mapping for custom models
                if label is None:
                    cls_name = result.names.get(cls_id, "unknown").lower()
                    label = _LABEL_STRING_MAP.get(cls_name, BlockLabel.UNKNOWN)

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf.item())

                block = Block(
                    id=block_id,
                    label=label,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    page=page,
                    confidence=conf,
                    source_file=source,
                )
                blocks.append(block)
                logger.debug(f"Detected block {block.label.value} (conf: {block.confidence:.2f}) at [x1={block.bbox.x1:.1f}, y1={block.bbox.y1:.1f}, x2={block.bbox.x2:.1f}, y2={block.bbox.y2:.1f}]")
                block_id += 1

        return blocks
