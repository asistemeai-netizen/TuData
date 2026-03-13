# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — PaddleOCR Engine (Primary OCR)
=========================================
Replicates the Chandra OCR stage using PaddleOCR 3.x (PP-OCRv5).

Accepts cropped PIL Images per block and returns extracted text strings.
Handles automatic deskew via OpenCV for rotated/skewed scans.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from PIL import Image


class PaddleOCREngine:
    """
    Primary OCR engine powered by PaddleOCR 3.x (PP-OCRv5).

    Args:
        lang:        Language code ('en', 'es', 'ch', etc.)
        use_angle:   Enable automatic text orientation correction.
        use_gpu:     Use GPU if available. Defaults to CPU-only.
        min_conf:    Minimum word-level confidence to include in output.
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle: bool = True,
        use_gpu: bool = False,
        min_conf: float = 0.5,
    ) -> None:
        import os
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCR  # type: ignore

        self._min_conf = min_conf
        logger.info(f"Initializing PaddleOCR (lang={lang}, use_gpu={use_gpu}) ...")
        
        self._ocr = PaddleOCR(
            use_angle_cls=use_angle,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,
        )
        logger.success("PaddleOCR initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(self, image: Image.Image) -> str:
        """
        Run OCR on a single-block PIL Image.

        Returns cleaned, concatenated text string.
        Returns empty string if no text is detected.
        """
        image = self._deskew(image)
        img_array = np.array(image.convert("RGB"))

        result = self._ocr.ocr(img_array, cls=True)

        if not result or result[0] is None:
            return ""

        lines: list[str] = []
        for line in result[0]:
            ext_text, conf = line[1]
            if conf >= self._min_conf and ext_text.strip():
                lines.append(ext_text.strip())

        return " ".join(lines)

    def extract_blocks(self, image: Image.Image) -> list[dict]:
        """
        Run OCR and return raw word-level data dicts for advanced use.
        """
        image = self._deskew(image)
        img_array = np.array(image.convert("RGB"))
        result = self._ocr.ocr(img_array, cls=True)

        words: list[dict] = []
        if not result or result[0] is None:
            return words

        for line in result[0]:
            bbox, (ext_text, conf) = line
            if conf < self._min_conf:
                continue
            pts = np.array(bbox)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            words.append({
                "text": ext_text.strip(),
                "confidence": round(float(conf), 4),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })
        return words

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _deskew(self, image: Image.Image) -> Image.Image:
        """
        Correct minor skew in a cropped block image using OpenCV moments.
        Skips deskew if the angle is within ±1 degree (already straight).
        """
        img_gray = np.array(image.convert("L"))
        thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        # Normalize angle: OpenCV returns values in (-90, 0]
        if angle < -45:
            angle = 90 + angle

        if abs(angle) <= 1.0:  # don't rotate nearly-straight images
            return image

        (h, w) = img_gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_np = cv2.warpAffine(
            np.array(image.convert("RGB")), M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(rotated_np)
