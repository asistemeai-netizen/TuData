# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Gemini Vision OCR (Fallback Engine)
=============================================
Sends cropped block images to Gemini for text extraction.
Now uses the google-genai SDK (replaces deprecated google-generativeai).
"""
from __future__ import annotations

import base64
import io
import os
from typing import Optional

from loguru import logger
from PIL import Image


# Prompt templates for different block types
_PROMPTS = {
    "default": (
        "Extract all text from this document block exactly as it appears. "
        "Preserve line breaks where meaningful. "
        "Return ONLY the raw text, no commentary."
    ),
    "table": (
        "This image contains a table. Convert it to Markdown table format. "
        "Preserve all rows and columns accurately. "
        "Return ONLY the Markdown table, no commentary."
    ),
    "formula": (
        "This image contains a mathematical or chemical formula. "
        "Represent it in LaTeX notation surrounded by $$ delimiters. "
        "Return ONLY the LaTeX, no commentary."
    ),
    "figure": (
        "Describe the content of this figure in 1-2 sentences, then extract "
        "any embedded text or labels. Return in format: 'Description: ... | Text: ...'"
    ),
}


class GeminiVisionOCR:
    """
    Gemini Vision OCR using the google-genai SDK.

    Args:
        api_key:    Gemini API key. Defaults to GEMINI_API_KEY env var.
        model:      Gemini model name. Defaults to GEMINI_MODEL env var.
        max_tokens: Maximum tokens in the response.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> None:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Add it to .env or pass api_key= directly."
            )

        self._client = genai.Client(api_key=self._api_key)
        self._model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._max_tokens = max_tokens
        self._types = types
        logger.info(f"Gemini Vision OCR initialized (model={self._model}).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(
        self,
        image: Image.Image,
        block_type: str = "default",
    ) -> str:
        """
        Send an image block to Gemini for text extraction.

        Args:
            image:      Cropped PIL Image of the block.
            block_type: One of 'default', 'table', 'formula', 'figure'.

        Returns:
            Extracted text string (Markdown for tables, LaTeX for formulas).
        """
        prompt = _PROMPTS.get(block_type, _PROMPTS["default"])
        img_bytes = self._pil_to_bytes(image)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    prompt,
                    self._types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                ],
                config=self._types.GenerateContentConfig(
                    max_output_tokens=self._max_tokens,
                ),
            )
            text = response.text.strip() if response.text else ""
            logger.debug(f"Gemini OCR ({block_type}): {len(text)} chars returned.")
            return text
        except Exception as exc:
            logger.warning(f"Gemini Vision OCR failed: {exc}")
            return ""

    @staticmethod
    def _pil_to_bytes(image: Image.Image) -> bytes:
        """Convert PIL Image to JPEG bytes."""
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        return buffer.read()
