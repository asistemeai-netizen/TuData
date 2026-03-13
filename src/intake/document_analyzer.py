# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Document Analyzer (Stage 0)
======================================
Analyzes a PDF before processing to determine:
  1. Whether it is a digital (text-embedded) or scanned PDF.
  2. Page count and basic stats.
  3. Block-type distribution (used as features for project classification).

This avoids running OCR on PDFs that already have embedded text, which is
the single biggest performance win in the pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from loguru import logger

from src.models import DocumentFormat, IntakeResult, ProjectType


# Minimum char density (chars / page area pixels²) to consider a page "digital"
_DIGITAL_THRESHOLD = 0.0005


class DocumentAnalyzer:
    """
    Fast pre-analysis of a PDF before running the heavy pipeline.

    Returns an IntakeResult with:
      - doc_format: DIGITAL / SCANNED / MIXED
      - digital_page_ratio: fraction of pages with embedded text
      - page_count
      - language (heuristic from PyMuPDF metadata)
      - block_type_hints: rough content profile for project classification
    """

    def analyze(self, pdf_path: str | Path) -> dict:
        """
        Analyze a PDF file and return a profile dictionary.

        Returns:
            {
              'page_count': int,
              'doc_format': DocumentFormat,
              'digital_page_ratio': float,
              'language': str,
              'text_sample': str,         # first ~2000 chars for classifier
              'has_tables': bool,
              'has_figures': bool,
              'avg_chars_per_page': float,
              'metadata': dict            # raw PDF metadata
            }
        """
        pdf_path = Path(pdf_path)
        logger.info(f"[Intake] Analyzing: {pdf_path.name}")

        doc = fitz.open(str(pdf_path))
        page_count = len(doc)

        digital_pages = 0
        total_chars = 0
        text_sample_parts: list[str] = []
        has_images = False
        pdf_meta = doc.metadata or {}

        for page_num, page in enumerate(doc):
            # Get native text blocks
            text = page.get_text("text").strip()
            char_count = len(text)
            total_chars += char_count

            # Heuristic: pages with >50 chars are likely digital
            if char_count > 50:
                digital_pages += 1

            # Collect text sample from first 5 pages for classifier (v2: expanded from 3)
            if page_num < 5 and text:
                text_sample_parts.append(text[:1000])

            # Check for raster images (signals scanned or diagram-heavy)
            img_list = page.get_images(full=False)
            if img_list:
                has_images = True

        doc.close()

        digital_ratio = digital_pages / max(1, page_count)
        avg_chars = total_chars / max(1, page_count)

        if digital_ratio >= 0.9:
            doc_format = DocumentFormat.DIGITAL
        elif digital_ratio <= 0.1:
            doc_format = DocumentFormat.SCANNED
        else:
            doc_format = DocumentFormat.MIXED

        text_sample = "\n---\n".join(text_sample_parts)[:2000]

        # Infer language from PyMuPDF metadata or text (basic heuristic)
        language = pdf_meta.get("language", "en") or "en"

        result = {
            "page_count": page_count,
            "doc_format": doc_format,
            "digital_page_ratio": round(digital_ratio, 4),
            "avg_chars_per_page": round(avg_chars, 1),
            "language": language,
            "text_sample": text_sample,
            "has_images": has_images,
            "metadata": pdf_meta,
        }

        logger.info(
            f"[Intake] {pdf_path.name}: {page_count}p, "
            f"format={doc_format.value}, "
            f"digital_ratio={digital_ratio:.0%}, "
            f"avg_chars={avg_chars:.0f}/page"
        )
        return result
