# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Project Classifier (Stage 0)
========================================
Classifies a document into a ProjectType using Gemini Flash.

Input:  text_sample (first ~2000 chars) + block_type_hints
Output: ProjectType enum + confidence float

This classification drives the rest of the pipeline:
  - Which extractors to prioritize
  - Which LLM prompt template to use in the Consolidator
  - Which Knowledge Base categories to query
"""
from __future__ import annotations

import json
import os
from typing import Optional

from loguru import logger

from src.models import ProjectType


# ── Classification prompt ─────────────────────────────────────────────────────

_CLASSIFY_PROMPT = """You are a document classification expert for construction and industrial projects.

Analyze the following text sample (from the first pages of a document) and classify the document type.

Choose EXACTLY ONE of these categories:
- electrical_switchgear_quote   (switchgear equipment quotes, ATS, panels, breakers)
- electrical_panel_quote        (general electrical panel or MCC quotes)
- mechanical_spec               (mechanical equipment specs, HVAC, pumps, chillers)
- architectural_drawing         (floor plans, elevations, site plans)
- material_invoice              (supplier invoice or purchase order)
- project_report                (project status, engineer reports, RFIs)
- form                          (data entry form, submittal form)
- technical_datasheet           (product datasheet, cut sheet)
- general_document              (anything else)

TEXT SAMPLE:
{text_sample}

Additional signals:
- Has images/diagrams: {has_images}
- Pages: {page_count}
- Digital PDF (has embedded text): {is_digital}

Return a JSON object ONLY:
{{"project_type": "<category>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}
"""


class ProjectClassifier:
    """
    Classifies incoming documents into a ProjectType using Gemini Flash.

    Falls back to heuristic keyword matching if Gemini is unavailable.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._model = model or os.getenv("GEMINI_CLASSIFY_MODEL", "gemini-2.0-flash")
        self._client = None
        self._types = None

        if self._api_key:
            try:
                from google import genai  # type: ignore
                from google.genai import types  # type: ignore
                self._client = genai.Client(api_key=self._api_key)
                self._types = types
                logger.info(f"[Classifier] Gemini backend: {self._model}")
            except Exception as exc:
                logger.warning(f"[Classifier] Gemini unavailable, using heuristic fallback: {exc}")

    def classify(
        self,
        text_sample: str,
        page_count: int = 1,
        has_images: bool = False,
        is_digital: bool = True,
    ) -> tuple[ProjectType, float]:
        """
        Classify a document into a ProjectType.

        Returns:
            (project_type, confidence)
        """
        if self._client:
            return self._classify_with_gemini(text_sample, page_count, has_images, is_digital)
        return self._classify_heuristic(text_sample)

    # ── Gemini-based classification ────────────────────────────────────────────

    def _classify_with_gemini(
        self,
        text_sample: str,
        page_count: int,
        has_images: bool,
        is_digital: bool,
    ) -> tuple[ProjectType, float]:
        prompt = _CLASSIFY_PROMPT.format(
            text_sample=text_sample[:2000],
            has_images=has_images,
            page_count=page_count,
            is_digital=is_digital,
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=256,
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text or "{}")
            raw_type = data.get("project_type", "general_document")
            confidence = float(data.get("confidence", 0.7))
            reasoning = data.get("reasoning", "")

            project_type = self._parse_type(raw_type)
            logger.info(
                f"[Classifier] → {project_type.value} "
                f"(conf={confidence:.2f}) — {reasoning}"
            )
            return project_type, confidence

        except Exception as exc:
            logger.warning(f"[Classifier] Gemini classify failed: {exc}. Using heuristic.")
            return self._classify_heuristic(text_sample)

    # ── Heuristic fallback ─────────────────────────────────────────────────────

    def _classify_heuristic(self, text: str) -> tuple[ProjectType, float]:
        """Keyword-based fallback classifier."""
        text_lower = text.lower()

        rules: list[tuple[list[str], ProjectType]] = [
            (["switchgear", "asco", "transfer switch", "ats", "swgr"], ProjectType.ELECTRICAL_SWITCHGEAR_QUOTE),
            (["panel", "circuit breaker", "mcc", "mlo", "400a", "800a", "1200a"], ProjectType.ELECTRICAL_PANEL_QUOTE),
            (["invoice", "purchase order", "ship to", "bill to", "unit price", "subtotal"], ProjectType.MATERIAL_INVOICE),
            (["floor plan", "elevation", "site plan", "drawing no", "scale 1:"], ProjectType.ARCHITECTURAL_DRAWING),
            (["chiller", "hvac", "pump", "compressor", "valve", "btu"], ProjectType.MECHANICAL_SPEC),
            (["datasheet", "specifications", "part no", "catalog", "model no"], ProjectType.TECHNICAL_DATASHEET),
            (["report", "summary", "status", "rfi", "submittal"], ProjectType.PROJECT_REPORT),
            (["form", "signature", "date:", "name:", "address:"], ProjectType.FORM),
        ]

        for keywords, project_type in rules:
            if any(kw in text_lower for kw in keywords):
                logger.info(f"[Classifier] Heuristic → {project_type.value}")
                return project_type, 0.6

        return ProjectType.GENERAL_DOCUMENT, 0.5

    @staticmethod
    def _parse_type(raw: str) -> ProjectType:
        """Map a raw string to ProjectType, defaulting to GENERAL_DOCUMENT."""
        try:
            return ProjectType(raw)
        except ValueError:
            return ProjectType.GENERAL_DOCUMENT
