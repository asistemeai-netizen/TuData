# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Figure / Diagram Analyzer (Stage 2C)
===============================================
Analyzes FIGURE blocks using Gemini Vision with rich context-aware prompts.

Unlike the previous approach (generic "describe this image"), this analyzer:
  1. Passes the project type as context so Gemini knows what to look for.
  2. Requests structured output: figure type, components, technical specs, text labels.
  3. Separates the visual description from embedded text labels.

Output: FigureData with components, specs, labels — ready for the Consolidator.
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from PIL import Image

from src.models import Block, ProjectType


@dataclass
class FigureData:
    """Structured representation of an analyzed figure or diagram."""
    block_id: int
    page: int
    figure_type: str = "unknown"           # e.g. "single_line_diagram", "photo", "chart"
    description: str = ""                  # 1-2 sentence human-readable description
    components: list[str] = field(default_factory=list)       # named components/equipment
    technical_specs: dict[str, str] = field(default_factory=dict)  # voltage, amperage, etc.
    text_labels: list[str] = field(default_factory=list)       # text visible in diagram
    references: list[str] = field(default_factory=list)        # references to other pages/docs
    confidence: float = 1.0
    analysis_method: str = "gemini_vision"

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "page": self.page,
            "figure_type": self.figure_type,
            "description": self.description,
            "components": self.components,
            "technical_specs": self.technical_specs,
            "text_labels": self.text_labels,
            "references": self.references,
            "confidence": self.confidence,
        }

    def to_text(self) -> str:
        """Convert figure data to a descriptive text block for downstream LLM."""
        parts = [f"[FIGURE: {self.figure_type.replace('_', ' ').title()}]"]
        if self.description:
            parts.append(self.description)
        if self.components:
            parts.append("Components: " + ", ".join(self.components))
        if self.technical_specs:
            specs = ", ".join(f"{k}: {v}" for k, v in self.technical_specs.items())
            parts.append("Technical Specs: " + specs)
        if self.text_labels:
            parts.append("Labels: " + ", ".join(self.text_labels))
        if self.references:
            parts.append("References: " + ", ".join(self.references))
        return "\n".join(parts)


# ── Prompt templates per project type ────────────────────────────────────────

_BASE_PROMPT = """You are an expert document analyst specialized in {domain}.

Analyze this figure from page {page} of a {doc_type} document.

Return ONLY a JSON object with this format:
{{
  "figure_type": "<one of: single_line_diagram, wiring_diagram, floor_plan, site_plan, photo, chart, graph, table_image, map, schematic, other>",
  "description": "<1-2 sentences describing what this figure shows>",
  "components": ["<equipment or component names>"],
  "technical_specs": {{"<param>": "<value>"}},
  "text_labels": ["<text visible in the image>"],
  "references": ["<references to other docs/pages if visible>"],
  "confidence": 0.0-1.0
}}

Rules:
- components: list all named equipment (e.g. "ASCO 7000 ATS", "100A Main Breaker").
- technical_specs: extract electrical/mechanical values (voltage, amperage, kW, RPM, etc.).
- text_labels: ALL text visible in the diagram including tag names, panel IDs, etc.
- references: drawing numbers, spec references, page cross-refs (e.g. "See Sheet E-3").
- Return ONLY JSON, no commentary.
"""

_DOMAIN_MAP: dict[str, str] = {
    ProjectType.ELECTRICAL_SWITCHGEAR_QUOTE.value: "electrical switchgear and automatic transfer switches",
    ProjectType.ELECTRICAL_PANEL_QUOTE.value: "electrical panels and distribution systems",
    ProjectType.MECHANICAL_SPEC.value: "mechanical equipment (HVAC, pumps, chillers)",
    ProjectType.ARCHITECTURAL_DRAWING.value: "architectural drawings and floor plans",
    ProjectType.TECHNICAL_DATASHEET.value: "technical product specifications",
}


class FigureAnalyzer:
    """
    Analyzes figure blocks with Gemini Vision using context-aware prompts.
    
    Key improvement over the old approach:
    - Passes project type so Gemini recognizes industry-specific components.
    - Returns structured data (components, specs, labels) not just a text description.
    - 15px padding on image crop to avoid cutting diagram borders.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        key = api_key or os.getenv("GEMINI_API_KEY")
        self._client = genai.Client(api_key=key)
        self._model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._types = types
        logger.info(f"[FigureAnalyzer] Gemini Vision backend: {self._model}")

    def analyze(
        self,
        image: Image.Image,
        block: Block,
        project_type: ProjectType = ProjectType.GENERAL_DOCUMENT,
    ) -> FigureData:
        """
        Analyze a figure image and return structured FigureData.

        Args:
            image:        Cropped PIL Image of the figure block.
            block:        Source Block (for metadata).
            project_type: Detected project type for context-aware prompting.

        Returns:
            FigureData with components, specs, text labels, and references.
        """
        domain = _DOMAIN_MAP.get(project_type.value, "industrial/technical documents")
        doc_type = project_type.value.replace("_", " ")

        prompt = _BASE_PROMPT.format(
            domain=domain,
            page=block.page + 1,
            doc_type=doc_type,
        )
        img_bytes = self._pil_to_bytes(image, padding=15)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    prompt,
                    self._types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                ],
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=2048,
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text or "{}")
            fd = FigureData(
                block_id=block.id,
                page=block.page,
                figure_type=data.get("figure_type", "other"),
                description=data.get("description", ""),
                components=data.get("components", []),
                technical_specs=data.get("technical_specs", {}),
                text_labels=data.get("text_labels", []),
                references=data.get("references", []),
                confidence=float(data.get("confidence", 0.9)),
            )
            logger.info(
                f"[FigureAnalyzer] Block #{block.id} (page {block.page+1}): "
                f"type={fd.figure_type}, "
                f"components={len(fd.components)}, "
                f"specs={len(fd.technical_specs)}"
            )
            return fd

        except Exception as exc:
            logger.warning(f"[FigureAnalyzer] Failed for block #{block.id}: {exc}")
            return FigureData(
                block_id=block.id,
                page=block.page,
                description=block.text or "Figure analysis failed.",
                confidence=0.2,
                analysis_method="fallback",
            )

    @staticmethod
    def _pil_to_bytes(image: Image.Image, padding: int = 15) -> bytes:
        """Add padding and convert to JPEG bytes."""
        padded = Image.new("RGB", (image.width + padding*2, image.height + padding*2), "white")
        padded.paste(image, (padding, padding))
        buf = io.BytesIO()
        padded.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        return buf.read()
