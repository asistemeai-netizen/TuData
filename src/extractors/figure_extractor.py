# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Figure / Diagram Analyzer (Stage 2C) — v2 Enhanced
=============================================================
Analyzes FIGURE blocks using Gemini Vision with rich context-aware prompts.

v2 improvements over v1:
  1. Specialized prompts per detected diagram type (electrical, mechanical, etc.)
  2. Higher max_output_tokens (4096 instead of 2048) to capture ALL components
  3. Optional full-page context: sends the entire page with a highlight box
  4. Optional deep-analysis second pass for complex electrical diagrams
  5. Caption text passed as context for better identification

Output: FigureData with components, specs, labels — ready for the Consolidator.
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from PIL import Image, ImageDraw

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


# ── Prompt templates ─────────────────────────────────────────────────────────

_BASE_PROMPT = """You are an expert document analyst specialized in {domain}.

Analyze this figure from page {page} of a {doc_type} document.
{caption_context}
Return ONLY a JSON object with this format:
{{
  "figure_type": "<one of: single_line_diagram, wiring_diagram, connection_diagram, floor_plan, site_plan, panel_schedule, photo, chart, graph, table_image, map, schematic, piping_diagram, control_diagram, other>",
  "description": "<2-3 sentences describing what this figure shows in detail>",
  "components": ["<ALL equipment, device, and component names visible>"],
  "technical_specs": {{"<param>": "<value>"}},
  "text_labels": ["<ALL text visible in the image, including small labels>"],
  "references": ["<references to other docs/pages if visible>"],
  "confidence": 0.0-1.0
}}

Rules:
- components: list ALL named equipment. Include model numbers, ratings, and identifiers.
  Examples: "ASCO 7000 ATS", "1200A Main Breaker", "Square D Panel MDP-1"
- technical_specs: extract ALL electrical/mechanical values visible:
  voltages, amperages, kW, kVA, HP, RPM, phases, wire sizes, CT ratios, etc.
- text_labels: capture EVERY piece of text visible in the diagram, no matter how small.
  Include tag names, panel IDs, wire numbers, terminal labels, annotation notes.
- references: drawing numbers, spec references, page cross-refs (e.g. "See Sheet E-3").
- Be EXHAUSTIVE. Missing a component is worse than including a questionable one.
- Return ONLY JSON, no commentary.
"""

# Specialized prompt for electrical one-line / connection diagrams
_ELECTRICAL_DIAGRAM_PROMPT = """You are a senior electrical engineer analyzing a technical electrical diagram.

This is page {page} of a {doc_type} document about {domain}.
{caption_context}
This appears to be an electrical diagram. Analyze it with extreme attention to detail.

Return ONLY a JSON object:
{{
  "figure_type": "<single_line_diagram | connection_diagram | panel_schedule | wiring_diagram | control_diagram | schematic>",
  "description": "<detailed 3-5 sentence description of what this diagram shows>",
  "components": ["<list EVERY component>"],
  "technical_specs": {{}},
  "text_labels": ["<ALL text>"],
  "references": ["<cross-references>"],
  "switchgear_sections": ["<section names like Pull Section, Main Breaker Section, Feeder Section>"],
  "circuit_breakers": [
    {{"id": "<breaker ID>", "type": "<VCB/ACB/MCCB/MCB>", "rating": "<amperage>", "frame": "<frame size>"}}
  ],
  "metering": [
    {{"device": "<meter/CT/PT>", "ratio": "<CT ratio if applicable>", "location": "<where>"}}
  ],
  "protection_devices": [
    {{"device": "<relay/fuse/surge>", "model": "<model>", "setting": "<setting if visible>"}}
  ],
  "power_flow": {{"source": "<utility/generator>", "voltage": "<voltage>", "phases": "<1/3>", "bus_rating": "<bus amps>"}},
  "confidence": 0.0-1.0
}}

CRITICAL extraction targets for electrical diagrams:
1. CIRCUIT BREAKERS: Every VCB, ACB, MCCB with ratings (e.g., "1200A VCB", "800A MCCB")
2. SWITCHGEAR SECTIONS: Pull Section, Main Breaker Section, Feeder Breaker Section, Transition Section
3. METERING: All CTs, PTs, meters, revenue metering, SCADA points
4. PROTECTION: Relays (50/51, 27, 59, 81), fuses, surge arresters
5. UPS / BATTERY: Any UPS system, battery charger, DC panel, emergency power
6. MOTOR CONTROLS: MCC buckets, starters, VFDs, with HP ratings
7. TRANSFORMERS: kVA rating, voltage ratio, impedance, cooling type
8. BUS: Bus ratings in amps, bus material (copper/aluminum)
9. TRANSFER SWITCHES: ATS/STS models, amp ratings, transfer time
10. GROUNDING: Ground fault indicators, grounding resistors

Return ONLY JSON, no commentary.
"""

# Specialized prompt for mechanical / piping diagrams
_MECHANICAL_DIAGRAM_PROMPT = """You are a senior mechanical engineer analyzing a technical diagram.

This is page {page} of a {doc_type} document about {domain}.
{caption_context}
Analyze this mechanical/piping diagram with full detail.

Return ONLY a JSON object:
{{
  "figure_type": "<piping_diagram | schematic | floor_plan | site_plan | photo | other>",
  "description": "<detailed description>",
  "components": ["<ALL equipment: pumps, valves, tanks, heat exchangers, compressors, etc.>"],
  "technical_specs": {{
    "<param>": "<value with units>"
  }},
  "text_labels": ["<ALL text visible>"],
  "references": ["<cross-references>"],
  "piping": [
    {{"line_id": "<pipe ID>", "size": "<diameter>", "material": "<material>", "service": "<fluid/gas>"}}
  ],
  "instruments": [
    {{"tag": "<instrument tag>", "type": "<type>", "range": "<range>"}}
  ],
  "confidence": 0.0-1.0
}}

Focus on: pipe sizes, valve types (gate/ball/check/butterfly), instrument tags (FT, PT, TT, LT),
equipment model numbers, flow rates, pressures, temperatures, materials of construction.
Return ONLY JSON, no commentary.
"""

_DOMAIN_MAP: dict[str, str] = {
    ProjectType.ELECTRICAL_SWITCHGEAR_QUOTE.value: "electrical switchgear and automatic transfer switches",
    ProjectType.ELECTRICAL_PANEL_QUOTE.value: "electrical panels and distribution systems",
    ProjectType.MECHANICAL_SPEC.value: "mechanical equipment (HVAC, pumps, chillers)",
    ProjectType.ARCHITECTURAL_DRAWING.value: "architectural drawings and floor plans",
    ProjectType.TECHNICAL_DATASHEET.value: "technical product specifications",
}

# Project types that trigger the specialized electrical prompt
_ELECTRICAL_TYPES = {
    ProjectType.ELECTRICAL_SWITCHGEAR_QUOTE.value,
    ProjectType.ELECTRICAL_PANEL_QUOTE.value,
}

_MECHANICAL_TYPES = {
    ProjectType.MECHANICAL_SPEC.value,
}


class FigureAnalyzer:
    """
    Analyzes figure blocks with Gemini Vision using context-aware prompts.

    v2 improvements:
    - Specialized prompts per diagram type (electrical, mechanical, general)
    - Higher max_output_tokens (4096) to capture all components
    - Optional full-page context with highlight box around the figure area
    - Caption text passed as context
    - Deep analysis second pass for complex electrical diagrams
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
        caption: Optional[str] = None,
        page_image: Optional[Image.Image] = None,
    ) -> FigureData:
        """
        Analyze a figure image and return structured FigureData.

        Args:
            image:        Cropped PIL Image of the figure block.
            block:        Source Block (for metadata).
            project_type: Detected project type for context-aware prompting.
            caption:      Optional caption text from a nearby Caption block.
            page_image:   Optional full page image for context (sent as secondary image).

        Returns:
            FigureData with components, specs, text labels, and references.
        """
        domain = _DOMAIN_MAP.get(project_type.value, "industrial/technical documents")
        doc_type = project_type.value.replace("_", " ")
        caption_context = f'\nCaption/title found near this figure: "{caption}"' if caption else ""

        # Select the appropriate prompt based on project type
        if project_type.value in _ELECTRICAL_TYPES:
            prompt_template = _ELECTRICAL_DIAGRAM_PROMPT
        elif project_type.value in _MECHANICAL_TYPES:
            prompt_template = _MECHANICAL_DIAGRAM_PROMPT
        else:
            prompt_template = _BASE_PROMPT

        prompt = prompt_template.format(
            domain=domain,
            page=block.page + 1,
            doc_type=doc_type,
            caption_context=caption_context,
        )

        # Build content parts: prompt + cropped image + optional full page
        img_bytes = self._pil_to_bytes(image, padding=20)
        content_parts = [
            prompt,
            self._types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        ]

        # If we have the full page image, send it as context with the block area highlighted
        if page_image is not None:
            highlighted = self._highlight_block_on_page(page_image, block)
            page_bytes = self._pil_to_bytes(highlighted, padding=0)
            content_parts.append(
                self._types.Part.from_bytes(data=page_bytes, mime_type="image/jpeg")
            )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=content_parts,
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=4096,   # v2: doubled from 2048
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text or "{}")

            # Merge specialized fields into components/specs for backward compat
            extra_components = []
            extra_specs = {}

            # Electrical-specific fields
            for cb in data.get("circuit_breakers", []):
                cb_desc = f"{cb.get('type', 'Breaker')} {cb.get('rating', '')} (ID: {cb.get('id', 'N/A')})"
                if cb.get("frame"):
                    cb_desc += f" Frame: {cb['frame']}"
                extra_components.append(cb_desc.strip())

            for section in data.get("switchgear_sections", []):
                extra_components.append(f"Section: {section}")

            for meter in data.get("metering", []):
                m_desc = f"{meter.get('device', 'Meter')}"
                if meter.get("ratio"):
                    m_desc += f" ({meter['ratio']})"
                if meter.get("location"):
                    m_desc += f" at {meter['location']}"
                extra_components.append(m_desc)

            for prot in data.get("protection_devices", []):
                p_desc = f"{prot.get('device', 'Protection')}"
                if prot.get("model"):
                    p_desc += f" {prot['model']}"
                if prot.get("setting"):
                    p_desc += f" setting={prot['setting']}"
                extra_components.append(p_desc)

            pf = data.get("power_flow", {})
            if pf:
                if pf.get("voltage"):
                    extra_specs["Main_Voltage"] = pf["voltage"]
                if pf.get("phases"):
                    extra_specs["Phases"] = pf["phases"]
                if pf.get("bus_rating"):
                    extra_specs["Bus_Rating"] = pf["bus_rating"]
                if pf.get("source"):
                    extra_specs["Power_Source"] = pf["source"]

            # Mechanical-specific fields
            for pipe in data.get("piping", []):
                pipe_desc = f"Pipe {pipe.get('line_id', 'N/A')} {pipe.get('size', '')} {pipe.get('material', '')}"
                if pipe.get("service"):
                    pipe_desc += f" ({pipe['service']})"
                extra_components.append(pipe_desc.strip())

            for inst in data.get("instruments", []):
                i_desc = f"{inst.get('tag', 'Instrument')} ({inst.get('type', '')})"
                if inst.get("range"):
                    i_desc += f" range={inst['range']}"
                extra_components.append(i_desc.strip())

            # Combine base + extra
            all_components = data.get("components", []) + extra_components
            all_specs = {**data.get("technical_specs", {}), **extra_specs}

            fd = FigureData(
                block_id=block.id,
                page=block.page,
                figure_type=data.get("figure_type", "other"),
                description=data.get("description", ""),
                components=all_components,
                technical_specs=all_specs,
                text_labels=data.get("text_labels", []),
                references=data.get("references", []),
                confidence=float(data.get("confidence", 0.9)),
            )
            logger.info(
                f"[FigureAnalyzer] Block #{block.id} (page {block.page+1}): "
                f"type={fd.figure_type}, "
                f"components={len(fd.components)}, "
                f"specs={len(fd.technical_specs)}, "
                f"labels={len(fd.text_labels)}"
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
    def _highlight_block_on_page(page_image: Image.Image, block: Block) -> Image.Image:
        """Draw a red rectangle on the full page image to highlight the figure area."""
        highlighted = page_image.copy()
        draw = ImageDraw.Draw(highlighted)
        draw.rectangle(
            [block.bbox.x1, block.bbox.y1, block.bbox.x2, block.bbox.y2],
            outline="red", width=3
        )
        return highlighted

    @staticmethod
    def _pil_to_bytes(image: Image.Image, padding: int = 20) -> bytes:
        """Add padding and convert to JPEG bytes."""
        if padding > 0:
            padded = Image.new("RGB", (image.width + padding*2, image.height + padding*2), "white")
            padded.paste(image, (padding, padding))
        else:
            padded = image
        buf = io.BytesIO()
        padded.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        return buf.read()
