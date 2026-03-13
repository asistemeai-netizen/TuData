# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Document Processing Pipeline
======================================
Shared data models used across all pipeline stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BlockLabel(str, Enum):
    """Semantic category for a detected layout block."""
    TEXT = "Text"
    TITLE = "Title"
    TABLE = "Table"
    FIGURE = "Figure"
    CAPTION = "Caption"
    LIST = "List"
    FORMULA = "Formula"
    HEADER = "Header"
    FOOTER = "Footer"
    UNKNOWN = "Unknown"


class ProjectType(str, Enum):
    """High-level project/document classification used to drive the pipeline."""
    ELECTRICAL_SWITCHGEAR_QUOTE = "electrical_switchgear_quote"
    ELECTRICAL_PANEL_QUOTE      = "electrical_panel_quote"
    MECHANICAL_SPEC             = "mechanical_spec"
    ARCHITECTURAL_DRAWING       = "architectural_drawing"
    MATERIAL_INVOICE            = "material_invoice"
    PROJECT_REPORT              = "project_report"
    FORM                        = "form"
    TECHNICAL_DATASHEET         = "technical_datasheet"
    GENERAL_DOCUMENT            = "general_document"   # fallback


class DocumentFormat(str, Enum):
    """Whether the PDF has embedded text (digital) or needs OCR (scanned)."""
    DIGITAL  = "digital"
    SCANNED  = "scanned"
    MIXED    = "mixed"


@dataclass
class BoundingBox:
    """Pixel-space bounding box: (x1, y1, x2, y2) — top-left origin."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def x_center(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def y_center(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Block:
    """A single detected layout region on a document page."""
    id: int
    label: BlockLabel
    bbox: BoundingBox
    page: int
    confidence: float = 1.0
    text: Optional[str] = None          # populated after OCR stage
    column: Optional[int] = None        # populated after reading-order stage
    source_file: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label.value,
            "bbox": {
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
                "x2": self.bbox.x2,
                "y2": self.bbox.y2,
            },
            "page": self.page,
            "confidence": round(self.confidence, 4),
            "text": self.text,
            "column": self.column,
        }


@dataclass
class IntakeResult:
    """Output of Stage 0: document metadata before any processing."""
    page_count: int
    doc_format: DocumentFormat          # digital / scanned / mixed
    project_type: ProjectType           # classified project type
    language: str = "en"
    classification_confidence: float = 1.0
    digital_page_ratio: float = 1.0    # fraction of pages with embedded text
    notes: list[str] = field(default_factory=list)
