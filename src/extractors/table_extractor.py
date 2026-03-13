# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Table Extractor (Stage 2B)
======================================
Extracts tables from TABLE blocks with maximum precision.

Strategy:
  - DIGITAL PDFs: Camelot or tabula-py for direct table extraction from PDF vector data.
  - SCANNED PDFs: Gemini Vision in JSON mode — returns rows/columns as structured JSON,
    NOT Markdown, to preserve cell boundaries.

Output per table block: structured TableData with headers + rows.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger
from PIL import Image

from src.models import Block


@dataclass
class TableData:
    """Structured representation of a single extracted table."""
    block_id: int
    page: int
    caption: Optional[str] = None
    headers: list[str] = field(default_factory=list)
    rows: list[dict[str, str]] = field(default_factory=list)
    raw_markdown: Optional[str] = None   # fallback if structured extraction fails
    extraction_method: str = "unknown"
    confidence: float = 1.0

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return len(self.headers)

    def to_markdown(self) -> str:
        """Render the table as a Markdown string."""
        if self.raw_markdown:
            return self.raw_markdown
        if not self.headers:
            return ""
        header_row = "| " + " | ".join(self.headers) + " |"
        separator  = "| " + " | ".join(["---"] * len(self.headers)) + " |"
        body_rows  = [
            "| " + " | ".join(row.get(h, "") for h in self.headers) + " |"
            for row in self.rows
        ]
        return "\n".join([header_row, separator] + body_rows)

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "page": self.page,
            "caption": self.caption,
            "headers": self.headers,
            "rows": self.rows,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
        }


# ── Gemini Vision Table Extractor (primary for scanned) ─────────────────────

_TABLE_JSON_PROMPT = """You are a table extraction expert analyzing a document image.

This table is from a document of type: {doc_type}.
Page {page} of the document.

Extract the complete table data and return ONLY a JSON object with this exact format:
{{
  "caption": "<table caption if visible, or null>",
  "headers": ["col1", "col2", "col3"],
  "rows": [
    {{"col1": "value", "col2": "value", "col3": "value"}},
    ...
  ],
  "confidence": 0.0-1.0
}}

Rules:
- Headers: use the first row if it looks like a header, otherwise infer from context.
- Preserve ALL rows including totals, subtotals, notes rows.
- For monetary values, preserve the exact format ($1,234.56).
- For merged cells, repeat the value in each applicable column.
- For empty cells, use an empty string "" (never null).
- If this is an electrical panel schedule, ensure you capture:
  circuit numbers, breaker sizes (amps), wire sizes, load descriptions, phases, poles.
- If this is a Bill of Materials (BOM), ensure you capture:
  item numbers, part numbers, descriptions, quantities, unit prices, totals.
- Return ONLY the JSON, no commentary.
"""


class GeminiTableExtractor:
    """
    Extracts tables from block images using Gemini Vision in JSON mode.
    Returns structured TableData instead of Markdown for maximum precision.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        import os
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        key = api_key or os.getenv("GEMINI_API_KEY")
        self._client = genai.Client(api_key=key)
        self._model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._types = types
        logger.info(f"[TableExtractor] Gemini Vision backend: {self._model}")

    def extract(
        self,
        image: Image.Image,
        block: Block,
        doc_type: str = "general",
    ) -> TableData:
        """
        Extract a table from a cropped block image.

        Args:
            image:    Cropped PIL Image of the table block.
            block:    The source Block (for metadata).
            doc_type: Project type hint to contextualize extraction.

        Returns:
            Structured TableData with headers and rows.
        """
        prompt = _TABLE_JSON_PROMPT.format(
            doc_type=doc_type,
            page=block.page + 1,
        )
        img_bytes = self._pil_to_bytes(image)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    prompt,
                    self._types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                ],
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=8192,   # v2: doubled from 4096 for large tables
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text or "{}")
            td = TableData(
                block_id=block.id,
                page=block.page,
                caption=data.get("caption"),
                headers=data.get("headers", []),
                rows=data.get("rows", []),
                confidence=float(data.get("confidence", 0.9)),
                extraction_method="gemini_vision_json",
            )
            logger.info(
                f"[TableExtractor] Block #{block.id} → "
                f"{td.col_count} cols × {td.row_count} rows "
                f"(conf={td.confidence:.2f})"
            )
            return td

        except Exception as exc:
            logger.warning(f"[TableExtractor] Gemini failed for block #{block.id}: {exc}")
            # Fallback: store the raw OCR text as markdown
            return TableData(
                block_id=block.id,
                page=block.page,
                raw_markdown=block.text or "",
                extraction_method="fallback_raw",
                confidence=0.3,
            )

    @staticmethod
    def _pil_to_bytes(image: Image.Image, padding: int = 15) -> bytes:
        """Add padding and convert PIL Image to JPEG bytes."""
        # Add white padding to avoid cropping table borders
        padded = Image.new("RGB", (image.width + padding*2, image.height + padding*2), "white")
        padded.paste(image, (padding, padding))
        buf = io.BytesIO()
        padded.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        return buf.read()


# ── Native PDF Table Extractor (primary for digital PDFs) ──────────────────

class CamelotTableExtractor:
    """
    Extracts tables from digital PDFs using Camelot.
    Camelot reads the PDF vector stream directly — no image conversion needed.

    Falls back to GeminiTableExtractor if Camelot is not installed.
    """

    def __init__(self) -> None:
        try:
            import camelot  # type: ignore  # noqa: F401
            self._available = True
            logger.info("[TableExtractor] Camelot backend available.")
        except ImportError:
            self._available = False
            logger.warning(
                "[TableExtractor] Camelot not installed. "
                "Install with: pip install camelot-py[cv]. "
                "Will use Gemini Vision for table extraction."
            )

    @property
    def is_available(self) -> bool:
        return self._available

    def extract_from_pdf(
        self,
        pdf_path: str | Path,
        pages: Optional[list[int]] = None,
    ) -> list[TableData]:
        """
        Extract all tables from a digital PDF file using Camelot.

        Args:
            pdf_path: Path to the PDF.
            pages:    0-indexed page numbers to extract. None = all pages.

        Returns:
            List of TableData, one per detected table.
        """
        if not self._available:
            return []

        import camelot  # type: ignore
        pdf_path = str(Path(pdf_path))

        # Camelot uses 1-indexed pages
        page_str = ",".join(str(p + 1) for p in pages) if pages else "all"

        try:
            tables = camelot.read_pdf(pdf_path, pages=page_str, flavor="lattice")
            if not tables.n:
                # Try stream mode (no visible borders)
                tables = camelot.read_pdf(pdf_path, pages=page_str, flavor="stream")
        except Exception as exc:
            logger.warning(f"[TableExtractor:Camelot] Failed: {exc}")
            return []

        results: list[TableData] = []
        for i, table in enumerate(tables):
            df = table.df
            if df.empty or len(df) < 2:
                continue

            # First row = headers
            headers = [str(h).strip() for h in df.iloc[0].tolist()]
            rows = []
            for _, row in df.iloc[1:].iterrows():
                row_dict = {headers[j]: str(v).strip() for j, v in enumerate(row)}
                rows.append(row_dict)

            td = TableData(
                block_id=i,
                page=table.page - 1,  # convert to 0-indexed
                headers=headers,
                rows=rows,
                confidence=float(table.accuracy) / 100 if hasattr(table, "accuracy") else 0.9,
                extraction_method="camelot_lattice",
            )
            logger.info(
                f"[TableExtractor:Camelot] Table {i+1}: "
                f"{td.col_count} cols × {td.row_count} rows (page {td.page + 1})"
            )
            results.append(td)

        return results
