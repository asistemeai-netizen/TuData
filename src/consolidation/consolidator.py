# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Consolidator (Stage Final)
======================================
Takes the outputs from all 3 content streams and produces a unified
ProjectDocument using an LLM with structured, type-specific prompting.

Key improvements over the old LLM extractor:
  1. The LLM receives structured data (tables as JSON rows, figure descriptions
     with components), NOT raw Markdown — dramatically reducing hallucinations.
  2. The prompt is customized per project type.
  3. Sections are passed as individual segments (≤4000 chars each) to avoid
     context window truncation on large documents.
  4. Missing fields are validated against expected fields for the project type.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from loguru import logger

from src.consolidation.project_schemas import (
    FigureResult,
    LineItem,
    MissingField,
    MissingFieldSeverity,
    ProjectDocument,
    ProjectSummary,
    TableResult,
    TechnicalSpec,
)
from src.extractors.figure_extractor import FigureData
from src.extractors.table_extractor import TableData
from src.models import Block, ProjectType


# ── Expected fields per project type ─────────────────────────────────────────

_EXPECTED_FIELDS: dict[str, list[tuple[str, str, MissingFieldSeverity]]] = {
    ProjectType.ELECTRICAL_SWITCHGEAR_QUOTE.value: [
        ("quote_number",    "Quote / proposal number",          MissingFieldSeverity.CRITICAL),
        ("quote_date",      "Date of the quote",                MissingFieldSeverity.CRITICAL),
        ("client",          "Client / customer name",           MissingFieldSeverity.CRITICAL),
        ("vendor",          "Vendor / manufacturer name",       MissingFieldSeverity.WARNING),
        ("total_amount",    "Total quoted amount",              MissingFieldSeverity.CRITICAL),
        ("line_items",      "Equipment line items list",        MissingFieldSeverity.CRITICAL),
        ("valid_until",     "Quote validity / expiration date", MissingFieldSeverity.WARNING),
    ],
    ProjectType.MATERIAL_INVOICE.value: [
        ("quote_number",    "Invoice number",                   MissingFieldSeverity.CRITICAL),
        ("quote_date",      "Invoice date",                     MissingFieldSeverity.CRITICAL),
        ("client",          "Bill-to party",                    MissingFieldSeverity.CRITICAL),
        ("total_amount",    "Invoice total",                    MissingFieldSeverity.CRITICAL),
        ("line_items",      "Line items",                       MissingFieldSeverity.CRITICAL),
    ],
}

_CONSOLIDATE_PROMPT = """You are an expert document analyst specializing in {domain}.

Analyze the following structured content extracted from a {doc_type} document and produce a consolidated project summary.

=== TEXT SECTIONS ===
{text_content}

=== TABLES ===
{table_content}

=== FIGURES & DIAGRAMS ===
{figure_content}

Return a SINGLE valid JSON object:
{{
  "executive_summary": "<3-5 sentence summary of the entire document>",
  "project_summary": {{
    "project_name":     "<project name or null>",
    "project_number":   "<project/job number or null>",
    "client":           "<client/customer name or null>",
    "vendor":           "<vendor/supplier name or null>",
    "quote_number":     "<quote/proposal/invoice number or null>",
    "quote_date":       "<date as string or null>",
    "valid_until":      "<expiration date or null>",
    "project_location": "<location or null>",
    "total_amount":     "<total dollar amount as string, e.g. '$45,230.00' or null>",
    "currency":         "USD"
  }},
  "line_items": [
    {{
      "item_no": "<number>", "description": "<full description>",
      "quantity": "<qty>", "unit": "<ea/lot/set>",
      "unit_price": "<$>", "total": "<$>", "notes": "<any>"
    }}
  ],
  "tech_specs": [
    {{"category": "<electrical/mechanical/etc>", "parameter": "<name>", "value": "<val>", "unit": "<unit or null>"}}
  ],
  "text_sections": [
    {{"section": "<heading>", "content": "<full text>", "page": <page_int>}}
  ],
  "missing_fields": ["<field names that could not be found>"],
  "quality_score": <0-100 integer>
}}

Rules:
- line_items: extract ALL equipment/material items with pricing. If it's a quote, this is THE most important output.
- tech_specs: electrical ratings, voltages, amperages, weights, dimensions, certifications.
- quality_score: 100 = everything found with high confidence. Deduct 10 per critical missing field.
- Return ONLY JSON, no markdown fencing.
"""

_DOMAIN_MAP: dict[str, str] = {
    ProjectType.ELECTRICAL_SWITCHGEAR_QUOTE.value: "electrical switchgear and automatic transfer switches",
    ProjectType.ELECTRICAL_PANEL_QUOTE.value: "electrical panel and distribution equipment",
    ProjectType.MECHANICAL_SPEC.value: "mechanical equipment specifications",
    ProjectType.MATERIAL_INVOICE.value: "material procurement and invoicing",
    ProjectType.TECHNICAL_DATASHEET.value: "technical product specifications",
}


class Consolidator:
    """
    Consolidates outputs from all 3 content streams into a unified ProjectDocument.

    The Consolidator is the final "brain" of the pipeline — it receives already-
    structured data (tables as JSON, figures as components lists) and asks the LLM
    to synthesize a complete project picture rather than parse raw OCR text.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        import os
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        key = api_key or os.getenv("GEMINI_API_KEY")
        self._client = genai.Client(api_key=key)
        self._model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._types = types
        logger.info(f"[Consolidator] Gemini backend: {self._model}")

    def consolidate(
        self,
        doc_id: str,
        source_path: str,
        page_count: int,
        project_type: ProjectType,
        classification_confidence: float,
        doc_format: str,
        text_blocks: list[Block],
        table_data: list[TableData],
        figure_data: list[FigureData],
        metrics: Optional[dict] = None,
        log_callback=None,
    ) -> ProjectDocument:
        """
        Consolidate all stream outputs into a ProjectDocument.

        Args:
            doc_id, source_path, page_count: Document identity.
            project_type:   Detected project type (drives prompt + validation).
            text_blocks:    Text blocks with .text populated.
            table_data:     Structured tables from TableExtractor.
            figure_data:    Structured figure analyses from FigureAnalyzer.

        Returns:
            Validated ProjectDocument.
        """
        def _log(msg: str):
            logger.info(msg)
            if log_callback:
                log_callback(msg)

        _log(f"[Consolidator] Assembling project document for '{doc_id}' ({project_type.value})")

        # Build structured content for each stream
        text_content  = self._format_text(text_blocks)
        table_content = self._format_tables(table_data)
        figure_content= self._format_figures(figure_data)

        domain  = _DOMAIN_MAP.get(project_type.value, "industrial documents")
        doc_type= project_type.value.replace("_", " ")

        prompt = _CONSOLIDATE_PROMPT.format(
            domain=domain,
            doc_type=doc_type,
            text_content=text_content[:8000],
            table_content=table_content[:6000],
            figure_content=figure_content[:3000],
        )

        _log(f"[Consolidator] Calling LLM ({self._model}) ...")
        t0 = time.time()

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=16384,
                    temperature=0.0,
                ),
            )
            llm_ms = int((time.time() - t0) * 1000)
            _log(f"[Consolidator] LLM responded in {llm_ms}ms")

            data = json.loads(response.text or "{}")
        except Exception as exc:
            logger.error(f"[Consolidator] LLM call failed: {exc}")
            data = {}

        # Build ProjectDocument from LLM output
        proj_summary_raw = data.get("project_summary", {})
        proj_summary = ProjectSummary(**{
            k: proj_summary_raw.get(k)
            for k in ProjectSummary.model_fields.keys()
            if k in proj_summary_raw
        }) if proj_summary_raw else None

        # Convert line items
        line_items = [
            LineItem(
                item_no=li.get("item_no"),
                description=li.get("description", ""),
                quantity=li.get("quantity"),
                unit=li.get("unit"),
                unit_price=li.get("unit_price"),
                total=li.get("total"),
                notes=li.get("notes"),
            )
            for li in data.get("line_items", [])
            if li.get("description")
        ]

        # Convert tech specs
        tech_specs = [
            TechnicalSpec(
                category=ts.get("category", "general"),
                parameter=ts.get("parameter", ""),
                value=ts.get("value", ""),
                unit=ts.get("unit"),
            )
            for ts in data.get("tech_specs", [])
            if ts.get("parameter") and ts.get("value")
        ]

        # Convert tables to schema
        table_results = [
            TableResult(
                block_id=td.block_id,
                page=td.page,
                caption=td.caption,
                headers=td.headers,
                rows=td.rows,
                row_count=td.row_count,
                col_count=td.col_count,
                extraction_method=td.extraction_method,
                confidence=td.confidence,
            )
            for td in table_data
        ]

        # Convert figures to schema
        figure_results = [
            FigureResult(
                block_id=fd.block_id,
                page=fd.page,
                figure_type=fd.figure_type,
                description=fd.description,
                components=fd.components,
                technical_specs=fd.technical_specs,
                text_labels=fd.text_labels,
                references=fd.references,
                confidence=fd.confidence,
            )
            for fd in figure_data
        ]

        # Validate missing fields
        missing = self._check_missing_fields(
            project_type, proj_summary, line_items,
            data.get("missing_fields", [])
        )

        quality_score = float(data.get("quality_score", 0))

        if metrics is not None:
            metrics["consolidator_llm_ms"] = llm_ms if "llm_ms" in dir() else 0
            metrics["line_items_extracted"] = len(line_items)
            metrics["tech_specs_extracted"] = len(tech_specs)
            metrics["missing_fields_count"] = len(missing)
            metrics["consolidator_quality_score"] = quality_score

        _log(
            f"[Consolidator] Done: {len(line_items)} line items, "
            f"{len(tech_specs)} specs, {len(missing)} missing fields, "
            f"score={quality_score}"
        )

        return ProjectDocument(
            doc_id=doc_id,
            source_path=source_path,
            page_count=page_count,
            project_type=project_type.value,
            classification_confidence=classification_confidence,
            doc_format=doc_format,
            executive_summary=data.get("executive_summary"),
            project_summary=proj_summary,
            tables=table_results,
            figures=figure_results,
            line_items=line_items,
            tech_specs=tech_specs,
            text_sections=data.get("text_sections", []),
            missing_fields=missing,
            quality_score=quality_score,
        )

    # ── Stream formatters ─────────────────────────────────────────────────────

    @staticmethod
    def _format_text(blocks: list[Block]) -> str:
        """Format text blocks for the LLM prompt."""
        sections = []
        for b in blocks:
            if b.text and b.text.strip():
                sections.append(f"[Page {b.page+1}] {b.label.value}: {b.text.strip()}")
        return "\n\n".join(sections)

    @staticmethod
    def _format_tables(tables: list[TableData]) -> str:
        """Format tables as JSON strings for the LLM prompt."""
        parts = []
        for i, t in enumerate(tables):
            parts.append(
                f"TABLE {i+1} (Page {t.page+1}, {t.col_count} cols × {t.row_count} rows):\n"
                + json.dumps({"headers": t.headers, "sample_rows": t.rows[:10]}, indent=2)
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_figures(figures: list[FigureData]) -> str:
        """Format figure analyses for the LLM prompt."""
        parts = []
        for i, f in enumerate(figures):
            parts.append(
                f"FIGURE {i+1} (Page {f.page+1}, type={f.figure_type}):\n"
                + f.to_text()
            )
        return "\n\n".join(parts)

    def _check_missing_fields(
        self,
        project_type: ProjectType,
        summary: Optional[ProjectSummary],
        line_items: list[LineItem],
        llm_missing: list[str],
    ) -> list[MissingField]:
        """Validate that critical fields were found for this project type."""
        expected = _EXPECTED_FIELDS.get(project_type.value, [])
        missing: list[MissingField] = []

        for field_name, description, severity in expected:
            # Check if LLM flagged it as missing
            if field_name in llm_missing:
                missing.append(MissingField(
                    field_name=field_name,
                    description=description,
                    severity=severity,
                ))
                continue

            # Extra validation: check summary fields
            if summary and field_name in ProjectSummary.model_fields:
                val = getattr(summary, field_name, None)
                if not val:
                    missing.append(MissingField(
                        field_name=field_name,
                        description=description,
                        severity=severity,
                    ))
                    continue

            # Check line_items directly
            if field_name == "line_items" and not line_items:
                missing.append(MissingField(
                    field_name="line_items",
                    description=description,
                    severity=severity,
                ))

        return missing
