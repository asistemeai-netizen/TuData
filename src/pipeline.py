# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Content-First Parallel Pipeline (v2)
==============================================
Implements the new architecture:

  Stage 0: Intake & Project Classification
    └─ DocumentAnalyzer  → detect digital vs scanned, page stats
    └─ ProjectClassifier → classify document type (Gemini Flash)

  Stage 1: Layout Detection
    └─ LayoutDetector (YOLOv8) → list[Block]

  Stage 2: Content Splitter
    └─ ContentSplitter → ContentSplit (3 typed streams)

  Stage 3: Parallel Specialized Extraction
    ├─ 3A: TextExtractor  → Block.text (native PyMuPDF or OCR)
    ├─ 3B: TableExtractor → list[TableData] (Camelot or Gemini Vision JSON)
    └─ 3C: FigureAnalyzer → list[FigureData] (Gemini Vision structured)

  Stage 4: Consolidation
    └─ Consolidator (Gemini) → ProjectDocument

  Stage 5 (optional): Legacy JSON / Markdown
    └─ MarkdownBuilder + LLMExtractor (backward compat)

Backward compatible: process_file() still returns ExtractedDocument.
process_project() returns the richer ProjectDocument.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Callable, Optional

import fitz  # PyMuPDF
from loguru import logger
from PIL import Image

# ── Legacy imports (kept for backward compat) ──────────────────────────────
from src.assembly.markdown_builder import MarkdownBuilder
from src.assembly.reading_order import ReadingOrderResolver
from src.detection.layout_detector import LayoutDetector
from src.detection.visualizer import BlockVisualizer
from src.extraction.llm_extractor import LLMExtractor
from src.extraction.schemas import ExtractedDocument, ChunkType

# ── New Content-First Pipeline imports ─────────────────────────────────────
from src.intake.document_analyzer import DocumentAnalyzer
from src.intake.project_classifier import ProjectClassifier
from src.splitter.content_splitter import ContentSplitter, ContentSplit
from src.extractors.text_extractor import NativeTextExtractor, OcrTextExtractor
from src.extractors.table_extractor import CamelotTableExtractor, GeminiTableExtractor, TableData
from src.extractors.figure_extractor import FigureAnalyzer, FigureData
from src.consolidation.consolidator import Consolidator
from src.consolidation.project_schemas import ProjectDocument

from src.models import Block, BlockLabel, DocumentFormat, ProjectType
from src.ocr.gemini_ocr import GeminiVisionOCR
from src.ocr.ocr_engine import PaddleOCREngine


# Block types that benefit from Gemini OCR over PaddleOCR (legacy)
_GEMINI_PREFERRED = {BlockLabel.TABLE, BlockLabel.FORMULA, BlockLabel.FIGURE}


class DocumentPipeline:
    """
    End-to-end document processing pipeline (v2 — Content-First Architecture).

    Args:
        output_dir:    Directory for .md and .json results.
        artifacts_dir: Directory for annotated PNG visualizations.
        visualize:     If True, save bounding-box artifact images.
        ocr_lang:      PaddleOCR language code (default 'en').
        workers:       Max concurrent files in batch mode (default 4).
        use_v2:        If True, use the new Content-First pipeline.
                       If False, use the legacy linear pipeline. Default: True.
    """

    def __init__(
        self,
        output_dir: str = "results",
        artifacts_dir: str = "artifacts",
        visualize: bool = False,
        ocr_lang: str = "en",
        workers: int = 4,
        use_v2: bool = True,
    ) -> None:
        self._output_dir    = Path(output_dir)
        self._artifacts_dir = Path(artifacts_dir)
        self._visualize     = visualize
        self._workers       = workers
        self._use_v2        = use_v2

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing pipeline components ...")

        # Shared / legacy
        self._detector   = LayoutDetector()
        self._resolver   = ReadingOrderResolver()
        self._builder    = MarkdownBuilder()
        self._ocr        = PaddleOCREngine(lang=ocr_lang)
        self._gemini_ocr: Optional[GeminiVisionOCR] = self._try_init_gemini_ocr()
        self._extractor  = LLMExtractor()
        self._visualizer = BlockVisualizer(output_dir=artifacts_dir) if visualize else None

        # v2 — new components
        if use_v2:
            self._analyzer    = DocumentAnalyzer()
            self._classifier  = ProjectClassifier()
            self._splitter    = ContentSplitter()
            self._native_text = NativeTextExtractor()
            self._ocr_text    = OcrTextExtractor(lang=ocr_lang)
            self._camelot     = CamelotTableExtractor()
            self._gemini_table= GeminiTableExtractor() if os.getenv("GEMINI_API_KEY") else None
            self._figure_ai   = FigureAnalyzer() if os.getenv("GEMINI_API_KEY") else None
            self._consolidator= Consolidator()

        logger.success("Pipeline ready (v2={}).".format(use_v2))

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def process_file(
        self,
        pdf_path: str | Path,
        progress_callback: Callable = None,
        log_callback: Callable = None,
        metrics: dict = None,
    ) -> ExtractedDocument:
        """
        Backward-compatible entry point. Returns ExtractedDocument.
        Internally uses v2 pipeline if use_v2=True.
        """
        if self._use_v2:
            proj_doc = await self.process_project(
                pdf_path, progress_callback, log_callback, metrics
            )
            # Convert ProjectDocument → ExtractedDocument for backward compat
            return self._project_to_legacy(proj_doc)
        return await self._process_legacy(pdf_path, progress_callback, log_callback, metrics)

    async def process_project(
        self,
        pdf_path: str | Path,
        progress_callback: Callable = None,
        log_callback: Callable = None,
        metrics: dict = None,
    ) -> ProjectDocument:
        """
        New v2 entry point. Returns a rich ProjectDocument.
        """
        pdf_path = Path(pdf_path)
        doc_id   = pdf_path.stem
        metrics  = metrics if metrics is not None else {}

        def _log(msg: str):
            logger.info(msg)
            if log_callback: log_callback(msg)

        # ── Stage 0: Intake & Classification ──────────────────────────────
        if progress_callback: progress_callback(0)
        _log(f"▶ [Stage 0] Intake: {pdf_path.name}")

        analysis = self._analyzer.analyze(pdf_path)
        page_count   = analysis["page_count"]
        doc_format   = analysis["doc_format"]
        is_digital   = doc_format == DocumentFormat.DIGITAL
        text_sample  = analysis["text_sample"]
        has_images   = analysis["has_images"]

        metrics["pages_processed"]   = page_count
        metrics["doc_format"]        = doc_format.value
        metrics["digital_page_ratio"]= analysis["digital_page_ratio"]
        metrics["avg_chars_per_page"]= analysis["avg_chars_per_page"]

        _log(f"  ✓ Format: {doc_format.value} | Pages: {page_count} | Digital: {analysis['digital_page_ratio']:.0%}")

        project_type, cls_confidence = self._classifier.classify(
            text_sample, page_count, has_images, is_digital
        )
        metrics["project_type"]               = project_type.value
        metrics["classification_confidence"]  = cls_confidence
        _log(f"  ✓ Project Type: {project_type.value} (confidence: {cls_confidence:.0%})")

        # ── Stage 1: Layout Detection ──────────────────────────────────────
        if progress_callback: progress_callback(1)
        _log("▶ [Stage 1] Layout Detection ...")

        t0 = time.time()
        blocks = self._detector.detect_pdf(pdf_path)

        metrics["blocks_total"]             = len(blocks)
        if blocks:
            metrics["detection_confidence_mean"] = round(sum(b.confidence for b in blocks) / len(blocks), 4)
            metrics["low_confidence_ratio"]      = round(sum(1 for b in blocks if b.confidence < 0.5) / len(blocks), 4)
        metrics["layout_duration_s"] = round(time.time() - t0, 2)

        _log(f"  ✓ {len(blocks)} blocks detected in {metrics['layout_duration_s']}s")

        # Optional visualization
        if self._visualize and self._visualizer:
            pages = self._render_all_pages(pdf_path)
            self._visualizer.save_all_pages(pages, blocks, doc_id)

        # ── Stage 2: Content Splitter ──────────────────────────────────────
        if progress_callback: progress_callback(2)
        _log("▶ [Stage 2] Content Splitter ...")

        split = self._splitter.split(blocks)
        metrics["split_profile"] = split.content_profile
        metrics["table_ratio"]   = round(split.table_ratio, 4)
        metrics["figure_ratio"]  = round(split.figure_ratio, 4)

        _log(
            f"  ✓ Streams: text={len(split.text_blocks)}, "
            f"tables={len(split.table_blocks)}, "
            f"figures={len(split.figure_blocks)}"
        )

        # ── Stage 3: Parallel Specialized Extraction ──────────────────────
        if progress_callback: progress_callback(3)
        _log("▶ [Stage 3] Parallel Extraction (text + tables + figures) ...")

        page_images = self._render_all_pages(pdf_path)
        t0 = time.time()

        # Run all 3 streams concurrently
        text_task   = self._extract_text_stream(split, pdf_path, page_images, is_digital, _log)
        table_task  = self._extract_table_stream(split, pdf_path, page_images, project_type, _log)
        figure_task = self._extract_figure_stream(split, page_images, project_type, _log)

        text_blocks, table_data, figure_data = await asyncio.gather(
            text_task, table_task, figure_task
        )

        metrics["extraction_duration_s"] = round(time.time() - t0, 2)
        metrics["tables_extracted"]      = len(table_data)
        metrics["figures_analyzed"]      = len(figure_data)
        _log(
            f"  ✓ Extraction done in {metrics['extraction_duration_s']}s: "
            f"{len(table_data)} tables, {len(figure_data)} figures"
        )

        # ── Stage 4: Consolidation ─────────────────────────────────────────
        if progress_callback: progress_callback(4)
        _log("▶ [Stage 4] Consolidation (LLM) ...")

        project_doc = self._consolidator.consolidate(
            doc_id=doc_id,
            source_path=str(pdf_path),
            page_count=page_count,
            project_type=project_type,
            classification_confidence=cls_confidence,
            doc_format=doc_format.value,
            text_blocks=text_blocks,
            table_data=table_data,
            figure_data=figure_data,
            metrics=metrics,
            log_callback=_log,
        )

        # Build pipeline quality score
        quality_score = project_doc.quality_score
        missing_critical = sum(1 for m in project_doc.missing_fields if m.severity.value == "critical")
        metrics["pipeline_quality_score"] = quality_score
        metrics["missing_critical_fields"]= missing_critical
        metrics["line_items_extracted"]   = len(project_doc.line_items)

        _log(
            f"[KPIs] Quality: {quality_score}/100 | "
            f"Line items: {len(project_doc.line_items)} | "
            f"Missing critical: {missing_critical}"
        )
        _log(f"✅ Pipeline v2 complete: {pdf_path.name}")

        # Save JSON
        import json
        json_path = self._output_dir / f"{doc_id}.json"
        json_path.write_text(project_doc.model_dump_json(indent=2), encoding="utf-8")

        # Also build legacy Markdown (for display)
        all_text_blocks = sorted(text_blocks, key=lambda b: (b.page, b.bbox.y1, b.bbox.x1))
        sorted_all = self._resolver.sort(all_text_blocks)
        markdown = self._builder.build(sorted_all, doc_title=doc_id.replace("_", " ").title())
        md_path = self._output_dir / f"{doc_id}.md"
        self._builder.save(markdown, md_path)

        return project_doc

    # ──────────────────────────────────────────────────────────────────────────
    # Stream helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _extract_text_stream(
        self,
        split: ContentSplit,
        pdf_path: Path,
        page_images: list[Image.Image],
        is_digital: bool,
        _log,
    ) -> list[Block]:
        """Extract text from text/title/list blocks using optimal method."""
        text_blocks = split.text_blocks

        if is_digital and text_blocks:
            _log(f"  [Text] Using native PyMuPDF extraction for {len(text_blocks)} text blocks")
            doc = fitz.open(str(pdf_path))
            # Group by page and process with NativeTextExtractor
            by_page: dict[int, list[Block]] = {}
            for b in text_blocks:
                by_page.setdefault(b.page, []).append(b)
            for page_num, pg_blocks in by_page.items():
                page = doc[page_num]
                self._native_text.extract_from_page(page, pg_blocks)
            doc.close()
        elif text_blocks:
            _log(f"  [Text] Using PaddleOCR for {len(text_blocks)} text blocks (scanned)")
            text_blocks = await self._ocr_text.extract_blocks(text_blocks, page_images)

        empty = sum(1 for b in text_blocks if not b.text or not b.text.strip())
        _log(f"  [Text] ✓ {len(text_blocks)} blocks, {empty} empty")
        return text_blocks

    async def _extract_table_stream(
        self,
        split: ContentSplit,
        pdf_path: Path,
        page_images: list[Image.Image],
        project_type: ProjectType,
        _log,
    ) -> list[TableData]:
        """Extract tables using Camelot (digital) or Gemini Vision JSON (scanned)."""
        table_blocks = split.table_blocks
        if not table_blocks:
            return []

        _log(f"  [Tables] Processing {len(table_blocks)} table blocks ...")

        # Try Camelot first for digital PDFs
        if self._camelot.is_available:
            pages = list(set(b.page for b in table_blocks))
            result = self._camelot.extract_from_pdf(pdf_path, pages=pages)
            if result:
                _log(f"  [Tables] ✓ Camelot extracted {len(result)} tables")
                return result

        # Fall back to Gemini Vision JSON for each block
        if self._gemini_table:
            _log(f"  [Tables] Using Gemini Vision JSON for {len(table_blocks)} table blocks")
            loop = asyncio.get_running_loop()
            table_data = []
            for block in table_blocks:
                crop = self._crop_block(page_images[block.page], block, padding=15)
                td = await loop.run_in_executor(
                    None, self._gemini_table.extract, crop, block, project_type.value
                )
                table_data.append(td)
            return table_data

        # Last resort: use legacy Gemini OCR for text
        loop = asyncio.get_running_loop()
        table_data = []
        for block in table_blocks:
            crop = self._crop_block(page_images[block.page], block, padding=15)
            if self._gemini_ocr:
                text = await loop.run_in_executor(None, self._gemini_ocr.extract_text, crop, "table")
            else:
                text = await loop.run_in_executor(None, self._ocr.extract_text, crop)
            block.text = text
            table_data.append(TableData(
                block_id=block.id, page=block.page,
                raw_markdown=text, extraction_method="legacy_ocr", confidence=0.5
            ))
        return table_data

    async def _extract_figure_stream(
        self,
        split: ContentSplit,
        page_images: list[Image.Image],
        project_type: ProjectType,
        _log,
    ) -> list[FigureData]:
        """Analyze figure blocks with Gemini Vision (context-aware)."""
        figure_blocks = split.figure_blocks
        if not figure_blocks:
            return []

        _log(f"  [Figures] Analyzing {len(figure_blocks)} figures ...")

        if not self._figure_ai:
            _log("  [Figures] FigureAnalyzer not available (no API key)")
            return []

        loop = asyncio.get_running_loop()
        tasks = []
        for block in figure_blocks:
            crop = self._crop_block(page_images[block.page], block, padding=15)
            tasks.append(
                loop.run_in_executor(None, self._figure_ai.analyze, crop, block, project_type)
            )
        figure_data = list(await asyncio.gather(*tasks))
        _log(f"  [Figures] ✓ {len(figure_data)} figures analyzed")
        return figure_data

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy pipeline (v1 — preserved for fallback)
    # ──────────────────────────────────────────────────────────────────────────

    async def _process_legacy(
        self,
        pdf_path: str | Path,
        progress_callback: Callable = None,
        log_callback: Callable = None,
        metrics: dict = None,
    ) -> ExtractedDocument:
        """Original linear pipeline — kept as fallback."""
        pdf_path = Path(pdf_path)
        doc_id = pdf_path.stem
        metrics = metrics if metrics is not None else {}

        def _log(msg: str):
            logger.debug(msg)
            if log_callback: log_callback(msg)

        _log(f"▶ [Legacy] Processing: {pdf_path.name}")

        if progress_callback: progress_callback(0)
        blocks = self._detector.detect_pdf(pdf_path)

        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        doc.close()

        if blocks:
            metrics["detection_confidence_mean"] = round(sum(b.confidence for b in blocks) / len(blocks), 4)
            metrics["low_confidence_ratio"]      = round(sum(1 for b in blocks if b.confidence < 0.5) / len(blocks), 4)
        metrics["pages_processed"] = page_count
        metrics["blocks_per_page_avg"] = round(len(blocks) / max(1, page_count), 2)
        _log(f"✓ Layout Detection: {len(blocks)} blocks")

        if self._visualize and self._visualizer:
            pages = self._render_all_pages(pdf_path)
            self._visualizer.save_all_pages(pages, blocks, doc_id)

        if progress_callback: progress_callback(1)
        t0 = time.time()
        blocks = await self._ocr_all_blocks(blocks, pdf_path, _log)
        metrics["ocr_duration_per_block"] = round((time.time() - t0) / max(1, len(blocks)), 3)
        metrics["ocr_blocks_total"]  = len(blocks)
        metrics["ocr_empty_ratio"]   = round(sum(1 for b in blocks if not b.text or not b.text.strip()) / max(1, len(blocks)), 4)
        _log("✓ OCR complete")

        if progress_callback: progress_callback(2)
        blocks = self._resolver.sort(blocks)
        _log("✓ Reading Order applied")

        if progress_callback: progress_callback(3)
        markdown = self._builder.build(blocks, doc_title=doc_id.replace("_", " ").title())
        md_path = self._output_dir / f"{doc_id}.md"
        self._builder.save(markdown, md_path)
        metrics["markdown_chars"] = len(markdown)
        metrics["content_density"] = round(len(markdown) / max(1, page_count), 1)
        _log(f"✓ Markdown: {len(markdown)} chars")

        if progress_callback: progress_callback(4)
        t0_llm = time.time()
        extracted = self._extractor.extract(markdown=markdown, doc_id=doc_id, source_path=str(pdf_path), page_count=page_count)
        metrics["llm_latency_ms"]    = int((time.time() - t0_llm) * 1000)
        metrics["chunks_extracted"]  = len(extracted.chunks)
        metrics["avg_chunk_confidence"] = round(sum(c.confidence for c in extracted.chunks) / max(1, len(extracted.chunks)), 4)
        metrics["entities_extracted"]= sum(len(c.entities) for c in extracted.chunks)
        metrics["doc_type_detected"] = extracted.chunks[0].doc_type_hint if extracted.chunks and hasattr(extracted.chunks[0], "doc_type_hint") else "unknown"
        pipeline_quality_score = ((1 - metrics["ocr_empty_ratio"]) * 40) + (metrics["avg_chunk_confidence"] * 40) + ((1 - metrics["low_confidence_ratio"]) * 20)
        metrics["pipeline_quality_score"] = round(pipeline_quality_score, 1)

        json_path = self._output_dir / f"{doc_id}.json"
        self._extractor.save_json(extracted, json_path)
        _log(f"✅ Legacy pipeline done: {len(extracted.chunks)} chunks")

        return extracted

    # ──────────────────────────────────────────────────────────────────────────
    # Shared helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _project_to_legacy(self, project_doc: ProjectDocument) -> ExtractedDocument:
        """Convert a ProjectDocument into a legacy ExtractedDocument for backward compat."""
        from src.extraction.schemas import DocumentChunk, DocumentMetadata, Entity, KeyValue

        meta = DocumentMetadata(
            doc_id=project_doc.doc_id,
            source_path=project_doc.source_path,
            page_count=project_doc.page_count,
            title=project_doc.project_summary.project_name if project_doc.project_summary else None,
        )

        chunks = []

        # Summary chunk
        if project_doc.executive_summary:
            chunks.append(DocumentChunk(
                chunk_id=f"{project_doc.doc_id}_summary",
                doc_id=project_doc.doc_id,
                page=0,
                chunk_type=ChunkType.TEXT,
                text=project_doc.executive_summary,
                summary="Executive summary of the document",
                doc_type_hint=project_doc.project_type,
                confidence=project_doc.quality_score / 100,
            ))

        # Line items as a chunk
        if project_doc.line_items:
            items_text = "\n".join(
                f"{li.item_no or ''} | {li.description} | {li.quantity or ''} {li.unit or ''} | {li.unit_price or ''} | {li.total or ''}"
                for li in project_doc.line_items
            )
            chunks.append(DocumentChunk(
                chunk_id=f"{project_doc.doc_id}_line_items",
                doc_id=project_doc.doc_id,
                page=0,
                chunk_type=ChunkType.TABLE,
                text=items_text,
                summary=f"{len(project_doc.line_items)} line items extracted",
                doc_type_hint=project_doc.project_type,
                key_values=[KeyValue(key=li.description[:50], value=li.total or "") for li in project_doc.line_items[:20]],
                confidence=0.9,
            ))

        # Tech specs chunk
        if project_doc.tech_specs:
            specs_text = "\n".join(f"{s.parameter}: {s.value} {s.unit or ''}" for s in project_doc.tech_specs)
            chunks.append(DocumentChunk(
                chunk_id=f"{project_doc.doc_id}_specs",
                doc_id=project_doc.doc_id,
                page=0,
                chunk_type=ChunkType.TEXT,
                text=specs_text,
                summary="Technical specifications",
                doc_type_hint=project_doc.project_type,
                confidence=0.9,
            ))

        # Figures as chunks
        for fig in project_doc.figures:
            txt = f"[{fig.figure_type}] {fig.description}"
            if fig.components:
                txt += f"\nComponents: {', '.join(fig.components)}"
            chunks.append(DocumentChunk(
                chunk_id=f"{project_doc.doc_id}_fig_{fig.block_id}",
                doc_id=project_doc.doc_id,
                page=fig.page,
                chunk_type=ChunkType.FIGURE,
                text=txt,
                summary=fig.description[:150],
                doc_type_hint=project_doc.project_type,
                keywords=fig.components[:10],
                confidence=fig.confidence,
            ))

        # Text sections as chunks
        for i, sec in enumerate(project_doc.text_sections[:20]):
            txt = sec.get("content", "")
            if txt:
                chunks.append(DocumentChunk(
                    chunk_id=f"{project_doc.doc_id}_text_{i}",
                    doc_id=project_doc.doc_id,
                    page=sec.get("page", 0),
                    chunk_type=ChunkType.TEXT,
                    text=txt,
                    section=sec.get("section"),
                    doc_type_hint=project_doc.project_type,
                    confidence=0.85,
                ))

        if not chunks:
            chunks.append(DocumentChunk(
                chunk_id=f"{project_doc.doc_id}_p0_0",
                doc_id=project_doc.doc_id,
                page=0,
                chunk_type=ChunkType.TEXT,
                text=project_doc.executive_summary or "No content extracted.",
                confidence=0.1,
            ))

        return ExtractedDocument(metadata=meta, chunks=chunks)

    async def _ocr_all_blocks(
        self, blocks: list[Block], pdf_path: Path, log_callback: Callable = None
    ) -> list[Block]:
        loop = asyncio.get_running_loop()
        page_images = self._render_all_pages(pdf_path)
        completed = 0

        async def ocr_block(block: Block) -> Block:
            nonlocal completed
            page_img = page_images[block.page]
            crop = self._crop_block(page_img, block)
            if block.label in _GEMINI_PREFERRED and self._gemini_ocr:
                block.text = await loop.run_in_executor(None, self._gemini_ocr.extract_text, crop, block.label.value.lower())
                engine = "Gemini Vision"
            else:
                block.text = await loop.run_in_executor(None, self._ocr.extract_text, crop)
                engine = "PaddleOCR"
            completed += 1
            if log_callback:
                t_str = (block.text or "").replace("\n", " ")[:60] or "(VACÍO)"
                log_callback(f"[OCR] Block #{block.id} — {block.label.value} → {engine} → \"{t_str}\"")
            return block

        return list(await asyncio.gather(*[ocr_block(b) for b in blocks]))

    def _render_all_pages(self, pdf_path: Path) -> list[Image.Image]:
        doc = fitz.open(str(pdf_path))
        dpi = 200   # upgraded from 150 → better quality
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pages: list[Image.Image] = []
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
        doc.close()
        return pages

    @staticmethod
    def _crop_block(page_image: Image.Image, block: Block, padding: int = 10) -> Image.Image:
        x1 = max(0, int(block.bbox.x1) - padding)
        y1 = max(0, int(block.bbox.y1) - padding)
        x2 = min(page_image.width,  int(block.bbox.x2) + padding)
        y2 = min(page_image.height, int(block.bbox.y2) + padding)
        return page_image.crop((x1, y1, x2, y2))

    @staticmethod
    def _try_init_gemini_ocr() -> Optional[GeminiVisionOCR]:
        if os.getenv("GEMINI_API_KEY"):
            try:
                return GeminiVisionOCR()
            except Exception as exc:
                logger.warning(f"Gemini OCR not available: {exc}")
        return None

    async def process_batch(self, pdf_paths: list[str | Path]) -> list[ExtractedDocument]:
        semaphore = asyncio.Semaphore(self._workers)
        async def bounded(path):
            async with semaphore:
                try:
                    return await self.process_file(path)
                except Exception as exc:
                    logger.error(f"Failed: {path} — {exc}")
                    return None
        results = await asyncio.gather(*[bounded(p) for p in pdf_paths])
        return [r for r in results if r is not None]
