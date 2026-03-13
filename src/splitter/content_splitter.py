# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Content Splitter (Stage 1)
======================================
Separates a flat list of detected Blocks into typed content streams.

Each stream is then processed by a specialized extractor:
  - text_blocks   → TextExtractor (PyMuPDF native or Surya OCR)
  - table_blocks  → TableExtractor (Camelot or Gemini Vision JSON)
  - figure_blocks → FigureAnalyzer (Gemini Vision multimodal)
  - formula_blocks→ Gemini Vision (LaTeX output)
  - meta_blocks   → discarded or stored as frontmatter
"""
from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from src.models import Block, BlockLabel


# Block types belonging to each stream
_TEXT_LABELS    = {BlockLabel.TEXT, BlockLabel.TITLE, BlockLabel.LIST, BlockLabel.CAPTION}
_TABLE_LABELS   = {BlockLabel.TABLE}
_FIGURE_LABELS  = {BlockLabel.FIGURE}
_FORMULA_LABELS = {BlockLabel.FORMULA}
_META_LABELS    = {BlockLabel.HEADER, BlockLabel.FOOTER}


@dataclass
class ContentSplit:
    """
    Typed container for the separated content streams.

    All lists preserve the original block order (page → y1 → x1).
    The `all_blocks` list is the original unsplit list for reference.
    """
    text_blocks:    list[Block] = field(default_factory=list)
    table_blocks:   list[Block] = field(default_factory=list)
    figure_blocks:  list[Block] = field(default_factory=list)
    formula_blocks: list[Block] = field(default_factory=list)
    meta_blocks:    list[Block] = field(default_factory=list)
    unknown_blocks: list[Block] = field(default_factory=list)

    @property
    def total_blocks(self) -> int:
        return (
            len(self.text_blocks) + len(self.table_blocks) +
            len(self.figure_blocks) + len(self.formula_blocks) +
            len(self.meta_blocks) + len(self.unknown_blocks)
        )

    @property
    def content_profile(self) -> dict[str, int]:
        """Distribution of block types — used for project classification signals."""
        return {
            "text":    len(self.text_blocks),
            "tables":  len(self.table_blocks),
            "figures": len(self.figure_blocks),
            "formulas":len(self.formula_blocks),
            "meta":    len(self.meta_blocks),
            "unknown": len(self.unknown_blocks),
        }

    @property
    def table_ratio(self) -> float:
        if not self.total_blocks:
            return 0.0
        return len(self.table_blocks) / self.total_blocks

    @property
    def figure_ratio(self) -> float:
        if not self.total_blocks:
            return 0.0
        return len(self.figure_blocks) / self.total_blocks


class ContentSplitter:
    """
    Separates a flat list of Blocks into typed content streams.

    This stage performs NO processing — it only routes blocks to their
    appropriate stream based on the BlockLabel assigned by the layout detector.
    """

    def split(self, blocks: list[Block]) -> ContentSplit:
        """
        Split a flat list of Blocks into typed streams.

        Args:
            blocks: Flat list of Blocks from LayoutDetector, sorted by page/y/x.

        Returns:
            ContentSplit with each block assigned to exactly one stream.
        """
        result = ContentSplit()

        for block in blocks:
            if block.label in _TEXT_LABELS:
                result.text_blocks.append(block)
            elif block.label in _TABLE_LABELS:
                result.table_blocks.append(block)
            elif block.label in _FIGURE_LABELS:
                result.figure_blocks.append(block)
            elif block.label in _FORMULA_LABELS:
                result.formula_blocks.append(block)
            elif block.label in _META_LABELS:
                result.meta_blocks.append(block)
            else:
                result.unknown_blocks.append(block)

        profile = result.content_profile
        logger.info(
            f"[Splitter] {result.total_blocks} blocks split → "
            f"text={profile['text']}, tables={profile['tables']}, "
            f"figures={profile['figures']}, formulas={profile['formulas']}, "
            f"meta={profile['meta']}, unknown={profile['unknown']}"
        )
        logger.info(
            f"[Splitter] Table ratio={result.table_ratio:.0%}, "
            f"Figure ratio={result.figure_ratio:.0%}"
        )

        return result
