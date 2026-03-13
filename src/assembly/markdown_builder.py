# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Markdown Builder
==========================
Replicates Marker's Markdown assembly stage.

Converts a reading-order-sorted list of Blocks (with .text populated)
into a well-formed Markdown document.

Handles:
  - Titles / headings
  - Paragraphs and body text
  - Bullet lists
  - Tables (Markdown table syntax)
  - Figures (image references)
  - Formulas (LaTeX fenced blocks)
  - Headers / footers (skipped or placed in frontmatter)
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Optional

from loguru import logger

from src.models import Block, BlockLabel


class MarkdownBuilder:
    """
    Assembles Markdown from an ordered list of OCR-populated Blocks.

    Args:
        skip_headers_footers: If True, Header/Footer blocks are omitted.
        max_line_width:       Wrap long text lines at this column width.
    """

    def __init__(
        self,
        skip_headers_footers: bool = True,
        max_line_width: int = 100,
    ) -> None:
        self._skip_hf = skip_headers_footers
        self._max_width = max_line_width

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, blocks: list[Block], doc_title: Optional[str] = None) -> str:
        """
        Convert sorted, OCR-populated Blocks into a Markdown string.

        Args:
            blocks:     Reading-order-sorted list with .text set.
            doc_title:  Optional: prepend an H1 title.

        Returns:
            Full Markdown document as a string.
        """
        parts: list[str] = []

        if doc_title:
            parts.append(f"# {doc_title}\n")

        prev_label: Optional[BlockLabel] = None

        for block in blocks:
            if not block.text:
                continue

            label = block.label

            if self._skip_hf and label in (BlockLabel.HEADER, BlockLabel.FOOTER):
                continue

            md_fragment = self._render_block(block)
            if md_fragment is None:
                continue

            # Add blank line separator between different block types
            if prev_label is not None and prev_label != label:
                parts.append("")

            parts.append(md_fragment)
            prev_label = label

        markdown = "\n".join(parts).strip()
        logger.debug(f"Markdown built: {len(markdown)} chars from {len(blocks)} blocks.")
        return markdown

    # ------------------------------------------------------------------
    # Private renderers
    # ------------------------------------------------------------------

    def _render_block(self, block: Block) -> Optional[str]:
        """Route each block to its specific renderer."""
        text = (block.text or "").strip()
        if not text:
            return None

        match block.label:
            case BlockLabel.TITLE:
                return self._render_title(text, block)
            case BlockLabel.TEXT | BlockLabel.CAPTION:
                return self._render_paragraph(text)
            case BlockLabel.LIST:
                return self._render_list(text)
            case BlockLabel.TABLE:
                return self._render_table(text)
            case BlockLabel.FORMULA:
                return self._render_formula(text)
            case BlockLabel.FIGURE:
                return self._render_figure(text, block)
            case BlockLabel.HEADER | BlockLabel.FOOTER:
                return None  # already filtered above
            case _:
                return self._render_paragraph(text)

    def _render_title(self, text: str, block: Block) -> str:
        """Emit heading level based on font size heuristic (bbox height)."""
        height = block.bbox.height
        # Heuristic: taller = higher-level heading
        if height > 60:
            return f"# {text}"
        elif height > 40:
            return f"## {text}"
        else:
            return f"### {text}"

    def _render_paragraph(self, text: str) -> str:
        """Wrap paragraph text at max_line_width."""
        cleaned = re.sub(r"\s+", " ", text)
        return textwrap.fill(cleaned, width=self._max_width)

    def _render_list(self, text: str) -> str:
        """
        Convert list text into Markdown bullet points.
        Splits on common bullet characters or newlines.
        """
        # Normalize common OCR artifacts: •, ●, -, *, numbers followed by .
        text = re.sub(r"^[\s•●\-\*]+", "", text, flags=re.MULTILINE)
        items = [line.strip() for line in text.splitlines() if line.strip()]
        if not items:
            return ""
        return "\n".join(f"- {item}" for item in items)

    def _render_table(self, text: str) -> str:
        """
        If Gemini already returned Markdown table syntax, pass it through.
        Otherwise, attempt a naive reconstruction from newline-delimited rows.
        """
        # Already Markdown table
        if "|" in text and "---" in text:
            return text.strip()

        # Naive reconstruction: split rows by newline, columns by ≥2 spaces or tab
        rows = [line for line in text.splitlines() if line.strip()]
        if not rows:
            return text

        parsed_rows = [re.split(r"\t|  +", row.strip()) for row in rows]
        if not parsed_rows:
            return text

        col_count = max(len(r) for r in parsed_rows)

        def pad_row(row: list[str], n: int) -> list[str]:
            return row + [""] * (n - len(row))

        header = pad_row(parsed_rows[0], col_count)
        separator = ["---"] * col_count
        body = [pad_row(r, col_count) for r in parsed_rows[1:]]

        md_rows = [header, separator] + body
        return "\n".join("| " + " | ".join(row) + " |" for row in md_rows)

    def _render_formula(self, text: str) -> str:
        """Wrap formula in LaTeX block delimiters if not already wrapped."""
        text = text.strip()
        if not text.startswith("$$"):
            return f"$$\n{text}\n$$"
        return text

    def _render_figure(self, text: str, block: Block) -> str:
        """Emit a figure reference with available caption text."""
        caption = text[:120].strip()
        figure_ref = f"figure_{block.page + 1}_{block.id}"
        return f"![{caption}](artifacts/{figure_ref}.png)"

    # ------------------------------------------------------------------
    # Save utility
    # ------------------------------------------------------------------

    def save(self, markdown: str, output_path: str | Path) -> Path:
        """Write Markdown string to a file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        logger.success(f"Markdown saved: {out}")
        return out
