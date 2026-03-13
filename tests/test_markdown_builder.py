# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
Unit tests — Markdown Builder
"""
from __future__ import annotations

import pytest

from src.assembly.markdown_builder import MarkdownBuilder
from src.models import Block, BlockLabel, BoundingBox


def make_block(bid, label, text, height=30, page=0) -> Block:
    b = Block(
        id=bid,
        label=label,
        bbox=BoundingBox(x1=50, y1=100, x2=500, y2=100 + height),
        page=page,
        confidence=0.95,
        text=text,
    )
    return b


def test_title_becomes_heading():
    builder = MarkdownBuilder()
    blocks = [make_block(0, BlockLabel.TITLE, "Annual Report 2025", height=50)]
    md = builder.build(blocks)
    assert md.startswith("#")
    assert "Annual Report 2025" in md


def test_text_block_rendered():
    builder = MarkdownBuilder()
    blocks = [make_block(0, BlockLabel.TEXT, "This is a paragraph of text.")]
    md = builder.build(blocks)
    assert "This is a paragraph of text." in md


def test_list_block_rendered_as_bullets():
    builder = MarkdownBuilder()
    blocks = [make_block(0, BlockLabel.LIST, "• Apple\n• Banana\n• Cherry")]
    md = builder.build(blocks)
    assert "- Apple" in md
    assert "- Banana" in md
    assert "- Cherry" in md


def test_formula_wrapped_in_latex():
    builder = MarkdownBuilder()
    blocks = [make_block(0, BlockLabel.FORMULA, r"E = mc^2")]
    md = builder.build(blocks)
    assert "$$" in md
    assert r"E = mc^2" in md


def test_table_passthrough():
    builder = MarkdownBuilder()
    table_md = "| Name | Age |\n| --- | --- |\n| Alice | 30 |"
    blocks = [make_block(0, BlockLabel.TABLE, table_md)]
    md = builder.build(blocks)
    assert "| Name | Age |" in md
    assert "| Alice | 30 |" in md


def test_empty_blocks_skipped():
    builder = MarkdownBuilder()
    blocks = [
        make_block(0, BlockLabel.TEXT, ""),
        make_block(1, BlockLabel.TEXT, "  "),
        make_block(2, BlockLabel.TEXT, "Real content."),
    ]
    md = builder.build(blocks)
    assert "Real content." in md
    # Only one non-empty block
    assert md.count("Real content.") == 1


def test_header_footer_skipped_by_default():
    builder = MarkdownBuilder(skip_headers_footers=True)
    blocks = [
        make_block(0, BlockLabel.HEADER, "Page Header"),
        make_block(1, BlockLabel.TEXT,   "Body text."),
        make_block(2, BlockLabel.FOOTER, "Page 1 of 10"),
    ]
    md = builder.build(blocks)
    assert "Page Header" not in md
    assert "Page 1 of 10" not in md
    assert "Body text." in md


def test_doc_title_prepended():
    builder = MarkdownBuilder()
    blocks = [make_block(0, BlockLabel.TEXT, "Content.")]
    md = builder.build(blocks, doc_title="My Document")
    assert md.startswith("# My Document")
