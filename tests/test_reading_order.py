# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
Unit tests — Reading Order Resolver
"""
from __future__ import annotations

import pytest

from src.assembly.reading_order import ReadingOrderResolver
from src.models import Block, BlockLabel, BoundingBox


def make_block(bid, label, x1, y1, x2, y2, page=0) -> Block:
    return Block(
        id=bid,
        label=label,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        page=page,
        confidence=0.9,
    )


def test_single_column_top_to_bottom():
    """Single-column document → blocks sorted top-to-bottom."""
    resolver = ReadingOrderResolver()
    blocks = [
        make_block(0, BlockLabel.TEXT, 50, 300, 500, 350),
        make_block(1, BlockLabel.TITLE,50, 100, 500, 150),
        make_block(2, BlockLabel.TEXT, 50, 200, 500, 250),
    ]
    ordered = resolver.sort(blocks)
    y_centers = [b.bbox.y_center for b in ordered]
    assert y_centers == sorted(y_centers), "Blocks should be sorted top-to-bottom"


def test_two_column_layout():
    """Two-column document → left column before right column."""
    resolver = ReadingOrderResolver()
    # Left column (x_center ~150)
    left_top    = make_block(0, BlockLabel.TEXT, 50,  100, 250, 150)
    left_bottom = make_block(1, BlockLabel.TEXT, 50,  200, 250, 250)
    # Right column (x_center ~550)
    right_top    = make_block(2, BlockLabel.TEXT, 400, 100, 700, 150)
    right_bottom = make_block(3, BlockLabel.TEXT, 400, 200, 700, 250)

    blocks = [right_bottom, left_top, right_top, left_bottom]
    ordered = resolver.sort(blocks)
    ids = [b.id for b in ordered]
    # Left column first (0, 1) then right column (2, 3)
    assert ids.index(0) < ids.index(1), "Left top before left bottom"
    assert ids.index(1) < ids.index(2), "Left column before right column"
    assert ids.index(2) < ids.index(3), "Right top before right bottom"


def test_header_footer_placement():
    """Headers should be first, footers last."""
    resolver = ReadingOrderResolver()
    blocks = [
        make_block(0, BlockLabel.TEXT,   50, 200, 500, 250),
        make_block(1, BlockLabel.FOOTER, 50, 900, 500, 950),
        make_block(2, BlockLabel.HEADER, 50,  10, 500,  50),
    ]
    ordered = resolver.sort(blocks)
    assert ordered[0].label == BlockLabel.HEADER
    assert ordered[-1].label == BlockLabel.FOOTER


def test_empty_input():
    resolver = ReadingOrderResolver()
    assert resolver.sort([]) == []


def test_column_assigned():
    """Column attribute must be set after sorting."""
    resolver = ReadingOrderResolver()
    blocks = [make_block(0, BlockLabel.TEXT, 50, 100, 200, 150)]
    ordered = resolver.sort(blocks)
    assert ordered[0].column is not None
