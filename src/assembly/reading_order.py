# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Reading Order Resolver
=================================
Replicates Marker's geometric reading-order stage.

Sorts detected blocks into reading order by:
  1. Assigning each block to a column (X-centroid clustering)
  2. Sorting columns left-to-right
  3. Sorting blocks within each column top-to-bottom

Handles mixed single-column and multi-column layouts automatically.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from loguru import logger

from src.models import Block, BlockLabel


class ReadingOrderResolver:
    """
    Assigns reading order to a flat list of Blocks.

    Args:
        column_gap_ratio: Fraction of page width that defines a column gap.
                          Lower values → more columns detected.
        page_width:       Reference page width in pixels. Used for gap calc.
                          If None, inferred from block positions.
    """

    def __init__(
        self,
        column_gap_ratio: float = 0.05,
        page_width: Optional[float] = None,
    ) -> None:
        self._gap_ratio = column_gap_ratio
        self._page_width = page_width

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sort(self, blocks: list[Block]) -> list[Block]:
        """
        Return a new list of Blocks sorted into reading order.

        Column assignment is stored in block.column.
        """
        if not blocks:
            return []

        # Process each page independently
        by_page: dict[int, list[Block]] = defaultdict(list)
        for block in blocks:
            by_page[block.page].append(block)

        ordered: list[Block] = []
        for page_num in sorted(by_page.keys()):
            page_blocks = by_page[page_num]
            sorted_page = self._sort_page(page_blocks)
            ordered.extend(sorted_page)

        return ordered

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sort_page(self, blocks: list[Block]) -> list[Block]:
        """Sort a single page's blocks into reading order."""
        if not blocks:
            return []

        # Separate full-width elements (headers, footers) from body blocks
        headers_footers = [
            b for b in blocks if b.label in (BlockLabel.HEADER, BlockLabel.FOOTER)
        ]
        body_blocks = [
            b for b in blocks if b.label not in (BlockLabel.HEADER, BlockLabel.FOOTER)
        ]

        # Assign column groups to body blocks
        columns = self._assign_columns(body_blocks)

        # Sort columns left-to-right, then blocks within each column top-to-bottom
        sorted_body: list[Block] = []
        for col_idx, col_blocks in sorted(columns.items()):
            col_sorted = sorted(col_blocks, key=lambda b: b.bbox.y1)
            for block in col_sorted:
                block.column = col_idx
            sorted_body.extend(col_sorted)

        # Place headers at the top, footers at the bottom
        headers = sorted([b for b in headers_footers if b.label == BlockLabel.HEADER],
                         key=lambda b: b.bbox.y1)
        footers = sorted([b for b in headers_footers if b.label == BlockLabel.FOOTER],
                         key=lambda b: b.bbox.y1)

        return headers + sorted_body + footers

    def _assign_columns(self, blocks: list[Block]) -> dict[int, list[Block]]:
        """
        Cluster blocks into columns by X-centroid proximity.

        Returns a dict mapping column_index → list of blocks.
        """
        if not blocks:
            return {}

        # Determine effective page width
        if self._page_width:
            page_w = self._page_width
        else:
            page_w = max(b.bbox.x2 for b in blocks)

        gap = page_w * self._gap_ratio

        # Sort block centers and cluster into column groups
        centers = sorted(set(b.bbox.x_center for b in blocks))
        column_centers: list[float] = []
        for c in centers:
            if not column_centers or c - column_centers[-1] > gap:
                column_centers.append(c)

        def nearest_col(x: float) -> int:
            return min(range(len(column_centers)),
                       key=lambda i: abs(column_centers[i] - x))

        columns: dict[int, list[Block]] = defaultdict(list)
        for block in blocks:
            col = nearest_col(block.bbox.x_center)
            columns[col].append(block)

        logger.debug(f"Detected {len(columns)} column(s) on page.")
        return dict(columns)
