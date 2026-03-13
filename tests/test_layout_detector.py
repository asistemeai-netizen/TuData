# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
Unit tests — Layout Detector
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.models import Block, BlockLabel, BoundingBox


def make_block(label=BlockLabel.TEXT, x1=10, y1=20, x2=200, y2=80, page=0) -> Block:
    return Block(
        id=0,
        label=label,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        page=page,
        confidence=0.95,
    )


@patch("src.detection.layout_detector.YOLO", autospec=True)
def test_detect_image_returns_blocks(mock_yolo_class):
    """Layout detector returns at least one block for a mock result."""
    from src.detection.layout_detector import LayoutDetector

    # Set up a fake YOLO result
    mock_box = MagicMock()
    mock_box.cls.item.return_value = 7   # TEXT label
    mock_box.conf.item.return_value = 0.91
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].tolist.return_value = [10.0, 20.0, 300.0, 80.0]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {7: "Text"}

    mock_model = mock_yolo_class.return_value
    mock_model.return_value = [mock_result]

    detector = LayoutDetector(model_path="dummy.pt")
    image = Image.new("RGB", (800, 1000), color=(255, 255, 255))
    blocks = detector.detect_image(image)

    assert len(blocks) >= 1
    assert blocks[0].label == BlockLabel.TEXT
    assert blocks[0].confidence == pytest.approx(0.91, rel=1e-3)


def test_bounding_box_properties():
    bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)
    assert bbox.width == 100
    assert bbox.height == 50
    assert bbox.x_center == 60.0
    assert bbox.y_center == 45.0
    assert bbox.area == 5000


def test_block_to_dict():
    block = make_block(label=BlockLabel.TABLE)
    d = block.to_dict()
    assert d["label"] == "Table"
    assert "bbox" in d
    assert d["confidence"] == 0.95
