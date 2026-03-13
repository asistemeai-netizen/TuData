# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
Unit tests — LLM Extractor (Gemini mocked)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.schemas import ChunkType, ExtractedDocument


MOCK_JSON = {
    "metadata": {
        "doc_id": "test_doc",
        "source_path": "/tmp/test_doc.pdf",
        "page_count": 2,
        "language": "en",
        "title": "Test Document",
        "author": None,
        "created_at": None,
        "custom_meta": {},
    },
    "chunks": [
        {
            "chunk_id": "test_doc_p0_0",
            "doc_id": "test_doc",
            "page": 0,
            "chunk_type": "text",
            "text": "This is the first paragraph of the document.",
            "entities": [{"type": "ORG", "value": "Acme Corp", "confidence": 0.95}],
            "key_values": [{"key": "Invoice No", "value": "INV-001"}],
            "table_rows": [],
            "section": "Introduction",
            "bbox_repr": None,
            "confidence": 0.98,
        }
    ],
}


@patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"})
@patch("src.extraction.llm_extractor.genai", create=True)
def test_extract_returns_valid_document(mock_genai):
    """LLM extractor returns a valid ExtractedDocument from mocked Gemini."""
    import google.generativeai as genai_module

    # Patch the GenerativeModel inside the module
    with patch("google.generativeai.GenerativeModel") as mock_model_cls:
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(MOCK_JSON)
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model

        with patch("google.generativeai.configure"):
            from src.extraction.llm_extractor import LLMExtractor
            extractor = LLMExtractor(api_key="fake-key")
            doc = extractor.extract(
                markdown="# Introduction\n\nThis is the first paragraph of the document.",
                doc_id="test_doc",
                source_path="/tmp/test_doc.pdf",
                page_count=2,
            )

    assert isinstance(doc, ExtractedDocument)
    assert doc.metadata.doc_id == "test_doc"
    assert len(doc.chunks) >= 1
    assert doc.chunks[0].chunk_type == ChunkType.TEXT
    assert "paragraph" in doc.chunks[0].text


def test_schema_validates_manually():
    """ExtractedDocument model_validate works on valid dict."""
    doc = ExtractedDocument.model_validate(MOCK_JSON)
    assert doc.metadata.title == "Test Document"
    assert doc.chunks[0].entities[0].type == "ORG"


def test_schema_chunk_text_stripped():
    """Chunk text is stripped of leading/trailing whitespace."""
    from src.extraction.schemas import DocumentChunk

    chunk = DocumentChunk(
        chunk_id="x_p0_0",
        doc_id="x",
        page=0,
        chunk_type=ChunkType.TEXT,
        text="   hello world   ",
    )
    assert chunk.text == "hello world"
