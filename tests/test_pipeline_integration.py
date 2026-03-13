# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
Integration test — Full Pipeline (requires GEMINI_API_KEY in .env + a sample PDF)
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.mark.skipif(
    not SAMPLE_PDF.exists(),
    reason="tests/fixtures/sample.pdf not found — add a real PDF to run integration tests",
)
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — integration test skipped",
)
def test_full_pipeline_on_sample_pdf(tmp_path):
    """
    End-to-end test: runs the full pipeline on a real PDF.

    Asserts:
    - Markdown output file is created and non-empty.
    - JSON output file is created, is valid JSON, and passes schema validation.
    - ExtractedDocument has metadata and at least one chunk.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from src.extraction.schemas import ExtractedDocument
    from src.pipeline import DocumentPipeline

    pipeline = DocumentPipeline(
        output_dir=str(tmp_path / "results"),
        artifacts_dir=str(tmp_path / "artifacts"),
        visualize=False,
    )

    doc = asyncio.run(pipeline.process_file(SAMPLE_PDF))

    # Check ExtractedDocument
    assert isinstance(doc, ExtractedDocument)
    assert doc.metadata.page_count >= 1
    assert len(doc.chunks) >= 1

    # Check Markdown output
    md_path = tmp_path / "results" / f"{SAMPLE_PDF.stem}.md"
    assert md_path.exists(), "Markdown file should be created"
    md_content = md_path.read_text(encoding="utf-8")
    assert len(md_content) > 10, "Markdown should have meaningful content"

    # Check JSON output
    json_path = tmp_path / "results" / f"{SAMPLE_PDF.stem}.json"
    assert json_path.exists(), "JSON file should be created"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    validated = ExtractedDocument.model_validate(data)
    assert validated.metadata.doc_id == SAMPLE_PDF.stem
    assert len(validated.chunks) >= 1
