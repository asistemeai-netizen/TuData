# ============================================================
# TuData — Document Processing Pipeline
# ============================================================

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow)](LICENSE)

**A local, modular Python pipeline that replicates Datalab's document intelligence architecture.**

</div>

---

## Architecture

```
PDF Input
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1 — Layout Detection  (src/detection/)            │
│  YOLOv8 (DocLayNet) → Block list with bounding boxes     │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐            │
│  │ Text   │ │ Table  │ │ Figure │ │ Title  │  ...        │
│  └────────┘ └────────┘ └────────┘ └────────┘            │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2 — OCR  (src/ocr/)                               │
│  PaddleOCR 3.x (PP-OCRv5)  — primary                    │
│  Gemini 2.0 Flash Vision   — fallback for tables/formulas│
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3 — Reading Order  (src/assembly/)                │
│  Geometric column clustering → left-to-right, top-bottom │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 4 — Markdown Assembly  (src/assembly/)            │
│  # Headings · paragraphs · - lists · | tables | · $$LaTeX│
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 5 — Structured JSON Extraction  (src/extraction/) │
│  Pydantic v2 schemas + Gemini JSON mode / Ollama         │
│  → ExtractedDocument  →  Qdrant-ready chunks             │
└──────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for Windows GPU users**: Replace `paddlepaddle` with `paddlepaddle-gpu` in `requirements.txt`.

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Run the pipeline

```bash
# Single file
python main.py --input docs/report.pdf --output results/

# Full directory, with visualizations
python main.py --input docs/ --output results/ --visualize --workers 4

# Spanish document
python main.py --input docs/contrato.pdf --output results/ --lang es

# Use local Ollama instead of Gemini
python main.py --input docs/ --output results/ --ollama
```

---

## Project Structure

```
TuData/
├── main.py                      # CLI entry point
├── requirements.txt
├── .env.example
├── pytest.ini
│
├── src/
│   ├── models.py                # Shared Block / BoundingBox dataclasses
│   ├── pipeline.py              # Async DocumentPipeline orchestrator
│   │
│   ├── detection/
│   │   ├── layout_detector.py   # YOLOv8 layout detection
│   │   └── visualizer.py        # Bounding-box PNG artifacts
│   │
│   ├── ocr/
│   │   ├── ocr_engine.py        # PaddleOCR 3.x (PP-OCRv5)
│   │   └── gemini_ocr.py        # Gemini 2.0 Flash Vision fallback
│   │
│   ├── assembly/
│   │   ├── reading_order.py     # Geometric column sort
│   │   └── markdown_builder.py  # Block → Markdown
│   │
│   └── extraction/
│       ├── schemas.py           # Pydantic v2 models (Qdrant-ready)
│       └── llm_extractor.py     # Gemini / Ollama JSON extraction
│
├── tests/
│   ├── fixtures/                # Place sample.pdf here for integration tests
│   ├── test_layout_detector.py
│   ├── test_reading_order.py
│   ├── test_markdown_builder.py
│   ├── test_extractor.py
│   └── test_pipeline_integration.py
│
├── results/                     # Output .md and .json files
└── artifacts/                   # Annotated bounding-box PNG images
```

---

## Outputs

| File | Description |
|------|-------------|
| `results/<name>.md` | Clean, hierarchical Markdown document |
| `results/<name>.json` | Validated ExtractedDocument JSON (Qdrant-ready) |
| `artifacts/<name>_page_NNN.png` | Annotated page images (with `--visualize`) |

### JSON Schema (abbreviated)

```json
{
  "metadata": { "doc_id": "...", "title": "...", "page_count": 5 },
  "chunks": [
    {
      "chunk_id": "report_p0_0",
      "chunk_type": "text",
      "page": 0,
      "text": "...",
      "entities": [{ "type": "ORG", "value": "Acme Corp", "confidence": 0.95 }],
      "key_values": [{ "key": "Invoice No", "value": "INV-001" }]
    }
  ]
}
```

---

## Running Tests

```bash
# Unit tests only (no API key or PDF needed)
pytest tests/ -m "not integration" -v

# Full integration test (requires .env with GEMINI_API_KEY + tests/fixtures/sample.pdf)
pytest tests/test_pipeline_integration.py -v
```

---

## Datalab Component Mapping

| Datalab | TuData Module | Technology |
|---------|--------------|------------|
| **Surya** (layout) | `src/detection/` | YOLOv8 (DocLayNet) + PyMuPDF |
| **Chandra** (OCR) | `src/ocr/` | PaddleOCR 3.x PP-OCRv5 + Gemini Vision |
| **Marker** (assembly) | `src/assembly/` | Geometric column sort + Markdown builder |
| **Forge Extract** (JSON) | `src/extraction/` | Pydantic v2 + Gemini 2.0 Flash / Ollama |

---

## License

Apache 2.0 — © 2026 Asisteme.AI
