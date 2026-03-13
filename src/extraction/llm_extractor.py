# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — LLM Extractor (Forge Extract)
========================================
Sends the assembled Markdown to an LLM to produce structured JSON
conforming to the ExtractedDocument Pydantic schema.

Primary: Gemini 2.0 Flash (JSON mode)
Fallback: Ollama (local model, if OLLAMA_BASE_URL is set)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional
from uuid import uuid4

from loguru import logger
from pydantic import ValidationError

from src.extraction.schemas import (
    ChunkType,
    DocumentChunk,
    DocumentMetadata,
    ExtractedDocument,
)

# ------------------------------------------------------------------
# Extraction prompt
# ------------------------------------------------------------------
_EXTRACTION_PROMPT = """You are an advanced document analysis AI. Analyze the following Markdown document
and extract highly structured information. Return a SINGLE valid JSON object with this exact schema:

{{
  "metadata": {{
    "doc_id": "{doc_id}",
    "source_path": "{source_path}",
    "page_count": {page_count},
    "language": "<detected language code, e.g. 'en'>",
    "title": "<document title or null>",
    "author": "<author name or null>",
    "created_at": null,
    "custom_meta": {{}}
  }},
  "chunks": [
    {{
      "chunk_id": "{doc_id}_p0_0",
      "doc_id": "{doc_id}",
      "page": 0,
      "chunk_type": "text",  // one of: text, title, table, figure, list, formula
      "text": "<exact text content>",
      "summary": "<1-2 sentence semantic summary of this chunk's content>",
      "keywords": ["keyword1", "keyword2", "keyword3"],
      "position": "<human readable location e.g. 'Page 1, top section'>",
      "doc_type_hint": "<detected document type e.g. 'invoice', 'quote', 'spec_sheet', 'report', etc>",
      "entities": [{{"type": "ENTITY_TYPE", "value": "entity", "confidence": 0.9}}],
      "key_values": [{{"key": "k", "value": "v"}}],
      "table_rows": [],
      "section": "<nearest preceding section heading or null>",
      "bbox_repr": null,
      "confidence": 1.0
    }}
  ]
}}

Rules:
1. Create one chunk per logical section (heading + its body text) or distinct element.
2. Tables must set chunk_type to "table" and populate table_rows.
3. Formulas must set chunk_type to "formula".
4. Extract entities: PERSON, ORG, DATE, AMOUNT, LOCATION, PRODUCT.
5. Extract key-value pairs for any labeled fields (e.g., "Invoice No: 123", "Date: 06/17/2025").
6. If the document is an invoice/quote, extract line items as key-values or table_rows.
7. For forms, capture all labeled fields as key_values.
8. For specifications, extract technical sections and part codes.
9. EVERY chunk MUST have a `summary`, `keywords`, `position`, and `doc_type_hint`.
10. chunk_id format: "{doc_id}_p{{page}}_{{index}}".
11. Return ONLY the JSON object, no explanation or markdown fencing.

DOCUMENT:
{markdown}
"""


class LLMExtractor:
    """
    Extracts structured data from assembled Markdown using an LLM.

    Primary backend: Gemini 2.0 Flash (JSON mode)
    Fallback:        Ollama (uses OLLAMA_BASE_URL + OLLAMA_MODEL env vars)

    Args:
        api_key:    Gemini API key (defaults to GEMINI_API_KEY).
        model:      Gemini model name (defaults to GEMINI_MODEL).
        use_ollama: Force Ollama even if Gemini key is present.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_ollama: bool = False,
    ) -> None:
        self._use_ollama = use_ollama or not (
            api_key or os.getenv("GEMINI_API_KEY")
        )

        if self._use_ollama:
            self._init_ollama()
        else:
            self._init_gemini(api_key, model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        markdown: str,
        doc_id: str,
        source_path: str,
        page_count: int = 1,
    ) -> ExtractedDocument:
        """
        Transform a Markdown document into a validated ExtractedDocument.

        Args:
            markdown:    Full assembled Markdown string.
            doc_id:      Document identifier (e.g., filename stem).
            source_path: Original file path string.
            page_count:  Number of pages in the source document.

        Returns:
            Validated ExtractedDocument Pydantic object.

        Raises:
            ValueError if LLM returns invalid JSON or fails schema validation.
        """
        prompt = _EXTRACTION_PROMPT.format(
            doc_id=doc_id,
            source_path=source_path,
            page_count=page_count,
            markdown=markdown,
        )

        logger.info(f"Extracting structured data from '{doc_id}' ...")

        raw_json = self._call_llm(prompt)
        return self._parse_and_validate(raw_json, doc_id, source_path, page_count)

    def save_json(self, doc: ExtractedDocument, output_path: str | Path) -> Path:
        """Serialize ExtractedDocument to a JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        logger.success(f"JSON saved: {out}")
        return out

    # ------------------------------------------------------------------
    # LLM backends
    # ------------------------------------------------------------------

    def _init_gemini(self, api_key: Optional[str], model: Optional[str]) -> None:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        key = api_key or os.getenv("GEMINI_API_KEY")
        self._gemini_client = genai.Client(api_key=key)
        self._gemini_model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._genai_types = types
        logger.info(f"LLM Extractor: Gemini backend ({self._gemini_model_name})")

    def _init_ollama(self) -> None:
        self._ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1")
        logger.info(
            f"LLM Extractor: Ollama backend "
            f"({self._ollama_model} @ {self._ollama_url})"
        )

    def _call_llm(self, prompt: str) -> str:
        """Dispatch to the active backend and return raw response text."""
        if self._use_ollama:
            return self._call_ollama(prompt)
        return self._call_gemini(prompt)

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini with JSON-only output mode (google-genai SDK)."""
        try:
            response = self._gemini_client.models.generate_content(
                model=self._gemini_model_name,
                contents=prompt,
                config=self._genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=16384,
                    temperature=0.1,
                ),
            )
            return response.text or ""
        except Exception as exc:
            logger.error(f"Gemini extraction failed: {exc}")
            raise

    def _call_ollama(self, prompt: str) -> str:
        """Call a local Ollama model via HTTP API."""
        import urllib.request

        payload = json.dumps({
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }).encode()

        req = urllib.request.Request(
            f"{self._ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
                return body.get("response", "")
        except Exception as exc:
            logger.error(f"Ollama extraction failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # Parsing & validation
    # ------------------------------------------------------------------

    def _parse_and_validate(
        self,
        raw: str,
        doc_id: str,
        source_path: str,
        page_count: int,
    ) -> ExtractedDocument:
        """Parse raw LLM JSON string into a validated ExtractedDocument."""
        # Strip markdown fences if the LLM included them despite instruction
        raw = re.sub(r"^```(?:json)?\n?", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\n?```$", "", raw.strip(), flags=re.MULTILINE)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(f"LLM returned invalid JSON: {exc}\nRaw: {raw[:500]}")
            # Return a minimal fallback document rather than crashing
            return self._fallback_document(raw, doc_id, source_path, page_count)

        try:
            return ExtractedDocument.model_validate(data)
        except ValidationError as exc:
            logger.warning(f"Schema validation failed (attempting repair): {exc}")
            return self._repair_document(data, doc_id, source_path, page_count)

    def _fallback_document(
        self, raw_text: str, doc_id: str, source_path: str, page_count: int
    ) -> ExtractedDocument:
        """Create a minimal valid document when LLM output is unparseable."""
        return ExtractedDocument(
            metadata=DocumentMetadata(
                doc_id=doc_id,
                source_path=source_path,
                page_count=page_count,
            ),
            chunks=[
                DocumentChunk(
                    chunk_id=f"{doc_id}_p0_0",
                    doc_id=doc_id,
                    page=0,
                    chunk_type=ChunkType.TEXT,
                    text=raw_text[:4000] if raw_text else "Extraction failed.",
                    confidence=0.1,
                )
            ],
        )

    def _repair_document(
        self, data: dict, doc_id: str, source_path: str, page_count: int
    ) -> ExtractedDocument:
        """Best-effort repair of partially valid LLM output."""
        meta_data = data.get("metadata", {})
        meta_data.setdefault("doc_id", doc_id)
        meta_data.setdefault("source_path", source_path)
        meta_data.setdefault("page_count", page_count)

        valid_chunks: list[DocumentChunk] = []
        for i, raw_chunk in enumerate(data.get("chunks", [])):
            try:
                raw_chunk.setdefault("chunk_id", f"{doc_id}_p0_{i}")
                raw_chunk.setdefault("doc_id", doc_id)
                raw_chunk.setdefault("page", 0)
                raw_chunk.setdefault("chunk_type", "text")
                valid_chunks.append(DocumentChunk.model_validate(raw_chunk))
            except ValidationError:
                logger.debug(f"Skipping invalid chunk #{i}")

        return ExtractedDocument(
            metadata=DocumentMetadata.model_validate(meta_data),
            chunks=valid_chunks,
        )
