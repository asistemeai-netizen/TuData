# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Pydantic v2 Extraction Schemas
=========================================
Strict data models for structured JSON output.

These schemas are used by the LLM extractor to enforce output format
and serve as the contract for Qdrant insertion and RAG chunking.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Chunk type taxonomy ───────────────────────────────────────────────────────

class ChunkType(str, Enum):
    TEXT      = "text"
    TITLE     = "title"
    TABLE     = "table"
    FIGURE    = "figure"
    LIST      = "list"
    FORMULA   = "formula"


# ─── Table cell / row models ───────────────────────────────────────────────────

class TableCell(BaseModel):
    column: str
    value: str


class TableRecord(BaseModel):
    """One row from a detected table."""
    row_index: int
    cells: list[TableCell]


# ─── Entity & key-value extraction ────────────────────────────────────────────

class Entity(BaseModel):
    """A named entity found within a text chunk."""
    type: str       = Field(..., description="e.g. PERSON, ORG, DATE, AMOUNT")
    value: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class KeyValue(BaseModel):
    """A key–value pair extracted from structured content."""
    key: str
    value: str


# ─── Core document chunk ───────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """
    A single semantic chunk ready for Qdrant insertion.

    Corresponds to one or more layout blocks after reading-order assembly.
    """
    chunk_id:   str  = Field(..., description="Unique chunk identifier: '{doc_id}_p{page}_{idx}'")
    doc_id:     str  = Field(..., description="Source document identifier (stem of filename)")
    page:       int  = Field(..., ge=0)
    chunk_type: ChunkType
    text:       str  = Field(..., min_length=1)

    # Optional structured sub-data
    entities:   list[Entity]   = Field(default_factory=list)
    key_values: list[KeyValue] = Field(default_factory=list)
    table_rows: list[TableRecord] = Field(default_factory=list)

    # Metadata for RAG retrieval
    section:    Optional[str] = None   # e.g. nearest preceding heading
    bbox_repr:  Optional[str] = None   # "page=1,x1=50,y1=100,x2=500,y2=200"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # Added chunk enrichment fields
    summary:       Optional[str] = None
    keywords:      list[str]     = Field(default_factory=list)
    raw_ocr:       Optional[str] = None
    position:      Optional[str] = None
    doc_type_hint: Optional[str] = None

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


# ─── Document-level output ────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Top-level document properties."""
    doc_id:       str
    source_path:  str
    page_count:   int                = Field(..., ge=1)
    language:     Optional[str]      = None
    title:        Optional[str]      = None
    author:       Optional[str]      = None
    created_at:   Optional[str]      = None  # ISO 8601
    custom_meta:  dict[str, Any]     = Field(default_factory=dict)


class ExtractedDocument(BaseModel):
    """
    Top-level container for a fully processed document.

    Serializes to a single JSON file ready for Qdrant.
    """
    metadata: DocumentMetadata
    chunks:   list[DocumentChunk]

    def to_qdrant_points(self, embedding_dim: int = 768) -> list[dict]:
        """
        Convert chunks to Qdrant upsert-ready point dicts.
        (Embeddings must be added externally before upserting.)
        """
        return [
            {
                "id":      chunk.chunk_id,
                "payload": {
                    "doc_id":     chunk.doc_id,
                    "page":       chunk.page,
                    "type":       chunk.chunk_type.value,
                    "text":       chunk.text,
                    "section":    chunk.section,
                    "summary":    chunk.summary,
                    "keywords":   chunk.keywords,
                    "position":   chunk.position,
                    "doc_type_hint": chunk.doc_type_hint,
                    "entities":   [e.model_dump() for e in chunk.entities],
                    "key_values": [kv.model_dump() for kv in chunk.key_values],
                },
                # "vector": [...] to be filled by your embedding model
            }
            for chunk in self.chunks
        ]
