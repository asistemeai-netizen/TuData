# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — Full Page Renderer (Gemini Vision)
=============================================
Sends each full page image to Gemini Vision and gets back a complete
markdown representation that includes:
  - All text verbatim (preserving structure)
  - Inline descriptions of visual elements (diagrams, logos, photos)
  - Tables in markdown format
  - Notes, legends, revision blocks — everything visible on the page

This replaces the old block-based MarkdownBuilder approach with a
much more faithful page-by-page rendering, similar to Datalab/Marker.
"""
from __future__ import annotations

import io
import os
from typing import Optional

from loguru import logger
from PIL import Image


_PAGE_RENDER_PROMPT = """You are an expert document digitizer. Your task is to convert this document page image into a complete, faithful markdown representation.

**CRITICAL RULES:**
1. **Extract ALL text** exactly as it appears on the page. Do not summarize, skip, or paraphrase anything. Every word, number, label, note, and annotation must be included.
2. **Describe visual elements inline.** For diagrams, photos, logos, plans, and any non-text visual content, write a detailed description in italic text. Format: *Description of the visual element including all visible labels, equipment, measurements, and annotations.*
3. **Tables** must be rendered in proper markdown table format with all rows and columns preserved. Empty cells should be empty, not omitted.
4. **Preserve structure:**
   - Titles and headings should use markdown heading syntax (# ## ###)
   - Lists should use markdown list syntax (- or 1.)
   - Notes sections should be labeled
   - Revision blocks / title blocks should be rendered as tables
5. **Include EVERYTHING visible on the page.** This includes:
   - Drawing title blocks (drawing number, date, revision, scale, etc.)
   - Legends and symbols
   - Notes and references to other drawings
   - Company logos (describe them)
   - Stamps, revision clouds, callout labels
6. **Do NOT add commentary.** Only output the page content as markdown.
7. **Do NOT wrap the output in code blocks.** Output raw markdown directly.

Output the complete markdown representation of this page now:"""


_PAGE_RENDER_PROMPT_WITH_CONTEXT = """You are an expert document digitizer. Your task is to convert this document page into a complete, faithful markdown representation.

**Document context:** This is page {page_num} of {total_pages} from a {doc_type} document.

**CRITICAL RULES:**
1. **Extract ALL text** exactly as it appears on the page. Do not summarize, skip, or paraphrase. Every word, number, label, note, and annotation must be included.
2. **Describe visual elements inline.** For diagrams, photos, logos, plans, schematics, and any non-text visual content, write a detailed description in italic text that includes ALL visible labels, equipment names, measurements, wire numbers, component IDs, and annotations.
3. **Tables** must be rendered in proper markdown table format with ALL rows and columns. For very large tables (schedules, BOMs), include every single row — do not truncate.
4. **Preserve structure:**
   - Titles and headings → markdown heading syntax (# ## ###)
   - Lists → markdown list syntax (- or 1.)
   - Notes sections → labeled clearly
   - Drawing title blocks → rendered as tables or structured text
5. **Include EVERYTHING visible on the page:**
   - Drawing numbers, dates, revisions, scales
   - Legends and symbol definitions
   - Notes and cross-references to other drawings
   - Company logos (describe them briefly)
   - Revision history tables
   - Stamps, clouds indicating new work, callout labels
6. **For electrical/engineering diagrams specifically:**
   - List every component with its label/tag (e.g., "VCB-1", "CT-200/5")
   - Include wire numbers, terminal designations, equipment ratings
   - Describe the power flow / signal flow direction
   - Note switchgear sections, breaker positions, bus connections
7. **Do NOT add commentary or explanations.** Only output the page content.
8. **Do NOT wrap the output in code blocks.** Output raw markdown directly.

Output the complete markdown for this page:"""


class FullPageRenderer:
    """
    Renders each page of a document as complete markdown using Gemini Vision.

    This approach produces output similar to Datalab/Marker — faithful
    page-by-page text extraction with inline descriptions of visual elements.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 16384,
    ) -> None:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        key = api_key or os.getenv("GEMINI_API_KEY")
        self._client = genai.Client(api_key=key)
        self._model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self._max_tokens = max_tokens
        self._types = types
        logger.info(f"[PageRenderer] Gemini Vision backend: {self._model}")

    def render_page(
        self,
        page_image: Image.Image,
        page_num: int = 1,
        total_pages: int = 1,
        doc_type: str = "general",
    ) -> str:
        """
        Render a single page image as complete markdown.

        Args:
            page_image:   PIL Image of the full page.
            page_num:     Current page number (1-indexed).
            total_pages:  Total pages in the document.
            doc_type:     Document type for context-aware extraction.

        Returns:
            Complete markdown string for this page.
        """
        prompt = _PAGE_RENDER_PROMPT_WITH_CONTEXT.format(
            page_num=page_num,
            total_pages=total_pages,
            doc_type=doc_type.replace("_", " "),
        )

        img_bytes = self._pil_to_bytes(page_image)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    prompt,
                    self._types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                ],
                config=self._types.GenerateContentConfig(
                    max_output_tokens=self._max_tokens,
                    temperature=0.0,
                ),
            )
            text = response.text.strip() if response.text else ""
            # Remove any markdown code block wrappers if the model added them
            if text.startswith("```markdown"):
                text = text[len("```markdown"):].strip()
            if text.startswith("```"):
                text = text[3:].strip()
            if text.endswith("```"):
                text = text[:-3].strip()

            logger.info(
                f"[PageRenderer] Page {page_num}/{total_pages}: "
                f"{len(text)} chars extracted"
            )
            return text

        except Exception as exc:
            logger.error(f"[PageRenderer] Failed on page {page_num}: {exc}")
            return f"*[Page {page_num}: rendering failed — {exc}]*"

    def render_document(
        self,
        page_images: list[Image.Image],
        doc_type: str = "general",
        log_callback=None,
    ) -> str:
        """
        Render all pages of a document as a single markdown string.

        Each page is separated by a horizontal rule and page marker.
        """
        total = len(page_images)
        pages_md: list[str] = []

        for i, page_img in enumerate(page_images):
            page_num = i + 1
            if log_callback:
                log_callback(f"  [PageRenderer] Rendering page {page_num}/{total} ...")

            page_md = self.render_page(
                page_img,
                page_num=page_num,
                total_pages=total,
                doc_type=doc_type,
            )
            # Add page separator
            separator = f"{{{i}}}" + "\n" + "-" * 48
            pages_md.append(f"{separator}\n\n{page_md}")

        return "\n\n".join(pages_md)

    @staticmethod
    def _pil_to_bytes(image: Image.Image) -> bytes:
        """Convert PIL Image to high-quality JPEG bytes."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=95)
        buf.seek(0)
        return buf.read()
