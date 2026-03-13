# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — CLI Entry Point
=========================
Usage:
    python main.py --input docs/ --output results/ --format json,md --workers 4
    python main.py --input report.pdf --output results/ --visualize
    python main.py --input docs/ --output results/ --ollama
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load .env before any other imports that need API keys
load_dotenv()


def configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{message}</cyan>"
        ),
        colorize=True,
    )
    logger.add(
        "logs/tudata.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )


def collect_pdfs(input_path: Path) -> list[Path]:
    """Return list of PDF files from a path (file or directory)."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            logger.error(f"Not a PDF: {input_path}")
            sys.exit(1)
        return [input_path]
    elif input_path.is_dir():
        pdfs = sorted(input_path.rglob("*.pdf"))
        if not pdfs:
            logger.error(f"No PDF files found in: {input_path}")
            sys.exit(1)
        logger.info(f"Found {len(pdfs)} PDF(s) in {input_path}")
        return pdfs
    else:
        logger.error(f"Path not found: {input_path}")
        sys.exit(1)


async def run(args: argparse.Namespace) -> None:
    from src.pipeline import DocumentPipeline  # lazy import after env is loaded

    pipeline = DocumentPipeline(
        output_dir=args.output,
        artifacts_dir=os.getenv("DEFAULT_ARTIFACTS_DIR", "artifacts"),
        visualize=args.visualize,
        ocr_lang=args.lang,
        workers=args.workers,
    )

    pdfs = collect_pdfs(Path(args.input))

    if len(pdfs) == 1:
        await pipeline.process_file(pdfs[0])
    else:
        results = await pipeline.process_batch(pdfs)
        logger.success(f"Batch complete: {len(results)}/{len(pdfs)} files processed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tudata",
        description="TuData — Document Processing Pipeline (Datalab equivalent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input docs/report.pdf --output results/
  python main.py --input docs/ --output results/ --visualize --workers 8
  python main.py --input docs/ --output results/ --ollama --lang es
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a PDF file or a directory of PDFs.",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.getenv("DEFAULT_OUTPUT_DIR", "results"),
        help="Output directory for .md and .json files. (default: results/)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=int(os.getenv("DEFAULT_WORKERS", "4")),
        help="Max concurrent files in batch mode. (default: 4)",
    )
    parser.add_argument(
        "--lang", "-l",
        default="en",
        help="OCR language code for PaddleOCR (e.g. en, es, ch). (default: en)",
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Save annotated bounding-box PNG artifacts.",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Force Ollama as LLM backend (ignores GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. (default: INFO)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)

    if args.ollama:
        os.environ["GEMINI_API_KEY"] = ""  # Force Ollama path

    logger.info("TuData Document Pipeline — Starting")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
