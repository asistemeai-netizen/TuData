# Copyright 2026 Asisteme.AI
# Licensed under the Apache License, Version 2.0.
"""
TuData — FastAPI Web Server
============================
Exposes the pipeline as HTTP endpoints for the web UI.

Routes:
  POST /api/process         — Upload a PDF, start pipeline, return job_id
  GET  /api/status/{job_id} — Poll job status and progress
  GET  /api/result/{job_id} — Get full ExtractedDocument JSON
  GET  /api/markdown/{job_id} — Get assembled Markdown
  GET  /api/artifact/{job_id}/{filename} — Serve PNG artifacts
  GET  /                    — Serve the SPA (index.html)
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

load_dotenv()

app = FastAPI(title="TuData API", version="1.0.0")

import time

# ── Directories ───────────────────────────────────────────────────────────────
UPLOAD_DIR   = Path("uploads")
RESULTS_DIR  = Path("results")
ARTIFACTS_DIR = Path("artifacts")
WEB_DIR      = Path("web")

for d in [UPLOAD_DIR, RESULTS_DIR, ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── In-memory job store ───────────────────────────────────────────────────────
# job_id → { status, progress, stages, error, filename }
_jobs: dict[str, dict] = {}


# ── Pipeline stages for progress tracking ─────────────────────────────────────
STAGES = [
    "Intake & Clasificación",      # Stage 0: Analyze + classify project type
    "Detección de Layout",         # Stage 1: YOLOv8 block detection
    "Content Splitter",            # Stage 2: Separate text / tables / figures
    "Extracción Paralela",         # Stage 3: Text + Tables + Figures in parallel
    "Consolidación (LLM)",         # Stage 4: Gemini synthesizes ProjectDocument
]


# ─────────────────────────────────────────────────────────────────────────────
# Background pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(job_id: str, pdf_path: Path, visualize: bool) -> None:
    """Run full pipeline in background, updating job state at each stage."""
    job = _jobs[job_id]
    job["status"] = "running"
    job["start_time"] = time.time()
    job["current_stage"] = None
    job["metrics"] = {}

    def _log(msg: str):
        logger.info(msg)
        job["logs"].append(msg)

    def set_stage(idx: int):
        now = time.time()
        # End previous stage
        prev_idx = job["current_stage"]
        if prev_idx is not None and prev_idx < len(STAGES):
            job["stages"][prev_idx]["status"] = "done"
            job["stages"][prev_idx]["duration"] = round(now - job.get("_stage_start", job["start_time"]), 2)
            
        job["_stage_start"] = now
        job["current_stage"] = idx
        if idx < len(STAGES):
            job["stages"][idx]["status"] = "running"
            job["progress"] = int((idx / len(STAGES)) * 100)
            stage_name = STAGES[idx]
            _log(f"▶ Etapa {idx+1}/{len(STAGES)}: {stage_name}")

    try:
        _log(f"▶ Iniciando pipeline para: {pdf_path.name}")
        from src.pipeline import DocumentPipeline

        pipeline = DocumentPipeline(
            output_dir=str(RESULTS_DIR / job_id),
            artifacts_dir=str(ARTIFACTS_DIR / job_id),
            visualize=visualize,
        )

        # We run the full pipeline intercepting progress via callback
        doc = await pipeline.process_file(
            pdf_path, 
            progress_callback=set_stage,
            log_callback=_log,
            metrics=job["metrics"]
        )

        # Mark all stages complete, finalizing timers
        now = time.time()
        prev_idx = job["current_stage"]
        if prev_idx is not None and prev_idx < len(STAGES):
            job["stages"][prev_idx]["status"] = "done"
            job["stages"][prev_idx]["duration"] = round(now - job.get("_stage_start", job["start_time"]), 2)

        for s in job["stages"]:
            s["status"] = "done"
            
        job["progress"] = 100
        job["status"] = "complete"
        job["end_time"] = now
        job["total_duration"] = round(now - job["start_time"], 2)
        
        job["chunk_count"]    = len(doc.chunks)
        job["page_count"]     = doc.metadata.page_count
        job["project_type"]   = job["metrics"].get("project_type", "general_document")
        job["cls_confidence"] = job["metrics"].get("classification_confidence", 0)

        # Collect artifact filenames
        art_dir = ARTIFACTS_DIR / job_id
        if art_dir.exists():
            job["artifacts"] = [f.name for f in sorted(art_dir.glob("*.png"))]

        _log(f"✅ Pipeline completado: {len(doc.chunks)} chunks extraídos.")
        logger.success(f"[{job_id[:8]}] Pipeline complete — {len(doc.chunks)} chunks")

    except asyncio.CancelledError:
        logger.warning(f"[{job_id[:8]}] Pipeline cancelled by user.")
        _log("🛑 Proceso cancelado por el usuario.")
        job["status"] = "error"
        job["error"] = "Cancelado por el usuario."
        raise

    except BaseException as exc:
        tb = traceback.format_exc()
        logger.error(f"[{job_id[:8]}] Pipeline failed: {exc}\n{tb}")
        job["status"] = "error"
        error_msg = str(exc) or repr(exc) or "Error desconocido (sin mensaje)"
        job["error"] = error_msg
        _log(f"❌ ERROR: {error_msg}")
        _log(f"--- Traceback ---")
        for line in tb.strip().splitlines():
            _log(f"  {line}")


# ─────────────────────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/process")
async def process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    visualize: bool = True,
):
    """Upload a PDF and start the pipeline. Returns a job_id to poll."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Solo se aceptan archivos PDF.")

    job_id = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"

    # Save upload
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Initialize job
    _jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "current_stage": None,
        "filename": file.filename,
        "start_time": time.time(),
        "total_duration": 0.0,
        "stages": [{"name": s, "status": "pending", "duration": 0.0} for s in STAGES],
        "logs": [],
        "artifacts": [],
        "chunk_count": 0,
        "page_count": 0,
        "error": None,
    }

    # Run in background as explicit task to allow cancellation
    task = asyncio.create_task(run_pipeline(job_id, pdf_path, visualize))
    _jobs[job_id]["task"] = task
    
    logger.info(f"Job started: {job_id[:8]} — {file.filename}")
    return {"job_id": job_id, "filename": file.filename}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll job progress."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado.")
    
    # Clean internal keys and tasks
    out = {k: v for k, v in job.items() if k != "task" and not k.startswith("_")}
    
    if out["status"] == "running" and "start_time" in out:
        out["total_duration"] = round(time.time() - out["start_time"], 2)
        
    return out


@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running pipeline job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado.")
    
    if job["status"] not in ["running", "pending"]:
        return {"status": "ignored", "detail": "Job is not running."}

    task: asyncio.Task | None = job.get("task")
    if task and not task.done():
        task.cancel()
        
    job["status"] = "error"
    job["error"] = "Cancelado por el usuario."
    return {"status": "cancelled"}


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """Return the ExtractedDocument JSON."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "complete":
        raise HTTPException(404, "Resultado no disponible aún.")

    result_dir = RESULTS_DIR / job_id
    json_files = list(result_dir.glob("*.json"))
    if not json_files:
        raise HTTPException(404, "JSON no encontrado.")

    return FileResponse(
        json_files[0],
        media_type="application/json",
        filename=json_files[0].name,
    )


@app.get("/api/markdown/{job_id}")
async def get_markdown(job_id: str):
    """Return the assembled Markdown as plain text."""
    job = _jobs.get(job_id)
    if not job or job["status"] != "complete":
        raise HTTPException(404, "Resultado no disponible aún.")

    result_dir = RESULTS_DIR / job_id
    md_files = list(result_dir.glob("*.md"))
    if not md_files:
        raise HTTPException(404, "Markdown no encontrado.")

    return FileResponse(
        md_files[0],
        media_type="text/plain; charset=utf-8",
        filename=md_files[0].name,
    )


@app.get("/api/artifact/{job_id}/{filename}")
async def get_artifact(job_id: str, filename: str):
    """Serve a bounding-box artifact PNG."""
    art_path = ARTIFACTS_DIR / job_id / filename
    if not art_path.exists() or art_path.suffix != ".png":
        raise HTTPException(404, "Artifact no encontrado.")
    return FileResponse(art_path, media_type="image/png")


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs (for history panel)."""
    return [
        {
            "job_id": jid,
            "filename": j["filename"],
            "status": j["status"],
            "progress": j["progress"],
            "chunk_count": j.get("chunk_count", 0),
        }
        for jid, j in reversed(list(_jobs.items()))
    ]


# ── Serve the SPA ─────────────────────────────────────────────────────────────
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

@app.get("/", response_class=HTMLResponse)
async def serve_spa():
    index = WEB_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>web/index.html not found</h1>", status_code=404)
    return HTMLResponse(index.read_text(encoding="utf-8"))


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
