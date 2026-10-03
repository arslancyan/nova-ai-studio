"""NOVA API with asynchronous MP4 render-preview jobs.

The renderer is infrastructure only; the trained NOVA model will replace it later.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .renderer import render_image, render_preview

VERSION = "0.7.0"
OUTPUT_DIR = Path(os.getenv("NOVA_OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_BASE_URL = os.getenv("NOVA_PUBLIC_BASE_URL", "").rstrip("/")

app = FastAPI(title="NOVA API", version=VERSION)

allowed_origins = [x.strip() for x in os.getenv("NOVA_ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

class JobState(str, Enum):
    queued = "queued"
    rendering = "rendering"
    complete = "complete"
    failed = "failed"

class DirectorSpec(BaseModel):
    subject: str = Field(default="", max_length=1000)
    action: str = Field(default="", max_length=1000)
    environment: str = Field(default="", max_length=1000)
    camera_movement: str = Field(default="auto", max_length=200)
    lighting: str = Field(default="", max_length=500)
    style: str = Field(default="", max_length=1000)
    motion: str = Field(default="natural", max_length=100)
    negative_prompt: str = Field(default="", max_length=2000)

class JobCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    project_name: str = Field(default="Untitled NOVA project", max_length=120)
    seed: int | None = Field(default=None, ge=0, le=2**63-1)
    director: DirectorSpec = Field(default_factory=DirectorSpec)
    model: str = "nova-cinematic"
    duration_seconds: int = Field(default=5, ge=1, le=60)
    aspect_ratio: str = "16:9"
    camera: str = "auto"
    reference_mode: str = "none"
    quality: str = "draft"
    sampler: str = Field(default="ddpm", pattern="^(ddpm|ddim)$")
    sampling_steps: int | None = Field(default=None, ge=2, le=1000)
    frames: int = Field(default=8, ge=4, le=64)
    height: int = Field(default=32, ge=4, le=512)
    width: int = Field(default=32, ge=4, le=512)
    timesteps: int = Field(default=1000, ge=2, le=1000)

class Job(BaseModel):
    id: str
    state: JobState
    request: JobCreate
    created_at: str
    output_url: str | None = None
    error: str | None = None

_jobs: dict[str, Job] = {}
_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=max(1, int(os.getenv("NOVA_RENDER_WORKERS", "1"))))

def _set_job(job_id: str, **changes) -> None:
    with _lock:
        current = _jobs.get(job_id)
        if current:
            _jobs[job_id] = current.model_copy(update=changes)

def _render_job(job_id: str, request: JobCreate) -> None:
    _set_job(job_id, state=JobState.rendering)
    try:
        output = render_preview(
            job_id=job_id,
            prompt=request.prompt,
            duration_seconds=request.duration_seconds,
            aspect_ratio=request.aspect_ratio,
            camera=request.camera,
            seed=request.seed,
        )
        relative = f"/outputs/{Path(output).name}"
        output_url = f"{PUBLIC_BASE_URL}{relative}" if PUBLIC_BASE_URL else relative
        _set_job(job_id, state=JobState.complete, output_url=output_url)
    except Exception as exc:
        _set_job(job_id, state=JobState.failed, error=str(exc))

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "nova-api",
        "version": VERSION,
        "renderer": "nova-render-preview",
        "native_model": "not trained",
        "job_schema": "0.7",
        "director_schema": "0.1",
    }

@app.post("/v1/jobs", response_model=Job, status_code=202)
def create_job(request: JobCreate):
    job = Job(
        id=str(uuid4()),
        state=JobState.queued,
        request=request,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    with _lock:
        _jobs[job.id] = job
    _executor.submit(_render_job, job.id, request)
    return job


class ImageCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    aspect_ratio: str = "16:9"

class ImageResult(BaseModel):
    output_url: str

@app.post("/v1/images", response_model=ImageResult)
def create_image(request: ImageCreate):
    image_id = str(uuid4())
    try:
        output = render_image(
            job_id=image_id,
            prompt=request.prompt,
            aspect_ratio=request.aspect_ratio,
        )
        relative = f"/outputs/{Path(output).name}"
        output_url = f"{PUBLIC_BASE_URL}{relative}" if PUBLIC_BASE_URL else relative
        return ImageResult(output_url=output_url)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/v1/jobs/{job_id}", response_model=Job)
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
