"""NOVA API with asynchronous preview and optional native-model jobs.

Preview rendering remains the safe default. Native inference is enabled explicitly
with NOVA_NATIVE_ENABLED=1 and documented checkpoints.
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
from engine.capacity_profiles import get_capacity_profile

NATIVE_ENABLED = os.getenv("NOVA_NATIVE_ENABLED", "0").lower() in {"1", "true", "yes"}
NATIVE_AE_CHECKPOINT = os.getenv("NOVA_AE_CHECKPOINT", "checkpoints/nova_ae.pt")
NATIVE_LATENT_CHECKPOINT = os.getenv("NOVA_LATENT_CHECKPOINT", "checkpoints/nova_latent.pt")
NATIVE_FPS = os.getenv("NOVA_NATIVE_FPS")
NATIVE_PROFILE = os.getenv("NOVA_NATIVE_PROFILE", "creator-16f-base")
try:
    _native_profile_spec = get_capacity_profile(NATIVE_PROFILE)
except ValueError:
    _native_profile_spec = get_capacity_profile("creator-16f-base")
    NATIVE_PROFILE = _native_profile_spec.name

VERSION = "0.9.1"
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


class ContinuitySpec(BaseModel):
    enabled: bool = True
    continuity_id: str = Field(default="", max_length=120)
    shot_index: int = Field(default=1, ge=1, le=999)
    context: str = Field(default="", max_length=3000)
    locked_elements: list[str] = Field(default_factory=list, max_length=12)


class ContinuitySpec(BaseModel):
    enabled: bool = True
    continuity_id: str = Field(default="", max_length=120)
    shot_index: int = Field(default=1, ge=1, le=999)
    context: str = Field(default="", max_length=3000)
    locked_elements: list[str] = Field(default_factory=list, max_length=12)


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
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    director: DirectorSpec = Field(default_factory=DirectorSpec)
    continuity: ContinuitySpec = Field(default_factory=ContinuitySpec)
    model: str = "nova-cinematic"
    duration_seconds: int = Field(default=5, ge=1, le=60)
    aspect_ratio: str = "16:9"
    camera: str = "auto"
    reference_mode: str = "none"
    quality: str = "draft"
    sampler: str = Field(default="ddpm", pattern="^(ddpm|ddim)$")
    sampling_steps: int | None = Field(default=None, ge=2, le=1000)
    # Native creator profile targets 16 frames at 64×64. Preview rendering
    # ignores these fields, so the API can use the same request contract.
    frames: int = Field(default=16, ge=4, le=64)
    height: int = Field(default=64, ge=4, le=512)
    width: int = Field(default=64, ge=4, le=512)
    timesteps: int = Field(default=1000, ge=2, le=1000)


class Job(BaseModel):
    id: str
    state: JobState
    request: JobCreate
    created_at: str
    progress: int = Field(default=0, ge=0, le=100)
    stage: str = "queued"
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
    _set_job(job_id, state=JobState.rendering, progress=18, stage="rendering")
    try:
        if NATIVE_ENABLED:
            from engine.native_runner import run_job

            output = OUTPUT_DIR / f"{job_id}.mp4"
            native_job = request.model_dump()
            native_job["autoencoder_checkpoint"] = NATIVE_AE_CHECKPOINT
            native_job["latent_checkpoint"] = NATIVE_LATENT_CHECKPOINT
            if NATIVE_FPS:
                native_job["fps"] = max(1, int(NATIVE_FPS))
            native_job["native_profile"] = NATIVE_PROFILE
            run_job(native_job, str(output))
        else:
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
        _set_job(
            job_id,
            state=JobState.complete,
            progress=100,
            stage="complete",
            output_url=output_url,
        )
    except Exception as exc:
        _set_job(
            job_id,
            state=JobState.failed,
            progress=100,
            stage="failed",
            error=str(exc),
        )


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "nova-api",
        "version": VERSION,
        "renderer": "nova-native" if NATIVE_ENABLED else "nova-render-preview",
        "native_model": "enabled" if NATIVE_ENABLED else "disabled",
        "native_profile": NATIVE_PROFILE if NATIVE_ENABLED else None,
        "native_shape": {
            "frames": _native_profile_spec.output_frames,
            "height": _native_profile_spec.output_height,
            "width": _native_profile_spec.output_width,
        } if NATIVE_ENABLED else None,
        "native_checkpoints": {
            "autoencoder": NATIVE_AE_CHECKPOINT,
            "latent": NATIVE_LATENT_CHECKPOINT,
        } if NATIVE_ENABLED else None,
        "job_schema": "0.9",
        "director_schema": "0.1",
        "native_ready": bool(
            NATIVE_ENABLED
            and Path(NATIVE_AE_CHECKPOINT).is_file()
            and Path(NATIVE_LATENT_CHECKPOINT).is_file()
        ),
    }


@app.post("/v1/preflight")
def preflight(request: JobCreate):
    """Validate a job before generation without consuming a render worker."""
    warnings = []
    errors = []
    if len(request.prompt.strip()) < 8:
        errors.append("Prompt is too short for a useful generation.")
    if request.continuity.enabled and not request.continuity.continuity_id:
        warnings.append("Continuity is enabled but no continuity_id was supplied.")
    if request.continuity.enabled and request.continuity.shot_index > 1 and not request.continuity.context.strip():
        warnings.append("Later continuity shots should carry continuity context.")
    if request.sampler == "ddim" and request.sampling_steps is not None and request.sampling_steps > 100:
        warnings.append("DDIM above 100 steps is usually unnecessary for the current research profile.")
    if NATIVE_ENABLED:
        if request.frames != 16 or request.height != 64 or request.width != 64:
            errors.append("Creator-16F native profile requires 16 frames at 64×64.")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "native_profile": NATIVE_PROFILE if NATIVE_ENABLED else None,
    }


@app.post("/v1/preflight")
def preflight(request: JobCreate):
    """Validate a job before generation without consuming a render worker."""
    warnings = []
    errors = []
    if len(request.prompt.strip()) < 8:
        errors.append("Prompt is too short for a useful generation.")
    if request.continuity.enabled and not request.continuity.continuity_id:
        warnings.append("Continuity is enabled but no continuity_id was supplied.")
    if request.continuity.enabled and request.continuity.shot_index > 1 and not request.continuity.context.strip():
        warnings.append("Later continuity shots should carry continuity context.")
    if request.sampler == "ddim" and request.sampling_steps is not None and request.sampling_steps > 100:
        warnings.append("DDIM above 100 steps is usually unnecessary for the current research profile.")
    if NATIVE_ENABLED:
        expected = (_native_profile_spec.output_frames, _native_profile_spec.output_height, _native_profile_spec.output_width)
        actual = (request.frames, request.height, request.width)
        if actual != expected:
            errors.append(
                f"{NATIVE_PROFILE} requires {expected[0]} frames at "
                f"{expected[1]}×{expected[2]}."
            )
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "native_profile": NATIVE_PROFILE if NATIVE_ENABLED else None,
    }


@app.post("/v1/jobs", response_model=Job, status_code=202)
def create_job(request: JobCreate):
    job = Job(
        id=str(uuid4()),
        state=JobState.queued,
        request=request,
        created_at=datetime.now(timezone.utc).isoformat(),
        progress=5,
        stage="queued",
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
