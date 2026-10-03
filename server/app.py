"""NOVA API foundation. Development-only job service."""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="NOVA API", version="0.4.0")

class JobState(str, Enum):
    queued = "queued"
    rendering = "rendering"
    complete = "complete"
    failed = "failed"

class JobCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    model: str = "nova-cinematic"
    duration_seconds: int = Field(default=5, ge=1, le=60)
    aspect_ratio: str = "16:9"
    camera: str = "auto"
    reference_mode: str = "none"
    quality: str = "draft"

class Job(BaseModel):
    id: str
    state: JobState
    request: JobCreate
    created_at: str
    output_url: str | None = None
    error: str | None = None

_jobs: dict[str, Job] = {}
_lock = Lock()

@app.get("/health")
def health():
    return {"ok": True, "service": "nova-api", "version": "0.4.0"}

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
    return job

@app.get("/v1/jobs/{job_id}", response_model=Job)
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
