# NOVA Render Backend

NOVA v0.5 adds a real MP4 render-preview worker to validate the product pipeline.

Pipeline: Prompt -> POST /v1/jobs -> queued -> rendering -> complete -> MP4

The renderer is intentionally not presented as the trained NOVA AI model. It is a deterministic cinematic preview renderer used for infrastructure testing. The native NOVA model will replace it after the dataset/training milestones are completed.

## Run locally

Install server/requirements.txt and run: uvicorn server.app:app --reload

A working ffmpeg installation is required.

## API
- GET /health
- POST /v1/jobs
- GET /v1/jobs/{job_id}

When complete, output_url points to the generated MP4 under /outputs/.

## Container
The root Dockerfile installs ffmpeg and starts the API on port 8000.

Set NOVA_ALLOWED_ORIGINS to the exact frontend origin in production.
Set NOVA_PUBLIC_BASE_URL if the API should return absolute output URLs.
