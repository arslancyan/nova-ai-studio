# NOVA API — v0.9.1

The API is the bridge between the static GitHub Pages studio and the native NOVA engine.

## Endpoints

- GET /health — service state, schema version, native profile and readiness.
- POST /v1/preflight — validate a shot before consuming a generation worker.
- POST /v1/jobs — create a Creative Director generation job.
- GET /v1/jobs/{job_id} — poll job state, progress and output URL.
- POST /v1/images — deterministic development image preview.

## Native profile defaults

The native bridge now targets the Creator-16F Base profile:

- 16 frames
- 64×64 pixels
- 128 model dimension
- 4 attention heads
- 4 spatial transformer layers
- 1,000 diffusion steps

The API request still allows different frame/size values, but the selected
checkpoint must have a matching latent shape.

## Preview mode

Preview mode is the default:

    uvicorn server.app:app --host 0.0.0.0 --port 8000

It uses the infrastructure renderer and is not a native AI generation.

## Native mode

Native inference is opt-in and requires actual NOVA checkpoints:

    export NOVA_NATIVE_ENABLED=1
    export NOVA_NATIVE_PROFILE=creator-16f-base
    export NOVA_AE_CHECKPOINT=checkpoints/nova_ae.pt
    export NOVA_LATENT_CHECKPOINT=checkpoints/nova_latent.pt
    uvicorn server.app:app --host 0.0.0.0 --port 8000

Optional NOVA_NATIVE_FPS can override the automatic duration-based FPS.

Install the base server dependencies plus server/requirements-native.txt.

The API never silently falls back from native inference to the preview renderer. If native checkpoints are missing or incompatible, the job fails and reports the error.

## Production architecture

Browser → NOVA API → persistent job queue → GPU worker → NOVA inference engine → object storage/CDN.

The current API stores jobs in memory. It is a development integration, not a production multi-user service yet.

## Continuity Engine

Jobs can carry a continuity ID, shot index, continuity context and locked elements. The native prompt builder includes this context as structured conditioning text. This is an orchestration layer today; visual identity locking still requires a future reference-aware model.

The /v1/preflight endpoint performs structural checks before rendering, including prompt quality, continuity context and native profile shape.

## Security note

The frontend Admin button is a local prototype control, not authentication. Production authentication and authorization must be enforced by the backend, with no secret admin credential embedded in frontend JavaScript.
