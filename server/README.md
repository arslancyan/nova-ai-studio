# NOVA API — v0.8

The API is the bridge between the static GitHub Pages studio and the native NOVA engine.

## Endpoints

- `GET /health` — service state, schema version and native readiness.
- `POST /v1/jobs` — create a Creative Director generation job.
- `GET /v1/jobs/{job_id}` — poll job state, progress and output URL.
- `POST /v1/images` — deterministic development image preview.

## Preview mode

Preview mode is the default:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

It uses the infrastructure renderer and is not a native AI generation.

## Native mode

Native inference is opt-in and requires actual NOVA checkpoints:

```bash
export NOVA_NATIVE_ENABLED=1
export NOVA_AE_CHECKPOINT=checkpoints/nova_ae.pt
export NOVA_LATENT_CHECKPOINT=checkpoints/nova_latent.pt
export NOVA_NATIVE_FPS=8
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Install the base server dependencies plus `server/requirements-native.txt`.

The API never silently falls back from native inference to the preview renderer. If native checkpoints are missing or incompatible, the job fails and reports the error.

## Production architecture

Browser → NOVA API → persistent job queue → GPU worker → NOVA inference engine → object storage/CDN.

The current API stores jobs in memory. It is a development integration, not a production multi-user service yet.

## Security note

The frontend Admin button is a local prototype control, not authentication. Production authentication and authorization must be enforced by the backend, with no secret admin credential embedded in frontend JavaScript.
