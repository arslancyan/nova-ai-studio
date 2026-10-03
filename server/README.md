# NOVA API — v0.4 foundation

Development API contract for the NOVA web studio.

POST /v1/jobs creates a generation job.
GET /v1/jobs/{job_id} returns job state.
GET /health reports service health.

Jobs are currently in memory. The future production path is Browser -> NOVA API -> Job Queue -> GPU Worker -> NOVA inference engine -> Object Storage.

This is not a hosted production backend yet.