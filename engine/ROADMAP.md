# NOVA Native Model Roadmap

## Stage 0 — Architecture
- [x] Random initialization
- [x] Text tokenizer
- [x] Text conditioning
- [x] Temporal/spatial video tensor
- [x] Diffusion objective
- [x] Training entry point
- [x] Inference entry point
- [x] Synthetic smoke test

## Stage 1 — Native representation learning
- [x] Random-init 3D video autoencoder
- [x] Rights-safe synthetic dataset generator
- [x] Autoencoder training entry point
- [x] Best-checkpoint persistence and training metadata
- [x] Autoencoder architecture smoke test
- [ ] Train first synthetic reconstruction checkpoint (GPU quality run)
- [ ] Benchmark first rights-safe real-video reconstruction checkpoint
- [x] Run first local creator-owned reconstruction benchmark (4 creator-owned clips; 3 animation + 1 live-action)
- [x] Tiny CPU training profile for end-to-end native pipeline validation
- [x] CI executes a tiny random-init training profile and preserves its checkpoint artifact
- [x] CPU-first NOVA Micro profile (AE + latent diffusion)
- [x] Reuse Micro checkpoints in CI instead of retraining the same profile twice
- [x] Measure reconstruction quality on held-out synthetic clips

## Stage 1B — Data foundation
- [x] Synthetic dataset path
- [x] Establish a creator-owned real-video benchmark path (4 local clips supplied by creator; raw media remains outside Git)
- [x] Identify a candidate open-license source for the first real-video benchmark (manual per-clip verification still required)
- [x] Add conservative YouTube candidate rights gate (metadata-only; no automatic downloading)
- [ ] Validate every real source license/permission
- [x] Normalize clips
- [x] Caption metadata field
- [x] Train/validation split
- [x] Real-video autoencoder training entry point
- [x] Real-video training resume/AMP path
- [x] Dataset checksum manifest
- [x] Real-video metadata/provenance validator
- [x] End-to-end real-video preparation smoke test (CI-generated media)
- [x] Creator-owned media inspector with SHA-256 and FFprobe provenance manifest

## Stage 2 — NOVA latent diffusion
- [x] Native latent denoiser architecture
- [x] Latent diffusion training entry point
- [x] Prepared real-video CSV input path for latent training
- [x] Latent denoiser smoke test
- [x] Train first tiny latent diffusion checkpoint through the CPU Micro profile
- [ ] Train first quality latent diffusion checkpoint (larger compute)
- [x] Run creator-owned CPU latent micro benchmark on four local clips
- [x] Loss curves
- [x] Reproducible seed
- [x] Validation loss / checkpoint metadata
- [x] Sample generation through the trained autoencoder (tiny validation profile)
- [x] Prompt-conditioning regression with same-seed prompt separation
- [x] Creator-owned latent prompt-conditioning regression
- [ ] Production-quality sample generation from a trained checkpoint
- [x] Native tensor-to-MP4 export boundary

## Stage 3 — Improve the architecture
- [x] Factorized spatial/temporal attention
- [x] Learned positional encoding
- [ ] Dynamic positional encoding
- [x] Efficient attention
- [x] Stronger text conditioning
- [ ] Better sampling / scheduler comparison
- [ ] Image/reference conditioning

## Stage 4 — Product engine
- [ ] NOVA native inference API
- [ ] GPU worker
- [ ] Job queue
- [ ] Object storage
- [ ] Admin unlimited credits
- [ ] User credit accounting
- [ ] Safety and abuse controls


## Compute simplification
- [x] Separate CPU Micro research profile from GPU Scale profile
- [x] Make GPU a quality/scaling accelerator rather than a prerequisite for every engineering step

## 60-Day execution target
- [x] Days 1–15 foundation: native representation learning, deterministic tests, real-video ingestion and provenance gates
- [ ] Days 16–30: first rights-safe real-video reconstruction checkpoint and reconstruction benchmark
- [ ] Days 31–45: first meaningful native text-conditioned latent generation on real data
- [ ] Days 46–60: stable native demo path, temporal consistency benchmark, reference-conditioning experiment
