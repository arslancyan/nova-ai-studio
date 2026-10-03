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
- [x] Tiny CPU training profile for end-to-end native pipeline validation
- [x] CI executes a tiny random-init training profile and preserves its checkpoint artifact
- [x] CPU-first NOVA Micro profile (AE + latent diffusion)
- [x] Measure reconstruction quality on held-out synthetic clips

## Stage 1B — Data foundation
- [x] Synthetic dataset path
- [ ] Establish a small, documented real-video dataset
- [ ] Validate every real source license/permission
- [x] Normalize clips
- [x] Caption metadata field
- [x] Train/validation split
- [x] Real-video autoencoder training entry point
- [x] Real-video training resume/AMP path
- [x] Dataset checksum manifest
- [x] Real-video metadata/provenance validator
- [x] End-to-end real-video preparation smoke test (CI-generated media)

## Stage 2 — NOVA latent diffusion
- [x] Native latent denoiser architecture
- [x] Latent diffusion training entry point
- [x] Prepared real-video CSV input path for latent training
- [x] Latent denoiser smoke test
- [x] Train first tiny latent diffusion checkpoint through the CPU Micro profile
- [ ] Train first quality latent diffusion checkpoint (larger compute)
- [x] Loss curves
- [x] Reproducible seed
- [x] Validation loss / checkpoint metadata
- [x] Sample generation through the trained autoencoder (tiny validation profile)
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
