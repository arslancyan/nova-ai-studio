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
- [x] Autoencoder architecture smoke test
- [ ] Train first synthetic reconstruction checkpoint
- [ ] Measure reconstruction quality on held-out synthetic clips

## Stage 1B — Data foundation
- [x] Synthetic dataset path
- [ ] Establish a small, documented real-video dataset
- [ ] Validate every real source license/permission
- [ ] Normalize clips
- [ ] Caption clips
- [ ] Train/validation split
- [ ] Dataset checksum manifest

## Stage 2 — NOVA latent diffusion
- [x] Native latent denoiser architecture
- [x] Latent diffusion training entry point
- [x] Latent denoiser smoke test
- [ ] Train first latent diffusion checkpoint (requires real compute)
- [x] Loss curves
- [x] Reproducible seed
- [x] Validation loss / checkpoint metadata
- [ ] Sample generation through the trained autoencoder

## Stage 3 — Improve the architecture
- [ ] Better temporal blocks
- [ ] Learned/dynamic positional encoding
- [ ] Efficient attention
- [ ] Stronger text conditioning
- [ ] Better sampling
- [ ] Image/reference conditioning

## Stage 4 — Product engine
- [ ] NOVA native inference API
- [ ] GPU worker
- [ ] Job queue
- [ ] Object storage
- [ ] Admin unlimited credits
- [ ] User credit accounting
- [ ] Safety and abuse controls
