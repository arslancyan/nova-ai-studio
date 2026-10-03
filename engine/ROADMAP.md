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
- [x] Tiny CPU training profile for end-to-end native pipeline validation
- [x] Measure reconstruction quality on held-out synthetic clips

## Stage 1B — Data foundation
- [x] Synthetic dataset path
- [ ] Establish a small, documented real-video dataset
- [ ] Validate every real source license/permission
- [ ] Normalize clips
- [ ] Caption clips
- [ ] Train/validation split
- [x] Dataset checksum manifest

## Stage 2 — NOVA latent diffusion
- [x] Native latent denoiser architecture
- [x] Latent diffusion training entry point
- [x] Latent denoiser smoke test
- [ ] Train first latent diffusion checkpoint (requires real compute)
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
