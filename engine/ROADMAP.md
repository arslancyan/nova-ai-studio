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
- [ ] Train first synthetic reconstruction checkpoint

## Stage 1B — Data
- [ ] Establish a small, documented dataset
- [ ] Validate every source license/permission
- [ ] Normalize clips
- [ ] Caption clips
- [ ] Train/validation split
- [ ] Dataset checksum manifest

## Stage 2 — NOVA latent diffusion training
- [ ] First real checkpoint
- [ ] Loss curves
- [ ] Reproducible seed
- [ ] Evaluation report
- [ ] Sample generation

## Stage 3 — Improve the architecture
- [ ] Better temporal blocks
- [ ] Latent representation
- [ ] Efficient attention
- [ ] Stronger text conditioning
- [ ] Better sampling
- [ ] Image/reference conditioning

## Stage 4 — Product engine
- [ ] NOVA inference API
- [ ] GPU worker
- [ ] Job queue
- [ ] Object storage
- [ ] Admin unlimited credits
- [ ] User credit accounting
- [ ] Safety and abuse controls
