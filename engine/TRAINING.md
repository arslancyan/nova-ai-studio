# NOVA Native Training Strategy

NOVA is developed from random initialization. The repository separates **pipeline validation** from **quality training**.

## What can run without a GPU

CPU CI validates the complete native path at tiny scale:

1. Generate rights-safe synthetic clips.
2. Train a small native 3D autoencoder.
3. Train a tiny native latent denoiser.
4. Run native text-conditioned latent sampling.
5. Exercise the real-video ingestion path with CI-generated media.
6. Validate documented train/validation splits and provenance fields.
7. On public CI, run a tiny random-init training profile and retain the resulting checkpoint as an engineering artifact.
8. Decode the latent result through the native autoencoder.
6. Validate the resulting video tensor.
7. Export the tensor to MP4.

This proves that the components connect correctly. It does **not** prove production video quality.

## Scaling path

When GPU compute becomes available, the same checkpoints can be resumed and the profile increased:

- larger synthetic datasets
- more autoencoder epochs
- larger latent denoiser width/depth
- longer diffusion schedules
- real video data with documented ML-training rights
- prepared real-video train/validation CSVs can now feed the same native latent trainer
- real-video autoencoder training is reproducible and resumable
- longer clips and higher resolution
- image/reference conditioning
- temporal refinement and upscaling

Training scripts expose model width, depth, head count, diffusion steps, batch size, learning rate, seed, and resume checkpoints where applicable.

## Compute rule

GitHub Actions is used for deterministic code and learning-path validation, not for claiming that a production model has been trained. A real NOVA quality checkpoint requires appropriate GPU compute.

## Native definition

A generation is considered **native NOVA** only when the trained NOVA autoencoder and NOVA latent denoiser checkpoints are used for generation. The web preview renderer is never counted as native model generation.
