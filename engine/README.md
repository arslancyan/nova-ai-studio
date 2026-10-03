# NOVA Engine — Research Core

NOVA Engine is the native research layer for the NOVA AI Video Studio.

## Principle

NOVA's native model starts from random initialization. We do not import pretrained video-model weights into the native checkpoint.

The first goal is NOT a giant production model. The goal is to establish a small, reproducible architecture that we control end-to-end.

## NOVA-0

NOVA-0 is a tiny text-conditioned video diffusion research model.

Pipeline:

Prompt
→ NOVA character tokenizer
→ text conditioning
→ temporal/spatial video denoiser
→ denoised low-resolution frame sequence

Initial research target:
- very small video tensors
- short clips
- low spatial resolution
- random initialization
- reproducible training
- explicit dataset provenance

## Why start small?

A tiny model lets us validate:
1. conditioning works,
2. temporal attention works,
3. training checkpoints are reproducible,
4. inference works end-to-end,
5. dataset licensing/provenance is documented.

Only after these are stable should model size increase.

## Important IP/data rule

The native NOVA checkpoint must only be trained on data that we have documented rights to use. Do not add third-party model weights or scraped copyrighted video datasets to the training directory.

## Research status

NOVA-0 is an engineering/research prototype, not yet a competitive video generator.


## Native job runner

Creative Director jobs can now be executed through the explicit native path:

`python -m engine.native_runner --job engine/examples/native_job.json --output outputs/native.mp4`

The runner requires trained NOVA autoencoder and latent checkpoints. It does **not** fall back to the preview renderer when checkpoints are missing. The pipeline is:

Creative Director JSON → native prompt conditioning → latent diffusion → native autoencoder decode → MP4 export.

The included example is a schema/integration fixture, not a claim of trained-model quality.


## Current creator-owned research path

The current local benchmark path can use four creator-owned source videos,
extract 4-second overlapping windows, and train a 16-frame 64x64
random-initialized autoencoder before latent diffusion. See
CREATOR_DATASET.md, REAL_TRAINING_PROFILE.md, CREATOR_BENCHMARK_RESULTS_v0.2.md,
and CREATOR_LATENT_16F_RESULTS.md.

Native inference now supports both the original DDPM sampler and a deterministic
DDIM-style sampler with a configurable number of sampling steps. This is a
research feature; it does not imply production-quality generation.
