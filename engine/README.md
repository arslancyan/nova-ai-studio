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
