# NOVA compute profiles

NOVA is split into staged profiles so free CPU work validates the pipeline
before larger native models consume GPU time. All native profiles use random
initialization; no pretrained video backbone is imported.

## Micro — $0 / CPU

- 4 frames
- 16×16 pixels
- 24 synthetic clips
- 32-dimensional denoiser
- 1 transformer layer
- 8 diffusion steps

Purpose: regression, checkpoint/resume, sampling and prompt-conditioning tests.

## Creator-16F Base — first real-data capacity step

- 16 output frames
- 64×64 pixels
- creator-owned windows
- 128 model dimension
- 4 attention heads
- 4 spatial transformer layers
- 2 temporal transformer layers
- 1,000 diffusion steps
- source-balanced window sampling
- one held-out window per source

This is the first meaningful quality bridge. It should be trained only after
held-out autoencoder reconstruction is stable.

## Creator-16F Large — GPU capacity step

- same 16-frame / 64×64 data target
- 192 model dimension
- 6 attention heads
- 6 spatial transformer layers
- 3 temporal transformer layers
- 1,000 diffusion steps

The larger denoiser increases representational capacity without changing the
data provenance rules. It is a training target, not a shipped checkpoint.

## Creator-24F Large — temporal capacity step

- 24 output frames
- 64×64 pixels
- 192 model dimension
- 6 attention heads
- 6 spatial transformer layers
- 3 temporal transformer layers
- 1,000 diffusion steps

This requires a matching 24-frame autoencoder checkpoint. It is deliberately
separate from the 16-frame checkpoint so shape mismatches cannot silently pass.

## Promotion rule

Promote one step at a time:

1. Micro regression is green.
2. Creator-owned dataset passes provenance checks.
3. 16F autoencoder improves on held-out windows.
4. 16F Base latent model improves against its own held-out validation.
5. Only then train 16F Large on a GPU.
6. Add substantially more independent creator-owned sources before 24F Large.

Window count is never treated as source count. Overlapping windows from one
video improve training coverage but do not create new visual diversity.
