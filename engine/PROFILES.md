# NOVA compute profiles

NOVA is split into small profiles so GPU compute is not a prerequisite for
every engineering step.

## Micro — $0 / CPU

The default research profile for free development:

- 4 frames
- 16x16 pixels
- 24 synthetic clips
- 2 autoencoder epochs
- 32-dimensional latent denoiser
- 1 transformer layer
- 8 diffusion steps

Purpose: prove that the native pipeline can learn, checkpoint, resume and
sample without pretrained weights.

## Scale — GPU later

Increase the same architecture after the Micro profile is stable:

- 8+ frames
- 32x32 or 64x64
- larger dataset
- larger denoiser
- longer diffusion schedule
- documented real-video data

GPU is therefore a **quality/scaling accelerator**, not a prerequisite for
building NOVA's native software and research pipeline.

## Practical rule

Do not spend time waiting for hardware. Every new model component should
first be testable through Micro. Only promote a component to Scale after the
Micro regression is useful.
