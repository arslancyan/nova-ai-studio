# NOVA Creator 16F Latent Micro Benchmark

- Independent source videos: 4
- Training windows: 45
- Held-out windows: 4
- Window length: 4 seconds
- Frames: 16
- Resolution: 64x64
- Latent shape: 8x4x16x16
- Latent denoiser: 32 model dimension, 1 layer, 4 heads
- Diffusion steps: 8
- Training epochs: 2
- Initialization: random

## Results

| Epoch | Train noise-prediction MSE | Validation MSE |
|---|---:|---:|
| 1 | 1.05474 | 1.02762 |
| 2 | 1.01070 | 1.02509 |

Same-noise prompt-conditioning separation:
- mean absolute output difference: 0.28341

The conditioning regression demonstrates that the denoiser output changes when
the prompt changes under the same random latent/timestep. This is a conditioning
path regression, not evidence of semantic video quality.

The model remains a micro research checkpoint. More independent source
diversity, more frames, longer training, better sampling and GPU-scale capacity
are required before production-quality generation can be evaluated.
