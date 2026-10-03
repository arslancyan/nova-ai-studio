# NOVA Creator-Owned Benchmark v0.2

## Dataset

- Independent source videos: 4
- Animation sources: 3
- Live-action source: 1
- Extracted windows: 49
- Training windows: 45
- Held-out windows: 4
- Window length: 4 seconds
- Stride: 2 seconds
- Frames per window: 16
- Resolution: 64x64
- All source media is creator-owned and kept outside the public repository.

Window count is not independent-source count.

## Autoencoder experiment

Random-initialized NOVA 3D autoencoder, 8 latent channels, trained on the
45 training windows for 5 CPU epochs.

Training loss:
- epoch 1: 0.73620
- epoch 2: 0.44616
- epoch 3: 0.33818
- epoch 4: 0.30462
- epoch 5: 0.26326

Held-out reconstruction metrics:

| Source | MSE | L1 | PSNR dB | Temporal delta L1 |
|---|---:|---:|---:|---:|
| creator-003 | 0.11909 | 0.21180 | 9.24 | 0.08989 |
| creator-001 | 0.09483 | 0.19582 | 10.23 | 0.13177 |
| creator-002 | 0.13175 | 0.21855 | 8.80 | 0.13306 |
| creator-004 | 0.17333 | 0.25154 | 7.61 | 0.18376 |

These are engineering diagnostics, not a human-quality score.

## Interpretation

The reconstruction objective is learning on creator-owned material. The
remaining gap is substantial, especially for the live-action clip, so this
checkpoint should not be presented as production-quality video generation.

The next scaling step is more independent source diversity, followed by a
larger latent diffusion run. The benchmark also now has a deterministic DDIM
sampling path for future speed/quality comparisons.

## Provenance

Raw media and private local paths are intentionally excluded from Git.
