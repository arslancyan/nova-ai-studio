# NOVA Real Training Profile

## Profile A — Creator Windows

The first meaningful real-data experiment uses the four creator-owned videos as source material and extracts short overlapping windows.

Recommended defaults:
- 4 second windows
- 2 second stride
- 16 frames
- 64×64
- maximum 24 windows per source
- one held-out window per source for validation

This can turn the current four source videos into many training samples while preserving source provenance. It does not create new visual diversity; the four originals remain the actual information source.

## Training order

1. Window extraction.
2. Creator provenance validation.
3. Autoencoder reconstruction.
4. Held-out reconstruction metrics.
5. Latent diffusion.
6. Prompt-conditioning regression.
7. Native generation samples.
8. Add more rights-safe sources before scaling model capacity.

## Anti-overfitting rule

Never report the number of windows as the number of independent source videos. Reports must show both source_count and window_count.

## Quality target

The first goal is not photorealism. The first goal is measurable improvement in held-out reconstruction and stable temporal behavior compared with the untrained baseline.
