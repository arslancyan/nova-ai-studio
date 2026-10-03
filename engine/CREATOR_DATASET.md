# NOVA Creator-Owned Video Dataset

This is the first real-media path for videos created and owned by the NOVA project creator.

## Purpose

Creator-owned media provides a clean provenance path without relying on scraped web videos. The repository does not store private source videos; the files stay local and only metadata/reproducibility tooling belongs in Git.

## Benchmark design

The initial benchmark can contain animation clips and live-action clips owned by the creator. Each clip is categorized so later experiments can compare an animation-only subset with a mixed creator-owned subset.

## Manifest

Use engine/CREATOR_VIDEO_TEMPLATE.csv with:
- sample_id
- video_path
- creator
- ownership
- permission
- media_category
- caption
- notes

creator_video_inspector.py records SHA-256, codec, dimensions, frame rate, frame count when available, duration, source size, and aspect ratio.

The inspector is local-only and never downloads anything.

## Workflow

1. Keep original videos untouched.
2. Add local paths to the creator CSV.
3. Run the inspector and review the manifest.
4. Run prepare_video_dataset.py to create fixed training tensors.
5. Run validate_real_dataset.py.
6. Train the autoencoder first.
7. Only after reconstruction is measurable, train latent diffusion.

## Limitation

A few creator videos are enough to validate the engineering pipeline, but not enough to claim a production-quality generative model. The first objective is reproducible reconstruction and data/rights integrity.

## Aspect ratio

The current preparation path normalizes clips to requested dimensions with padding. Original dimensions/aspect ratio remain in the creator manifest.

## Privacy

Do not commit personal videos or private metadata to the public repository. Keep raw creator media outside Git.
