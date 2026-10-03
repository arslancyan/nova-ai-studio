# NOVA Rights-Safe Real Video Dataset

This is the bridge from synthetic learning to real video learning.

## Goal

Build a small, documented collection of short videos that NOVA is explicitly allowed to use for machine-learning training and commercial product development.

A public URL is not enough. Every source needs documented permission/license terms that allow the intended ML use.

## Required record

Each source needs source ID, source/collection name, creator or rights holder, license or written permission, original URL or acquisition record, acquisition date, permitted use, attribution requirement, and notes about restrictions.

The repository template is engine/REAL_VIDEO_TEMPLATE.csv.

## Recommended first benchmark

Start small: 100–500 clips, 2–8 seconds each, 8–16 frames per training sample, 64×64 target resolution for the first real-video experiment, diverse motion, and captions written from source metadata/manual review.

Do not jump directly to a huge dataset. First prove the complete ingestion → validation → split → training path.

## Dataset layout

    data/real/
      videos/
      clips/
      metadata.csv
      train.csv
      val.csv
      manifest.json

Raw videos stay separate from prepared tensors so provenance can be audited.

## Training gate

A source is eligible only when its documentation is complete and its license/permission has been manually checked for ML training, commercial use, redistribution/processing requirements, attribution requirements, and geographic/platform restrictions.

The validation script checks documentation completeness; it does not make a legal determination.

## Important

Until real rights-safe videos are actually added, NOVA must continue using the synthetic dataset in CI. No copyrighted web video should be silently scraped into training.


## Acquisition rule for the first real benchmark

Do not download a dataset merely because it is publicly accessible. Before adding a source, record the exact license or written permission and manually confirm that the intended ML training and commercial product-development use is permitted or obtain explicit permission. Keep the original source URL and acquisition record in metadata.

The first benchmark should prefer a small number of clearly documented clips over a large ambiguous corpus. If a source has attribution requirements, preserve them in the metadata and in any public documentation.

The checksum manifest proves file identity/integrity only. It does not prove ownership, licensing, or permission to train.


## Candidate source identified — Prelinger open-license subset

A candidate source for the first real-video benchmark is the **Prelinger Archives open-license subset** published on Hugging Face by davanstrien.

Source: https://huggingface.co/datasets/davanstrien/prelinger-archives-open

The dataset description states that it contains videos filtered to Public Domain, CC0, CC BY, and CC BY-SA, excludes Non-Commercial licenses, and records the specific license for each video in metadata. The dataset is described as usable for research, machine learning, and creative projects.

### NOVA gate before ingestion

This source is a **candidate, not an automatic approval**. Before any clip enters `data/real/`, NOVA must record and manually verify:

1. exact source video identifier;
2. creator/rights holder when available;
3. exact license URL and license type;
4. permission for commercial use;
5. permission for modification/processing;
6. any attribution requirement;
7. any non-copyright restrictions that could affect model training or product use;
8. acquisition date and source URL;
9. SHA-256 checksum after acquisition.

Only individually verified clips should be copied into the training set and marked `training_rights_verified=true` in the project's provenance records.

### First benchmark target

Start with a very small manually reviewed subset (for example 10–20 clips) rather than downloading the full collection. The published dataset is much larger than what is needed for the first reconstruction experiment.

This keeps the first real-data experiment cheap, auditable, and reversible while we validate the complete pipeline.