# NOVA Creator-Owned Training Profile

## Current source set

The current local benchmark contains **4 independent creator-owned source
videos** and 49 extracted windows (45 train / 4 held out). The source videos
remain the actual independent information sources.

## Window policy

Default:
- 4-second windows
- 2-second stride
- 16 frames
- 64×64
- maximum 24 windows per source
- final chronological window held out from each source

The preparation script now preserves:
- source_id
- source duration
- window index/start
- creator
- ownership
- permission evidence
- media category
- caption
- training_rights_verified status

When --require-rights-verified is used, every source must explicitly contain
training_rights_verified=true. The pipeline does not infer legal permission
from a public URL, a checksum, or the fact that a file exists locally.

## Dataset strengthening

### 1. Source balance

Overlapping windows are sampled with inverse-frequency source weights during
latent training. A 24-window source therefore cannot dominate a 3-window
source simply because it has more extracted windows.

### 2. Chronological holdout

The final window from every source is validation-only. This is intentionally
stricter than a random window split because neighboring windows are highly
correlated.

### 3. Provenance

Every source should have a durable record containing:
creator/owner, permission or license evidence, source URL when applicable,
acquisition date, permitted use, attribution requirement, and notes.

### 4. Diversity target

The next data milestone is **more independent creator-owned sources**, not just
more windows. Aim for multiple subjects, environments, camera styles, motion
types and media categories before increasing the model again.

## Training order

1. Provenance validation.
2. Window extraction.
3. Held-out autoencoder reconstruction.
4. Source-balanced latent training.
5. Prompt-conditioning regression.
6. Native generation samples.
7. Compare Base vs Large on the same held-out sources.
8. Add new creator-owned sources before the next capacity step.

## Quality target

The goal of the first stages is measurable improvement in held-out
reconstruction, temporal consistency and prompt-conditioning behavior. Small
CPU checkpoints are engineering/regression artifacts, not production-quality
video generators.
