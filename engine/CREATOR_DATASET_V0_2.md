# NOVA Creator Dataset v0.2

Local creator-owned expansion prepared for native research.

## Dataset snapshot

- Independent source videos: 4
- Prepared 4-second windows: 74
- Training windows: 70
- Held-out validation windows: 4
- Resolution: 64×64
- Frames per window: 16
- Window stride: 1 second
- Maximum windows per source: 24
- Rights status: creator-owned / self-created and owned
- No third-party training media is included

The 74 windows are derived from the same four source videos. They are not 74
independent works.

## Source coverage

| Source | Media | Train windows | Held out |
|---|---|---:|---:|
| creator-001 | animation | 23 | 1 |
| creator-002 | animation | 20 | 1 |
| creator-003 | animation | 23 | 1 |
| creator-004 | live-action | 4 | 1 |

The source imbalance is handled during latent training with source-balanced
sampling. Validation remains chronological and source-specific.

## What this improves

Compared with the earlier 49-window benchmark, v0.2 increases temporal
coverage from the existing creator-owned videos without misrepresenting the
number of independent sources.

The next meaningful dataset improvement is **new independent creator-owned
videos**, ideally covering:
- different characters and subjects
- indoor and outdoor environments
- close, medium and wide framing
- static, handheld and moving-camera shots
- slow and fast motion
- animation and live-action
- different lighting conditions

## Rights/provenance rule

Every source must carry creator/owner information and explicit permission or
ownership evidence. The training pipeline supports an explicit
training_rights_verified field and can require it with
--require-rights-verified.

Public availability alone is not treated as permission to train.
