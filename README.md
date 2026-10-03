# NOVA — AI Video Studio

**Current release: v0.9 — staged native capacity + creator-owned dataset strengthening**

NOVA is an independent AI-video product prototype.

## Product direction

The long-term goal is a proprietary AI-video stack. Third-party/open-source components may be used only when their licenses and terms permit commercial use. NOVA must not copy or extract proprietary model weights, APIs, or training data from another provider.

## v0.9 additions

- Staged native capacity profiles: Micro → Creator-16F Base → Creator-16F Large → Creator-24F Large
- Source-balanced latent training so overlapping windows from one source do not dominate optimization
- Explicit creator dataset provenance and rights-verification fields
- Chronological held-out validation window per source
- Creator Dataset v0.2: 4 independent creator-owned videos, 74 prepared windows, 70 train / 4 held out
- Native API defaults aligned to the 16-frame / 64×64 creator bridge profile
- Native export timing now follows requested duration unless an explicit FPS override is supplied
- Capacity profile regression tests added to native CI

Small CPU checkpoints remain engineering/regression artifacts, not production-quality video generators.

## Native research path

1. Validate creator ownership/permission and provenance.
2. Extract 4-second creator windows.
3. Train/evaluate the random-init autoencoder on held-out source windows.
4. Train the 16F Base latent denoiser with source-balanced sampling.
5. Compare prompt conditioning and temporal behavior.
6. Scale to 16F Large only after the Base stage improves on held-out data.
7. Add substantially more independent creator-owned sources before the 24F step.

Window count is never treated as source count.

## Existing product foundation

- Cinematic creator-first landing page
- Creative Director generation UI
- Prompt / model / duration / aspect / camera controls
- Backend job API contract in server/
- Development job lifecycle: queued / rendering / complete / failed
- Docker backend with ffmpeg
- Dataset provenance manifest validator
- Static GitHub Pages UI and future GPU backend separation

## $NVAI utility design

NOVA currently documents $NVAI as a **planned** payment utility. The intended structure is:

$NVAI payment → NOVA subscription → eligible revenue → monthly Year 1 buyback → 100% of acquired $NVAI burned

The first-year program is limited to **12 scheduled monthly burn events**. The buyback allocation percentage, total supply, launch structure, and other token parameters remain intentionally unspecified until the product, technical, and compliance design is finalized.

See TOKENOMICS.md for the current specification. The website must describe these mechanisms as planned until real on-chain implementation and verification exist.
