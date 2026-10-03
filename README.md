# NOVA — AI Video Studio

**Current release: v0.8 — native inference adapter + creator-engine foundation + $NVAI utility design**

NOVA is an independent AI-video product prototype.

## v0.1
- Cinematic creator-first landing page
- Creative Director generation UI
- Prompt / model / duration / aspect / camera controls
- Character Lock / Director Mode / Storyboard / Social Export product concepts
- $9/month subscription floor
- Planned $NVAI payment utility discount of 57%
- Planned Year 1 buyback/burn program: 12 monthly burns, then automatic burn ends
- Demo-only generation flow; no external AI engine is connected yet

## Product direction
The long-term goal is a proprietary AI-video stack. Third-party/open-source components may be used only when their licenses and terms permit commercial use. NOVA must not copy or extract proprietary model weights, APIs, or training data from another provider.

## v0.4 additions
- Backend job API contract in `server/`
- Development job lifecycle: queued / rendering / complete / failed
- Real MP4 render-preview worker and downloadable output
- Docker backend with ffmpeg
- Dataset provenance manifest validator
- Clear separation between static GitHub Pages UI and future GPU backend

## v0.8 additions
- Explicit `NOVA_NATIVE_ENABLED=1` backend switch for native checkpoints
- Native Creative Director jobs can export MP4 directly through the API
- Preview renderer remains the default when native inference is disabled
- Health endpoint reports active renderer and native checkpoint configuration
- Optional native runtime dependencies are isolated in `server/requirements-native.txt`

## Next engineering milestones
1. Deploy the NOVA backend and connect the live frontend
2. Authentication + projects
3. Credits and billing
4. Job queue
5. Model adapter interface
6. First commercially permitted inference engine
7. Reference-image consistency pipeline
8. Video post-processing/upscale
9. Replace/augment components with proprietary NOVA models over time


## $NVAI utility design

NOVA currently documents $NVAI as a **planned** payment utility. The intended structure is:

**$NVAI payment → NOVA subscription → eligible revenue → monthly Year 1 buyback → 100% of acquired $NVAI burned**

The first-year program is limited to **12 scheduled monthly burn events**. The buyback allocation percentage, total supply, launch structure, and other token parameters remain intentionally unspecified until the product, technical, and compliance design is finalized.

See [TOKENOMICS.md](TOKENOMICS.md) for the current specification. The website must describe these mechanisms as planned until real on-chain implementation and verification exist.
