# NOVA — AI Video Studio

**Current release: v0.5 — live MP4 render pipeline + native-engine foundation**

NOVA is an independent AI-video product prototype.

## v0.1
- Cinematic creator-first landing page
- Creative Director generation UI
- Prompt / model / duration / aspect / camera controls
- Character Lock / Director Mode / Storyboard / Social Export product concepts
- 2-week, monthly and annual prototype pricing
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
