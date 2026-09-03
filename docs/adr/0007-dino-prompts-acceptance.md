# ADR 0007 — DINO prompt list & acceptance criteria

**Status:** Accepted (grilling)
**Date:** 2026-06-18

## Decisions
- **GroundingDINO prompt classes (flat wall-anchored occluders):**
  `window, mirror, door, painting, picture, tv, poster, clock, curtain`.
  (BiRefNet still catches all 3-D objects class-agnostically; this list only targets the
  flat items BiRefNet misses.)
- **Acceptance:** **visual overlay review** via `show_sample()` on sample images.
  Quantitative IoU vs the stored masks is out of scope (the stored masks are an unverified
  baseline).

## Consequences
- Prompt phrasing for HF GroundingDINO must be the dot-separated lowercase form
  (e.g. `"window. mirror. door. painting. picture. tv. poster. clock. curtain."`).
- A per-class box-confidence threshold will need tuning; mirrors/paintings are prone to
  false positives (reflections, framed art vs window). Tune during implementation.
