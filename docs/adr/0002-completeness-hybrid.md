# ADR 0002 — Object completeness: hybrid BiRefNet catch-all + GroundingDINO

**Status:** Accepted (grilling) — amends ADR 0001
**Date:** 2026-06-18

## Context
GroundingDINO is closed-set per run, so "surface without *any* object" is unachievable
with DINO alone (unlisted objects survive). Also: window/mirror/door are flat on the wall
plane and are **not** detected by BiRefNet (a salient 3-D foreground model), whereas
sofa/table/plant **are** already caught by BiRefNet class-agnostically.

## Decision
The exclusion mask is a **union of two complementary detectors**:
- **BiRefNet** — class-agnostic catch-all for *any 3-D object* (sofa, table, plant, box,
  pet, …), full-image + tiled passes (existing work).
- **GroundingDINO** — for the *flat wall-anchored* classes BiRefNet misses
  (window, mirror, door, and similar flush items like poster/TV/painting).

"Without any object" is therefore defined as: **any 3-D foreground object + any named
flat occluder**. Residual misses = flat, unnamed wall items.

## Consequences
- Supersedes ADR 0001's "DINO is the single source": DINO now covers only the flat
  occluder gap; BiRefNet remains the general object remover.
- DINO's class list only needs the flat wall-anchored set (much shorter than "all
  objects"). **(open, Thread 4: finalize the list.)**
- Creates an **anchoring-attribution** question for unlabeled BiRefNet blobs.
  **(open, Thread 3.)**
- BiRefNet cannot cut the flat occluders precisely — raises *who produces the precise
  cut mask* for window/mirror/door. **(open, Thread 3: SAM vs BiRefNet for cutting.)**
