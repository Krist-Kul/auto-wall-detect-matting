# ADR 0005 — Object subtraction rule

**Status:** Accepted (grilling)
**Date:** 2026-06-18

## Decision
Subtract the **union of all detected objects** (SAM-cut DINO occluders ∪ BiRefNet 3-D
foreground) from **both** the wall and floor surface masks. Anchoring (wall- vs
floor-anchored) is **descriptive / QA metadata only** and does not restrict subtraction.

## Rationale
A floor-anchored object can occlude the wall (sofa back, tall plant, bookshelf).
Restricting subtraction by anchoring would leave those pixels in the wall mask. Removing
every object from every surface is both simpler and more correct, because an object
pixel is never legitimately part of any surface.

## Consequences
- `wall = SAM_wall − objects_all`; `floor = SAM_floor − objects_all`.
- Anchoring labels still produced (from DINO classes) for reporting, not for masking.
