# ADR 0003 — SAM's two roles: occluder cut-masks and surface masks

**Status:** Accepted (grilling)
**Date:** 2026-06-18

## Context
SAM is class-agnostic and prompt-driven. We need precise masks both for the flat
occluders DINO finds and for the wall/floor surfaces themselves.

## Decision
- **Occluder cutting (Thread 3):** each GroundingDINO box → **SAM box-prompt** → precise
  occluder mask (window/mirror/door incl. round/odd shapes). Subtracted from the wall.
- **Surface segmentation (Thread 4):** **SAM point-prompt** with the JSON
  `*_selected_points`, union across points, then **intersect with that category's bbox
  region** so the mask cannot bleed into ceiling / adjacent surfaces.

BiRefNet is now used ONLY as the class-agnostic 3-D object catch-all (ADR 0002).

## Consequences
- SAM is loaded once and used for both object and surface masks.
- The earlier hard "wall wins the wall/floor bbox overlap" hack is likely obsolete:
  SAM's point-prompted plane segmentation should separate wall vs floor at the real seam.
  **(open: verify against the overlap case we previously fixed.)**
- Adds a hard dependency on a SAM checkpoint (none installed). **(open, Thread 5.)**
- Final surface masks:
  `wall = SAM_wall(points ∩ wall_bbox) − (SAM_occluders ∪ BiRefNet_objects)`;
  `floor = SAM_floor(points ∩ floor_bbox) − (BiRefNet_objects [∪ DINO floor items])`.
