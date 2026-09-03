# ADR 0001 — Detector scope: GroundingDINO vs existing auto_detect bboxes

**Status:** Accepted (grilling)
**Date:** 2026-06-18

## Context
`auto_detect.json` already provides `wall/floor/rug/door` bboxes, `*_selected_points`,
and `rug_to_floor_indices`. The proposed pipeline introduces GroundingDINO. We needed to
decide whether DINO augments, partially replaces, or fully replaces the existing data.

## Decision
**DINO is the single source for ALL object exclusions.** We keep from `auto_detect.json`
only the **surface regions and selected points** (wall/floor bboxes + points) used to
prompt/seed surface segmentation. Every object that must be *cut off* a surface
(window, mirror, door, sofa, table, plant, lamp, rug, …) comes from GroundingDINO, not
from the JSON's object bboxes.

## Consequences
- The JSON's `door_bboxes` / `rug_bboxes` are no longer the exclusion source (DINO
  re-detects these as objects). Avoids the door duplication.
- We still depend on `auto_detect.json` for wall/floor **surface** regions + points.
  **(open, Thread 3: confirm SAM is prompted by these points.)**
- A **closed-set risk** appears: DINO only finds classes we name, so "floor/wall without
  *any* object" depends on the prompt list's completeness. **(open, Thread 2.)**
- "Rug as its own mask" (previous GrabCut work) is demoted to just a floor-anchored
  object unless re-stated as a deliverable. **(open, confirm later.)**
