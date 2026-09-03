# ADR 0006 — Outputs, rug deliverable, run scope

**Status:** Accepted (grilling)
**Date:** 2026-06-18

## Decisions
- **Outputs:** write **new PNG files** per sample dir (`wall_mask.png`, `floor_mask.png`,
  `rug_mask.png`). Non-destructive — `auto_detect.json` is not modified.
- **Rug:** keep a **separate rug mask** (GrabCut/SAM on `rug_bboxes`, minus objects) AND
  keep the rug **subtracted from the floor** mask. (Overrides the "drop rug" implication
  of ADR 0005; rug stays a deliverable.)
- **Scope:** keep the **single-sample interactive notebook** flow (one `SAMPLE` at a
  time). Batch-over-all-`output-test` is deferred.

## Consequences
- `show_sample()` overlay stays the inspection surface (wall=red, floor=green, rug=blue).
- A future batch ADR will be needed to process all 200+ folders.
