> **Note (2026-07-28):** a second pipeline variant, **structure-first** (`main_normal.ipynb`),
> is defined in **ADR 0008** — it drops RAM++/DINO/Imagen/SAM, inverts a finetuned BiRefNet to
> get the wall+floor+ceiling shell (visible-only), and splits surfaces with Metric3D-v2
> normal+depth plane clustering. The design below is the original empty-room (`run.py`,
> amodal) path.

# Wall / Floor surface-mask pipeline — design

End-to-end design produced by the grilling session. See `adr/` for the decision trail and
`glossary.md` for terms. Goal: per image, produce **wall** and **floor** masks (plus a
**rug** mask) that contain the surface only, with every object removed.

## Inputs
- `output-test/<sample>/downsample.jpg` — 3000×2000 image.
- `output-test/<sample>/auto_detect.json` — used for **surface seeds only**:
  `wall_bboxes`, `floor_bboxes`, `rug_bboxes`, `wall_selected_points`,
  `floor_selected_points`. (Object bboxes in the JSON are NOT used for exclusion — ADR 0001/0002.)

## Models (ADR 0004)
- **GroundingDINO** — HF `transformers`, `IDEA-Research/grounding-dino-base` *(verify id)*.
- **SAM (huge)** — HF `transformers`, `facebook/sam-vit-huge` *(verify id)*.
- **BiRefNet** — existing (`zhengpeng7/birefnet`).

## Stage 1 — Objects to remove (the union that gets cut from every surface)
1. **DINO** on the image with prompt
   `window. mirror. door. painting. picture. tv. poster. clock. curtain.` → boxes (+labels). (ADR 0007)
2. For each DINO box → **SAM box-prompt** → precise occluder mask. Union → `occluders`. (ADR 0003)
3. **BiRefNet** full-image + 4×4 tiled, ≥ threshold → `objects_3d` (class-agnostic 3-D
   catch-all: sofa, table, plant, etc.). (ADR 0002)
4. `objects_all = occluders ∪ objects_3d`. (ADR 0002)

## Stage 2 — Surfaces (SAM, points ∩ bbox) (ADR 0003)
- `wall_surface  = union(SAM point-prompt @ wall_selected_points)  ∩ wall_bbox_region`
- `floor_surface = union(SAM point-prompt @ floor_selected_points) ∩ floor_bbox_region`

## Stage 3 — Rug (kept deliverable) (ADR 0006)
- `rug_footprint = GrabCut(rug_bbox)` (actual shape, not the rectangle).
- `rug_mask = rug_footprint − objects_all`.

## Stage 4 — Final masks (ADR 0005)
- `wall_mask  = wall_surface  − objects_all`
- `floor_mask = floor_surface − rug_footprint − objects_all`
- (Optionally keep only blobs touching the corresponding selected points.)

## Outputs (ADR 0006)
- `wall_mask.png`, `floor_mask.png`, `rug_mask.png` per sample dir (non-destructive).
- Inspect with `show_sample()` overlay: wall=red, floor=green, rug=blue. (ADR 0007)

## Open risks / to verify at implementation
- **VRAM**: BiRefNet + DINO + SAM-huge co-resident on the RTX 5070; may need sequential
  load or CPU offload. (ADR 0004)
- **sm_120 / torch 2.12** compatibility of HF SAM/DINO kernels. (ADR 0004)
- **SAM negative points**: only positive surface points exist; bleeding is bounded by the
  bbox intersection, but adding negative points (e.g. object centroids) could sharpen edges.
- **DINO false positives**: mirror↔window, framed art↔window; needs a confidence threshold.
- **Wall/floor overlap**: SAM plane segmentation should separate them at the real seam,
  likely making the old "wall-priority" hack obsolete — verify against that case. (ADR 0003)
