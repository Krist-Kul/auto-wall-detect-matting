# Glossary

Working vocabulary for the wall/floor surface-mask pipeline. Resolved during the grilling
session; see `adr/` and `pipeline-design.md`.

| Term | Definition |
|------|------------|
| **Surface** | A flat architectural plane we produce a clean mask for: **wall** and **floor** (plus **rug** as a kept side deliverable). |
| **Object / "stuff"** | Anything that is not the surface itself and must be removed from a surface mask. |
| **objects_all** | The union subtracted from every surface = SAM-cut DINO **occluders** ∪ BiRefNet **3-D foreground**. |
| **Occluder** | A **flat, wall-anchored** item DINO detects and SAM cuts: window, mirror, door, painting, picture, tv, poster, clock, curtain. BiRefNet cannot see these. |
| **3-D object** | A salient object BiRefNet catches class-agnostically (sofa, table, plant, ottoman, box, …), full-image + tiled. |
| **Wall-anchored / Floor-anchored** | Descriptive labels (anchoring) for where an object sits. **Used for QA only, not for masking** — every object is cut from both surfaces (ADR 0005). |
| **Cut off** | Subtract an object's pixels from a surface mask. |
| **Surface seed** | The JSON `*_selected_points` (+ `*_bboxes`) used to prompt SAM and bound the surface; the only thing we still take from `auto_detect.json`. |
| **rug_footprint** | The rug's true (non-rectangular) shape from GrabCut on `rug_bbox`; subtracted from floor and used as the rug mask base. |
| **auto_detect.json** | Existing per-image detections. We now use **only** its wall/floor/rug bboxes + selected points; its object bboxes and stored `wall_masks`/`floor_masks` are ignored. |
| **downsample.jpg** | The 3000×2000 input image each mask is computed against. |
| **GroundingDINO** | Open-vocab detector (HF transformers). Detects the flat occluder classes only. |
| **SAM** | Segment Anything (HF transformers, huge). Two roles: box-prompted occluder cut-masks; point-prompted (∩ bbox) surface masks. |
| **BiRefNet** | Salient-3-D-foreground segmenter; the class-agnostic object catch-all. Finetuned (`epoch_10_new.pth`) it segments movable objects only (ADR 0008). |
| **Structure mask** | `~objects` = the invert of the BiRefNet object matte: wall+floor+ceiling shell, **visible pixels only** (holes where objects were). ADR 0008. |
| **Structure-first** | The `main_normal` path: invert BiRefNet → structure, then split by normal/depth. No RAM++/DINO/Imagen/SAM. Opposite of `run.py`'s empty-room-then-segment path. |
| **Plane offset `d`** | Signed distance `point · n` of a surface pixel along its normal `n`. Splits **same-faced** walls that share a normal (grilling Q5); needs depth. ADR 0008. |
| **Normal cone (30°)** | Angular threshold: floor = normal within 30° of `up`, ceiling within 30° of `−up`, else wall; also the different-faced wall split. ADR 0008. |
| **Metric3D-v2** | One feed-forward net giving per-pixel metric depth + surface normal; replaces Marigold (no diffusion latency). ADR 0008. |
| **Amodal vs visible** | Amodal = full surface incl. pixels behind furniture (`run.py`, Imagen-filled). Visible = only unoccluded surface (`main_normal`, inversion). ADR 0008. |
