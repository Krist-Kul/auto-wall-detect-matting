# ADR 0008 — Structure-first surfaces: BiRefNet-invert + monocular normal/depth plane clustering

**Status:** Accepted (grilling) — supersedes the acquisition path of ADR 0001/0002/0003
for the `main_normal` pipeline; `run.py` (RAM++/DINO/Imagen/image2scene) stays as the
amodal variant.
**Date:** 2026-07-28

## Context
`run.py` reached wall/floor by: RAM++→DINO→BiRefNet remove objects → **Imagen** inpaint an
empty room → **image2scene API** segment surfaces on the empty room. Two problems drove a
rethink: (1) Imagen is slow and introduces **ghosting** (hallucinated content in the emptied
room) that image2scene then inherits; (2) it needs the whole detector+API stack.

A finetuned zero-shot BiRefNet (`ckpts/hypersim/epoch_10.pth`) segments *movable objects*
only; Hypersim's BG class set means door/window/mirror/ceiling are learned as **structure**,
not foreground. (The later `epoch_10_new.pth` re-train was **cancelled** — it regressed badly
on real images; see Verification.)

## Decision
Drop RAM++, GroundingDINO, Imagen, SAM and the image2scene API. Per image:
1. **BiRefNet (finetuned)** → object matte → **invert** → structure mask (wall+floor+ceiling),
   *visible pixels only*.
2. **Metric3D-v2** (one feed-forward net) → per-pixel **metric depth + surface normal**.
   Chosen over Marigold-normal: feed-forward (no diffusion latency, consistent with the
   "kill Imagen for speed" motive) and outputs depth+normal together.
3. Backproject structure pixels to a point cloud; split:
   - **floor / ceiling / wall** by `angle(normal, up)` with a **30° cone**;
   - **wall instances** by **HDBSCAN** on the feature `normal ⊕ W_OFFSET·(z-scored plane
     offset d)`, then 2-D connected components. `d = point · n` — the offset requires **depth**.
     (HDBSCAN fit on a 30k subsample, all wall px assigned by nearest cluster centroid —
     `approximate_predict` on ~1M px is too slow.)

## Rationale
Surface separation from monocular normals + depth is the standard plane-segmentation
recipe (normals + RANSAC / `(n,d)` clustering on the point cloud). The grilling's Q5 case —
two **same-faced** jogged sub-walls (see `user_2`, 4 walls) — has **identical normals**;
neither normal-angle nor 2-D connectivity can split them, but the **plane offset `d`** can.
Hence depth is mandatory, not optional. The 30° cone cleanly separates floor/ceiling/wall
(≈90° apart) and clearly different-faced walls; sub-walls below 30° are handled by `d`.

## Consequences
- **Visible-only.** Inversion cannot recover surface hidden behind furniture — every object
  leaves a hole. `run.py`'s `*_mask_amodal` (Imagen-filled) is **not** reproduced. Accepted
  as a known limit; add mask-space inpainting only if a downstream consumer needs amodal.
- **Inversion has no safety net**: a BiRefNet miss → object stays as fake surface; a false
  positive → hole. Quality rides entirely on the object model — `epoch_10.pth` (see Verification;
  the `epoch_10_new` re-train regressed and was cancelled) — see [[birefnet-hypersim-finetune]].
- Adds a 4th heavy net (Metric3D). VRAM on the RTX 5070 may require sequential load /
  offloading BiRefNet after step 1 (carries ADR 0004's open VRAM risk).
- **Ceiling** is now a first-class output (new vs ADR 0006's wall/floor/rug). Rug is dropped
  from this variant.
- No per-image seeds needed (no `auto_detect.json` selected_points/bboxes) — fully automatic.
- Knobs to tune on `user_2`: `MIN_CLUSTER_FRAC`, `W_OFFSET`, `NORMAL_CONE_DEG`, `UP_CAM` sign.
- Implemented in `main_normal.ipynb`.

## Verification (2026-07-28, `user_2`)
Full pipeline runs end-to-end. **`epoch_10.pth` structure frac = 0.995** — correctly leaves the
empty hallway as structure. Metric3D-v2 normals + HDBSCAN (4 clusters) cleanly split **left
wall / right wall / far wall / floor / ceiling** (8 instance masks after 2-D CC). The
separation method is validated. **The cancelled `epoch_10_new.pth` gave structure frac 0.34**
(marked 66% of the empty room as objects) — unusable; that measured the synth→real gap and is
why `epoch_10` is the chosen ckpt. Env notes: `mmengine` + `mmcv` shim (Config→mmengine; mmcv
1.x won't build on torch 2.12/cu130); `hdbscan` (sklearn 1.7.2); fit on 30k subsample +
nearest-centroid assign (not `approximate_predict`, too slow on ~1M px). Ran in `birefnet` env.

## References
- Metric3D-v2 — arXiv 2404.15506 (zero-shot metric depth + normal, one model).
- Plane extraction from normals + RANSAC on point clouds (ScienceDirect S0920548921001021;
  iterative super-surface removal PMC8198504).
