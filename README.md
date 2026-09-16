# auto-wall-detect-matting

Automatic **wall / floor surface masks** for a single room photo — the surface only, with
every object (furniture, rug, wall-hung items) removed — plus an optional **matting** pass
that turns the hard mask into a soft, edge-accurate alpha.

Two families of pipeline live here:

| Family | Idea | Gives you |
| --- | --- | --- |
| **Empty-room (amodal)** | Detect objects → generative-inpaint them away → segment the *bare* room | **Amodal** surfaces: includes wall/floor hidden *behind* furniture |
| **Structure-first (visible)** | Invert an object segmenter → get the wall+floor+ceiling shell → split it with monocular normals + depth | **Visible** surfaces only, no generative model, no external API |

> **The recommended pipeline is [`main_gd.ipynb`](main_gd.ipynb)** — the empty-room path
> with RAM++ → GroundingDINO → fine-tuned BiRefNet → Imagen 3 → image2scene, plus MoGe
> geometry running in parallel. Everything else in this repo is either an ablation of it,
> a downstream refinement, or the training that produced its BiRefNet checkpoint.

**New to the project?** Read [`docs/SUMMARY.md`](docs/SUMMARY.md) first — a full technical
walkthrough of how the pipeline works and why, written for an engineer picking this up cold.
Also shipped as [`summary.docx`](summary.docx) for sharing; regenerate it after editing the
markdown with:

```bash
pip install python-docx
python tools/md2docx.py docs/SUMMARY.md summary.docx \
    "auto-wall-detect-matting" "Project summary — a technical walkthrough"
```

See [`pipeline.html`](pipeline.html) for the flow diagram, [`docs/`](docs/) for the design
record.

---

## 1. The best pipeline — `main_gd.ipynb`

Per sample folder `users_gd/<SAMPLE>/`, starting from only `original.jpg`:

```
                      ┌─ (0) MoGe geometry ─────────────────────────────┐  (async, on the
original.jpg ─────────┤                                                 │   ORIGINAL image)
                      └─ (1) RAM++ tags ─→ GroundingDINO boxes ─→ BiRefNet matte
                                                    │
                                          objects ∪ rug = remove_mask
                                                    │
                             (2) Imagen 3 EDIT_MODE_INPAINT_REMOVAL
                                                    │
                                            empty_imagen.png
                                                    │
                             (3) image2scene /run  auto_detect
                                                    │
                                    amodal wall_masks / floor_masks
                                                    │
                             (4) visible = amodal − objects (floor also − rug)
```

**Stage detail**

0. **MoGe** (monocular geometry, image2scene `/run_moge`) is kicked off **first** on the
   *original* image and collected at the **last** step, so it overlaps the whole rest of the
   run. Result lands in `moge/`.
1. **RAM++** (`ram_plus_swin_large_14m.pth`) auto-tags the image. Tags are filtered
   (`RAM_EXCLUDE` / `RAM_EXCLUDE_KW`) so structural words — wall, floor, ceiling, room,
   window, curtain, flooring — are never handed to the detector, then joined
   `". "`-separated into the **GroundingDINO** open-vocab prompt. DINO returns boxes +
   labels; class-agnostic NMS (`NMS_IOU`) drops duplicates. **BiRefNet** mattes the
   foreground inside each padded box (`PAD=0.05`) plus one full-image fallback pass; the
   union ≥ `thresh` is the object mask. Boxes whose label hits `RUG_LABELS` produce a
   separate **rug** mask.
2. **Imagen 3** (`imagen-3.0-capability-001`, `EDIT_MODE_INPAINT_REMOVAL`) removes
   `objects ∪ rug` with an **empty prompt** — a descriptive prompt makes Imagen *generate*
   furniture instead of erasing it. `mask_dilation=0.03`. Inpaint only rewrites masked
   pixels, so the empty room is **pixel-aligned to the original — no homography needed**.
3. **image2scene API** (`POST /run` → poll `/jobs/{id}` → download result zip) re-runs
   `auto_detect` on the empty room. Because the furniture is gone, its `wall_masks` /
   `floor_masks` are **amodal**. The whole raw zip is kept under `from-api/`.
4. `visible = amodal − objects`, and floor additionally `− rug`. Walls and floors are kept
   **individually** (one mask per surface instance), not just as a union.

No `auto_detect.json` **input** is required: objects + rug come from RAM++/GroundingDINO,
wall/floor from the API re-run.

### Outputs (per sample folder)

| File | Meaning |
| --- | --- |
| `object_mask_birefnet.png` | foreground objects (removed) |
| `rug_mask_birefnet.png` | rug / floor-covering (removed) |
| `inpaint_mask.png` | mask sent to Imagen (`objects ∪ rug`) |
| `empty_imagen.png` | Imagen empty room |
| `wall_mask_amodal.png` / `floor_mask_amodal.png` | full surfaces incl. pixels hidden behind furniture |
| `wall_mask_visible.png` / `floor_mask_visible.png` | surfaces minus objects (floor also minus rug); union of all instances |
| `masked_images/wall_mask_visible_NN.png` | each **individual** wall as its own visible mask |
| `masked_images/floor_mask_visible_NN.png` | each **individual** floor as its own visible mask |
| `overlay.png` | input + each wall in its own colour + floor (green) |
| `from-api/` | raw image2scene result (incl. its own `auto_detect.json`) |
| `moge/` | raw MoGe geometry (run on the original image) |
| `time.txt` | per-stage seconds + `total` |

### Tuning knobs (top of the config cell / `run.py`)

- `RAM_EXCLUDE` / `RAM_EXCLUDE_KW` — RAM tags never passed to DINO (structural + scene words).
- `RAM_PLUG_KW` — plug/socket is only hunted when RAM++ actually tags one (else DINO
  hallucinates outlets on blank wall).
- `OBJECT_PROMPT` — fallback DINO prompt when RAM returns nothing usable.
- `DINO_BOX_THRESH` (0.25) / `DINO_TEXT_THRESH` (0.20) / `NMS_IOU` (0.5).
- `RUG_LABELS` — DINO labels treated as rug/floor-covering.
- `thresh` (0.5) — BiRefNet object binarization.
- `EMPTY_INPAINT_PROMPT` — keep it **empty**.

The notebook's lower half is the tuning surface: all DINO boxes + labels, the per-object
BiRefNet matte grid (one tile per box, so you can see exactly what BiRefNet returns inside
each detection), and the 3×3 amodal-vs-visible comparison.

### BiRefNet checkpoint used here

`main_gd.ipynb` loads the Hypersim fine-tune `hq-mat/BiRefNet/ckpts/hypersim/epoch_10_new.pth`
(the stock `zhengpeng7/birefnet` load is left in the cell, commented out).

Loading needs a workaround: the model was trained at `batch_size=1`, where BiRefNet swaps
every `BatchNorm` for `nn.Identity` (see the `config.batch_size > 1` gates). The notebook
forces `batch_size = 1` on `models.modules.aspp`, `decoder_blocks` and a subclassed
`Config` **before** building the net, otherwise `load_state_dict` raises
`Missing key(s) ... bn ...`.

> ⚠️ Checkpoint caveat: `epoch_10_new.pth` works well **here**, where BiRefNet only mattes
> inside a GroundingDINO box. It is *not* the right checkpoint for the structure-first
> pipeline — see [ADR 0008](docs/adr/0008-structure-first-normal-clustering.md), where
> `epoch_10_new` scored a structure fraction of 0.34 on an empty hallway (66 % of a bare
> room called "object") and `epoch_10.pth` is used instead.

---

## 2. All pipeline variants

| Entry point | Detector | Empty room | Surfaces | Output dir | Verdict |
| --- | --- | --- | --- | --- | --- |
| **`main_gd.ipynb`** | RAM++ → GroundingDINO | Imagen 3 | image2scene API | `users_gd/` | ⭐ **best — use this** |
| `run.py` | RAM++ → GroundingDINO | Imagen 3 | image2scene API | `output-test/` | CLI form of the same flow, stock BiRefNet, reads `downsample.jpg` |
| `new_gd.ipynb` | RAM++ → GroundingDINO | Imagen 3 | image2scene API | `users_new_gd/` | same flow, BiRefNet `epoch_15.pth` — earlier checkpoint sweep |
| `main_yolo.ipynb` | YOLO-World (fixed classes) | Imagen 3 | image2scene API | `users_yolo/` | detector ablation; fixed class list, no RAM++ |
| `main_yolo_nano.ipynb` | YOLO-World | **Gemini 2.5 Flash Image** ("nano banana") | image2scene API | `users_yolo_nano/` | prompt-based edit, **no mask API** → pixels outside the target can move, so alignment is not guaranteed |
| `main_normal.ipynb` | BiRefNet-invert | *none* | Metric3D-v2 normal + depth clustering | `image2scene/<s>/normal_out/` | structure-first; fast, no API, **visible-only** ([ADR 0008](docs/adr/0008-structure-first-normal-clustering.md)) |
| `main_sd.ipynb` / `main_sd.py` | — | — | **SDMatte** refines a wall mask → soft alpha | `users_sd/` | matting stage |
| `main_diff.ipynb` / `main_diff.py` | — | — | **DiffMatte** refines a wall mask → soft alpha | `users_diff/` | matting stage, 45× smaller net |
| `run_api.py` | — | — | image2scene API on the **raw** original | `image2scene/` | baseline / trimap source, no object removal |
| `main.ipynb` | — | — | — | — | viewer for `auto_detect.json` boxes & points |
| `trainer.ipynb`, `trainer_v2.ipynb` | — | — | — | `hq-mat/BiRefNet/ckpts/` | BiRefNet Hypersim fine-tune |

### Why Imagen over nano banana

Imagen's `EDIT_MODE_INPAINT_REMOVAL` takes an explicit mask and **only rewrites masked
pixels**, so the empty room stays pixel-registered with the input and the API's mask
coordinates transfer back directly. Gemini nano banana edits from a prompt with no mask
channel, so unmasked pixels can drift — that variant is kept as a comparison, not as the
production path.

### Why RAM++ over a fixed YOLO class list

GroundingDINO is open-vocab but **closed-set per run**: it only finds what you name. A
fixed list ([ADR 0007](docs/adr/0007-dino-prompts-acceptance.md)) misses whatever the room
happens to contain. RAM++ names the room's actual contents per image, so the prompt adapts;
the exclusion lists then subtract the structural words RAM++ also emits.

---

## 3. Matting stage — SDMatte vs DiffMatte

The API/pipeline masks are hard binary masks; edges are stair-stepped and miss soft
boundaries. `main_sd.ipynb` and `main_diff.ipynb` refine a wall mask into a soft alpha.

Both notebooks import **`generate_trimap.wall_trimap()`** — one shared trimap generator, so
the two networks receive byte-identical input and any difference isolates the network.
The trimap source is `image2scene/<sample>/wall_mask.png` (written by `run_api.py` straight
from `original.jpg`): the *actually visible* wall, unlike `users_gd/<s>/wall_mask_visible.png`
which the API derived from an object-removed image.

Trimap: `1` = definite wall, `0` = definite background, `0.5` = unknown band dilated
`TRIMAP_SIZE` outward from every boundary.

| | SDMatte | DiffMatte |
| --- | --- | --- |
| prompt types | trimap / mask / box / points | **trimap only** |
| backbone | Stable Diffusion U-Net (~1.3 B) | plain ViT-S (29 M) + tiny diffusion decoder |
| iterations | 1 step, denoise in latent space | N steps, denoise the **alpha** directly |
| VRAM @1024² | ~5 GB (fp16) | ~1.6 GB (fp32) |

**Head-to-head** (`compare_sd_diff/`, 50 samples, shared trimap, `alpha_thresh=0.5`):

| metric | value |
| --- | --- |
| mean IoU (SDMatte vs DiffMatte) | **0.981** |
| pixel agreement | 0.995 |
| MAE | 0.0044 |
| unknown-band fraction | 1.98 % |
| **band** agreement | 0.740 |
| **band** MAE | 0.250 |

Read: the two agree almost everywhere (0.98 IoU) — they only disagree **inside the unknown
band**, which is the entire point of matting. DiffMatte reaches that with a 45× smaller
backbone and ~⅓ the VRAM. Per-sample numbers in `compare_sd_diff/metrics.csv`,
`metrics.json`, side-by-side panels in `compare_sd_diff/user_N.png`, summary in
`_summary.png`.

Notes:
- SDMatte is built **config-only** (`load_weight=False` → diffusers `from_config` on the HF
  snapshot's `unet/vae/text_encoder/scheduler/tokenizer`), then `SDMatte.pth` is loaded on
  top. `weights_only=False` + `mmap=True` are required — the checkpoint is a
  `DetectionCheckpointer` dict pickling an omegaconf `ListConfig`.
- DiffMatte needs no detectron2: a `d2shim` supplies the few symbols it imports.
- With `PROMPT_MODE="trimap"` the surface already comes from image2scene, so RAM++ and
  GroundingDINO are skipped — ~4 GB VRAM and ~5 s/sample saved.
- SDMatte's `points` prompt mode **cannot encode negatives**, and signed blobs collapse the
  alpha — trimap is the only mode that works reliably.
- Samples with EXIF `Orientation != 1` (e.g. `user_3`, `user_13`) are normalized
  **in memory** via `ImageOps.exif_transpose`, never on disk.

---

## 4. Structure-first variant — `main_normal.ipynb`

An alternative that drops RAM++, GroundingDINO, Imagen, SAM **and** the image2scene API
([ADR 0008](docs/adr/0008-structure-first-normal-clustering.md)). Motivation: Imagen is slow
and ghosts hallucinated content into the emptied room, which the segmenter then inherits.

1. **BiRefNet** (fine-tuned `epoch_10.pth`) segments *movable objects* → **invert** →
   structure mask = wall + floor + ceiling shell, **visible pixels only**.
   (Hypersim's BG class set means door/window/mirror/ceiling are learned as *structure*, not
   foreground — which is exactly what this step wants.)
2. **Metric3D-v2** — one feed-forward net giving per-pixel **metric depth + surface normal**.
   Chosen over Marigold: no diffusion latency, and depth+normal from a single pass.
3. Backproject structure pixels to a point cloud, then split:
   - **floor / ceiling / wall** by `angle(normal, up)` with a **30° cone**;
   - **wall instances** by **HDBSCAN** on `normal ⊕ W_OFFSET·(z-scored plane offset d)`,
     where `d = point · n`. The offset is what splits **same-faced jogged sub-walls** that
     share a normal — neither normal angle nor 2-D connectivity can. This is why depth is
     mandatory, not optional.
   - **2-D connected components** for spatially disjoint same-plane pieces.

HDBSCAN is fit on a 30 k-pixel subsample and all wall pixels assigned by nearest cluster
centroid — `approximate_predict` on ~1 M px is too slow.

**Verified** on `user_2`: `epoch_10.pth` structure fraction **0.995** (correctly leaves an
empty hallway as structure); Metric3D-v2 + HDBSCAN found 4 clusters and cleanly split
left wall / right wall / far wall / floor / ceiling → 8 instance masks after 2-D CC.

**Known limit:** visible-only. Inversion cannot recover surface hidden behind furniture —
every object leaves a hole, so `*_mask_amodal` is not reproduced. Inversion also has no
safety net: a BiRefNet miss leaves an object as fake surface, a false positive punches a
hole. Quality rides entirely on the object model.

Knobs: `MIN_CLUSTER_FRAC`, `W_OFFSET`, `NORMAL_CONE_DEG`, `UP_CAM` sign.

Env notes: `mmengine` + `mmcv` shim (`Config` → mmengine; mmcv 1.x will not build on
torch 2.12 / cu130), `hdbscan` against sklearn 1.7.2.

---

## 5. Training — BiRefNet on Hypersim

**Goal:** teach BiRefNet to segment *everything that is NOT room structure* (all furniture
and movable objects), so the pipeline can remove them.

**Data:** [Hypersim](https://github.com/apple/ml-hypersim) (Apple) — photorealistic
*synthetic* indoor scenes, **CC BY-SA 3.0 (commercial OK)**, with pixel-perfect rendered
semantic labels. Foreground = non-structure.

| Notebook | Method |
| --- | --- |
| `trainer.ipynb` | Guided walkthrough of the first fine-tune: download (tonemap + semantic), prep, train. Heavy steps flag-guarded and OFF by default so *Run All* stays fast. |
| `trainer_v2.ipynb` | Proper methodology: **scene-level** 70/15/15 train/val/test split (frames of one room never cross splits); Phase A trains on TRAIN with early stopping on val mean-IoU (max 50 epochs, best ckpt kept); Phase B folds VAL back in for a few more epochs; TEST is evaluated **once**. Checkpoints + `history.json` written every epoch to `ckpts/hypersim_es/`, so progress survives interruption. |

Hardware target: RTX 5070, 12 GB — batch size 1 at 640², which is why the `batch_size=1`
BatchNorm→Identity swap matters at load time (see §1).

Checkpoints produced: `hq-mat/BiRefNet/ckpts/hypersim/epoch_{5,10,15,20,25,30}.pth`,
`ckpts/hypersim_es/{best,final}.pth`. Prep docs: `hq-mat/BiRefNet/hypersim_prep/README.md`.

---

## 6. Repo layout

```
main_gd.ipynb           ⭐ recommended pipeline (RAM++/DINO/BiRefNet/Imagen/image2scene/MoGe)
new_gd.ipynb            same flow, epoch_15 checkpoint
main_yolo.ipynb         YOLO-World detector ablation
main_yolo_nano.ipynb    YOLO-World + Gemini nano banana empty room
main_normal.ipynb       structure-first (BiRefNet-invert + Metric3D)
main_sd.ipynb  main_sd.py     SDMatte refinement
main_diff.ipynb main_diff.py  DiffMatte refinement
main.ipynb              auto_detect.json viewer
trainer.ipynb  trainer_v2.ipynb   BiRefNet Hypersim fine-tune
run.py                  CLI form of the empty-room pipeline
run_api.py              raw image2scene baseline -> image2scene/<user_N>/
generate_trimap.py      shared trimap generator (CLI + wall_trimap() import)
trimap_generator/       vendored trimap primitives (erosion/dilation)
pipeline.html           flow diagram
docs/                   design record: pipeline-design.md, glossary.md, adr/
compare_sd_diff/        SDMatte vs DiffMatte metrics + panels
hq-mat/                 BiRefNet / SDMatte / DiffMatte source repos   (not in git)
weights/                RAM++, SDMatte, DiffMatte, CLIP checkpoints   (not in git)
users_gd/ users_*/      per-sample outputs, one dir per pipeline      (not in git)
image2scene/            run_api.py baseline outputs                   (not in git)
result/                 overlay renders                               (not in git)
```

Everything marked *not in git* is model weights or generated output; see `.gitignore`.

---

## 7. Setup

```bash
conda create -n birefnet python=3.10 && conda activate birefnet
pip install -r requirements.txt
# then install torch/torchvision for your CUDA: https://pytorch.org
```

Assets that are **not** pip-installable and not in this repo:

| Asset | Where it goes |
| --- | --- |
| **BiRefNet** source repo | `hq-mat/BiRefNet` (added to `sys.path` at runtime; stock weights auto-pull from `zhengpeng7/birefnet` on the HF Hub) |
| BiRefNet Hypersim fine-tunes | `hq-mat/BiRefNet/ckpts/hypersim/epoch_*.pth` (produced by `trainer*.ipynb`) |
| **RAM++** checkpoint | `weights/ram_plus_swin_large_14m.pth` |
| **SDMatte** | `weights/SDMatte/` (HF snapshot + `SDMatte.pth`), source in `hq-mat/SDMatte` |
| **DiffMatte** | `weights/DiffMatte/DiffMatte_ViTS_1024.pth`, source in `hq-mat/DiffMatte` |
| CLIP ViT-B/32 | `weights/clip/ViT-B-32.pt` |

### Vertex / Imagen auth

Imagen runs on Vertex AI via **ADC** (gcloud Application Default Credentials), *not* an API
key. Put your project in a `.env` at the repo root:

```
GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
GOOGLE_CLOUD_LOCATION=us-central1
```

then

```bash
gcloud auth application-default login
```

`load_models()` drops a stale `GOOGLE_APPLICATION_CREDENTIALS` path if the file does not
exist, so a leftover service-account env var will not break ADC.

### image2scene API

`API_BASE = https://image2scene-dev.wedolabs.net` — `POST /run` (auto_detect) and
`POST /run_moge` (geometry), both async: poll `GET /jobs/{id}` until `completed`, then
download `result_zip_url`.

---

## 8. Usage

**Notebook (recommended).** Open `main_gd.ipynb`, set `SAMPLE`, run all:

```python
SAMPLE = "user_8"                       # any folder under users_gd/ with original.jpg
res = run_pipeline(OUTPUT_DIR / SAMPLE, thresh=0.5)
```

The batch cell at the bottom loops every sample folder (commented out by default —
**each iteration bills**).

**CLI:**

```bash
python run.py --sample living_23                    # one sample
python run.py --sample Bedroom --sample living_23   # several
python run.py --all                                 # every folder with downsample.jpg
python run_api.py                                   # raw API baseline, users_gd -> image2scene/
python generate_trimap.py --input <masks> --output <trimaps>
```

> 💸 Each empty-room run **bills 1 Imagen call + 1 image2scene job + 1 MoGe job per sample.**

---

## 9. Design record (`docs/`)

- [`docs/pipeline-design.md`](docs/pipeline-design.md) — assembled end-to-end design.
- [`docs/glossary.md`](docs/glossary.md) — working vocabulary (surface, occluder,
  `objects_all`, structure mask, plane offset `d`, amodal vs visible, …).
- [`docs/adr/`](docs/adr/) — the decision trail.

| ADR | Decision |
| --- | --- |
| [0001](docs/adr/0001-detector-scope.md) | Detector scope: DINO, not `auto_detect.json` object bboxes, drives exclusions *(amended by 0002)* |
| [0002](docs/adr/0002-completeness-hybrid.md) | Exclusion mask = **union of two complementary detectors**: BiRefNet (any 3-D object, class-agnostic) ∪ GroundingDINO (flat wall-anchored items BiRefNet cannot see) |
| [0003](docs/adr/0003-sam-roles.md) | SAM's two roles: box-prompt → occluder cut-masks; point-prompt ∩ bbox → surface masks |
| [0004](docs/adr/0004-infra.md) | HF `transformers` for DINO + SAM-huge in the existing `birefnet` env; no custom CUDA-op build |
| [0005](docs/adr/0005-subtraction-rule.md) | Subtract **all** objects from **both** surfaces. Anchoring (wall- vs floor-anchored) is QA metadata only |
| [0006](docs/adr/0006-outputs-rug-scope.md) | Non-destructive PNG outputs; rug stays a deliverable **and** is subtracted from the floor |
| [0007](docs/adr/0007-dino-prompts-acceptance.md) | DINO prompt list + acceptance by visual overlay review (not IoU against an unverified baseline) |
| [0008](docs/adr/0008-structure-first-normal-clustering.md) | Structure-first: BiRefNet-invert + Metric3D normal/depth plane clustering (`main_normal`) |

**Key ideas worth carrying forward**

- **"Without any object" = any 3-D foreground object ∪ any named flat occluder.** BiRefNet
  is a *salient 3-D foreground* model: it structurally cannot see window / mirror / door /
  poster, which sit flush on the wall plane. That gap is exactly what the open-vocab
  detector fills. Residual misses are flat, unnamed wall items.
- **Anchoring never restricts subtraction.** A floor-anchored object occludes the wall (sofa
  back, tall plant, bookshelf). An object pixel is never legitimately part of *any* surface,
  so every object is cut from every surface. (ADR 0005)
- **Amodal vs visible is a real fork, not a detail.** Amodal needs a generative fill
  (Imagen) — slow, billed, can ghost. Visible needs only inversion — fast, local, but every
  object leaves a hole. (ADR 0008)
- **Depth is mandatory for wall instances.** Two jogged sub-walls facing the same direction
  have identical normals and can be 2-D connected; only the plane offset `d = point · n`
  separates them.
- **The 30° normal cone** separates floor / ceiling / wall (≈90° apart) and clearly
  different-faced walls; anything below 30° falls to `d`.
- **Outputs are non-destructive** — new PNGs per sample dir; no input JSON is ever modified.

**Open risks**

- **VRAM.** BiRefNet + DINO + SAM/Metric3D co-resident on a 12 GB RTX 5070 may need
  sequential load or CPU offload.
- **DINO false positives** — mirror ↔ window, framed art ↔ window; confidence threshold
  needs per-deployment tuning.
- **Synth → real gap.** The Hypersim fine-tune is trained on synthetic renders; the
  `epoch_10_new` regression (structure fraction 0.34 on a real empty hallway) is the
  measurement of that gap.

---

## References

- BiRefNet — `zhengpeng7/birefnet`
- RAM++ (Recognize Anything Plus) — `ram_plus_swin_large_14m`
- GroundingDINO — `IDEA-Research/grounding-dino-base`
- SDMatte, DiffMatte — [Diffusion for Natural Image Matting](https://arxiv.org/abs/2312.05915) (ECCV 2024), [code](https://github.com/YihanHu-2022/DiffMatte)
- Metric3D-v2 — [arXiv 2404.15506](https://arxiv.org/abs/2404.15506) (zero-shot metric depth + normal, one model)
- Hypersim — [apple/ml-hypersim](https://github.com/apple/ml-hypersim)
- Imagen 3 editing — Vertex AI `imagen-3.0-capability-001`
