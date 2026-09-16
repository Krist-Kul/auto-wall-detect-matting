# Project summary — a technical walkthrough

Audience: an AI engineer who has never seen this repo and needs to (a) understand what it
does and why it is built this way, and (b) run it.

Read this top to bottom once. Then `main_gd.ipynb` will make sense cell by cell.

**Related reading:** [`../README.md`](../README.md) (quick reference + variant table),
[`pipeline-design.md`](pipeline-design.md) (original design), [`glossary.md`](glossary.md)
(vocabulary), [`adr/`](adr/) (why each decision was made).

---

## 1. The problem

**Input:** one ordinary photo of a room.
**Output:** a binary mask per architectural surface — each wall, the floor — containing
**only** the surface, with every object removed.

Downstream this feeds virtual redecoration: repaint a wall, swap the flooring. That use
case dictates two hard requirements:

1. **Nothing but surface.** A single sofa pixel left in the wall mask becomes a sofa
   painted the new colour. Recall on objects matters more than precision.
2. **Surface hidden behind furniture is still wall.** If you repaint a wall and the sofa
   is later moved, the strip behind it must already be painted. This is the **amodal**
   requirement, and it is the hardest part of the problem.

### Why this is not just semantic segmentation

An off-the-shelf `wall` / `floor` segmenter (ADE20K-style) fails both requirements:

- It gives you *visible* wall only. Pixels behind the sofa are labelled `sofa`. No amodal.
- It gives you **one** wall class, not wall instances. Redecoration needs "paint *that*
  wall", so the left wall and the far wall must be separate masks.
- Its boundaries are hard and blocky; matting-quality edges are a separate problem again.

Every design choice in this repo is an answer to one of those three gaps.

---

## 2. The central idea

> **You cannot segment what is hidden. So remove the furniture first, then segment.**

Concretely: detect every object, generatively **inpaint them out** of the photo to get an
empty room, and run surface segmentation on *that*. The segmenter now sees an unobstructed
wall, so its output is amodal by construction — no amodal-specific model, no hallucinating
mask completion network.

The trick that makes this sound is **pixel registration**. Imagen's
`EDIT_MODE_INPAINT_REMOVAL` takes an explicit mask and rewrites **only the masked pixels**.
Everything else is byte-identical to the input. So a mask computed on the empty room can be
applied to the original photo directly — **no homography, no re-alignment, no registration
error**. This is the single most important property of the pipeline, and the reason Imagen
beats a prompt-only image editor here (see §7.2).

Then, trivially:

```
amodal_surface  = segmentation of the empty room
visible_surface = amodal_surface − objects        (floor also − rug)
```

You get both variants from one run. The subtraction is free; the amodal version is the
expensive part.

---

## 3. Architecture

```
                original.jpg
                     │
      ┌──────────────┼──────────────────────────────────────┐
      │              │                                      │
   (0) MoGe      (1) RAM++ ──tags──▶ GroundingDINO ──boxes──▶ BiRefNet
   geometry          │                    │                     │
   (async,           │              rug-labelled boxes      object matte
   collected         │                    │                     │
   last)             │                 rug mask ─────┬──────────┘
      │              │                               │
      │                                    remove = objects ∪ rug
      │                                              │
      │                       (2) Imagen 3 INPAINT_REMOVAL (empty prompt)
      │                                              │
      │                                       empty_imagen.png
      │                                    (pixel-aligned to input)
      │                                              │
      │                       (3) image2scene POST /run → auto_detect
      │                                              │
      │                              wall_masks[] / floor_masks[]  ← AMODAL
      │                                              │
      │                       (4) visible = amodal − objects (− rug for floor)
      │                                              │
      └──────────────────────▶  outputs + per-stage timings  ◀────┘
```

Five models, each doing exactly one job it is uniquely good at. The design is a
**pipeline of specialists**, not one big model — because no single model does all of this,
and because each stage is independently inspectable when something goes wrong.

### Why each model is in the pipeline

| Model | Job | Why not something else |
| --- | --- | --- |
| **RAM++** | Names what is actually in *this* room | GroundingDINO is open-vocab but **closed-set per call** — it only finds classes you name. A hardcoded list misses whatever this particular room contains. RAM++ generates the class list per image, so the detector's vocabulary adapts. |
| **GroundingDINO** | Turns those names into boxes | RAM++ is a tagger — labels, no localization. Something must convert tag → box. |
| **BiRefNet** | Pixel-accurate matte inside each box | DINO gives rectangles. A box around a chair includes wall. You need the actual silhouette, and the silhouette must be *tight* or you erase wall you wanted to keep. |
| **Imagen 3** | Fills the holes → empty room | This is what makes the result amodal. Also the only billed, non-local, non-deterministic step. |
| **image2scene API** | Wall/floor **instance** masks | An internal service that returns per-wall masks, not one merged `wall` class. Instance separation is a requirement (§1). |
| **MoGe** | Monocular geometry of the original | Not used by the mask math — collected for downstream 3-D consumers. Async so it costs ~0 wall-clock. |

### Why two detectors, not one

This is the most important non-obvious decision in the repo
([ADR 0002](adr/0002-completeness-hybrid.md)).

**BiRefNet is a salient 3-D foreground segmenter.** It excels at sofas, tables, plants,
boxes — anything that visually pops off the background. It is structurally **blind to flat,
wall-anchored items**: a window, a mirror, a poster, a wall-mounted TV, a painting. Those
sit *in the wall plane*, have no foreground/background depth cue, and are not salient. They
will silently stay in your wall mask.

**GroundingDINO covers exactly that gap** — you name the flat classes and it finds them.

So the exclusion mask is defined as a union of two complementary detectors:

```
objects_all = BiRefNet(any 3-D foreground, class-agnostic)
            ∪ GroundingDINO(named flat occluders)
```

and "a surface without any object" means *any 3-D foreground object* ∪ *any named flat
occluder*. The honest residual failure mode is: **flat, unnamed wall items**. If you find a
new one, you add it to the prompt vocabulary.

The current implementation folds these together: RAM++ names the flat items, DINO boxes
them, and BiRefNet mattes *inside each box* — a box gives BiRefNet enough local context to
segment a flat object it would miss globally. Plus one full-image BiRefNet pass as a
fallback, unioned in, for anything DINO missed.

### Why anchoring never restricts subtraction

Objects are describable as wall-anchored (mirror) or floor-anchored (sofa). It is tempting
to subtract only wall-anchored objects from the wall. **Don't**
([ADR 0005](adr/0005-subtraction-rule.md)) — a sofa back, a tall plant, or a bookshelf is
floor-anchored *and* occludes the wall.

> An object pixel is never legitimately part of *any* surface.

So every object is subtracted from every surface. Anchoring labels survive as QA metadata
only. This is simpler *and* more correct — a rare combination worth noticing.

---

## 4. Code walkthrough — `main_gd.ipynb`

The notebook is the reference implementation. `run.py` is the same flow as a CLI.
Cells in order:

### Config cell

Everything tunable is at the top. The ones that matter:

```python
OUTPUT_DIR   = ROOT / "users_gd"     # per-sample folders live here
RAM_EXCLUDE      = {...}             # exact tags never given to DINO
RAM_EXCLUDE_KW   = ("wall","floor","ceiling","room")   # substring blocklist
RAM_PLUG_KW      = ("plug","socket","outlet",...)      # conditional vocabulary
DINO_BOX_THRESH  = 0.25              # detection confidence
DINO_TEXT_THRESH = 0.20              # text-grounding confidence
NMS_IOU          = 0.5               # class-agnostic NMS
RUG_LABELS       = ("rug","carpet","mat","runner")
IMAGEN_EDIT_MODEL     = "imagen-3.0-capability-001"
EMPTY_INPAINT_PROMPT  = ""           # MUST stay empty — see below
```

**`RAM_EXCLUDE` is load-bearing.** RAM++ tags the *scene*, so it emits `wall`, `floor`,
`ceiling`, `living room`, `hardwood`, `flooring`, `curtain`. Feed those to DINO and it
boxes the wall — which you then erase and inpaint away. You would delete the very thing you
are trying to segment. The exact set plus the substring keywords exist to prevent exactly
that.

**`RAM_PLUG_KW` is a nice pattern worth stealing.** Plugs and sockets are small and DINO
hallucinates them all over blank wall. So `plug`/`socket` enter the prompt **only when
RAM++ actually tagged one**. Conditional vocabulary: pay the false-positive cost only when
there is evidence the class is present.

**`EMPTY_INPAINT_PROMPT = ""` is not laziness.** Imagen treats the prompt as *what to
generate in the masked region*. A descriptive prompt ("empty room, bare floor") makes it
**generate furniture** — it reads the prompt as a scene description, not an instruction.
The empty string is what triggers clean content-aware removal. This is a real behavioural
gotcha, not an aesthetic choice.

### `load_models()`

Loads RAM++, GroundingDINO, BiRefNet and the Vertex client once into module globals.

Auth detail worth knowing: Imagen runs on Vertex via **ADC** (gcloud Application Default
Credentials), *not* an API key. The function defensively pops a stale
`GOOGLE_APPLICATION_CREDENTIALS` if it points at a nonexistent file, so a leftover
service-account env var cannot break ADC.

**The BiRefNet checkpoint load needs a workaround.** The fine-tune was trained at
`batch_size=1`, and BiRefNet's source swaps every `BatchNorm` for `nn.Identity` when
`config.batch_size <= 1` (see its `config.batch_size > 1` gates). So the *architecture
itself* depends on a global config value. If the kernel already imported `config.py` while
`batch_size` was > 1, the built model has BN layers the checkpoint does not, and
`load_state_dict` raises `Missing key(s) ... bn ...`.

The fix, executed before constructing the net:

```python
import models.birefnet as _bnmod
from models.modules import aspp as _aspp, decoder_blocks as _decblk
_aspp.config.batch_size = 1
_decblk.config.batch_size = 1
class _BS1Config(_bnmod.Config):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.batch_size = 1
_bnmod.Config = _BS1Config
_bi = _bnmod.BiRefNet(bb_pretrained=False)
_bi.load_state_dict(torch.load(CKPT, weights_only=True))
```

If you ever see that `Missing key(s)` error, this is why.

### Stage 1 — detection and matting

```python
ram_autotag(image)   -> (dino_prompt, tags)     # RAM++ → filtered, ". "-joined
detect_objects(...)  -> (boxes, labels)         # DINO + class-agnostic NMS
rug_boxes_from_labels(boxes, labels)            # boxes whose label ∈ RUG_LABELS
birefnet_objects(image, boxes, pad=0.05)        # union of per-box mattes + full-image pass
birefnet_rug(image, rug_bboxes, thresh)         # matte inside rug boxes only
```

`_birefnet_crop(image, box)` is the primitive: crop → resize 1024² → BiRefNet → sigmoid →
resize back → paste onto a full-image canvas. Per-box cropping is what gives small objects
enough resolution to be matted well; a single full-image pass would smear them.

`pad=0.05` grows each box 5 %: DINO boxes clip object edges (chair legs, plant fronds), and
matting a clipped crop produces a clipped matte, which leaves a fringe of object in the
final surface mask.

NMS is **class-agnostic** — RAM++ produces overlapping synonyms (`sofa`, `couch`,
`furniture`) that all box the same object. Deduplicating by IoU regardless of label is the
right call here; you want the pixels once, not the taxonomy.

### Stage 2 — Imagen

```python
imagen_empty_room(sample_dir, base_img, remove_mask, out_path)
```

Writes `inpaint_mask.png`, sends a `RawReferenceImage` + `MaskReferenceImage`
(`MASK_MODE_USER_PROVIDED`, `mask_dilation=0.03`) to `edit_image` with
`EDIT_MODE_INPAINT_REMOVAL`, saves the result, and resizes back to the original size if
Imagen returned a different one — **required**, because everything downstream assumes the
empty room and the original share a pixel grid.

`mask_dilation=0.03` grows the mask ~3 % so the inpaint covers object edges and contact
shadows. Too small leaves object halos that the segmenter reads as surface texture.

### Stage 3 — image2scene API

```python
api_autodetect(empty_path, out_dir)   -> (meta, api_downsample_img, job_id)
moge_start(image_path)                -> job_id      # fire and forget
moge_fetch(job_id, out_dir)           -> out_dir     # collect later
```

Both are async job APIs: `POST /run` (or `/run_moge`) → poll `GET /jobs/{id}` until
`completed` → download `result_zip_url` → extract. The **entire** zip is kept under
`from-api/`, including the service's own `auto_detect.json`, so nothing is lost if you
later need a field the pipeline ignores.

MoGe is started at step 0 and fetched at the very end. It is pure latency hiding — the job
runs server-side while BiRefNet, Imagen and auto_detect run locally, so its cost is roughly
zero wall-clock instead of ~30 s serial.

### Stage 4 — decode and subtract

```python
_decode_masks_list(b64_list, dst_size)  # base64 → list of bool arrays, NEAREST-resized
```

The API returns masks base64-encoded at *its* downsample resolution. They are decoded
**individually** (not merged) and NEAREST-resized to the original resolution — NEAREST
because bilinear on a binary mask invents grey edge pixels that then threshold
inconsistently.

Individual decoding is what preserves **wall instances**. The union is derived from the
list, never the other way round:

```python
walls_full  = _decode_masks_list(meta["wall_masks"], (w, h))   # per-wall
wall_full   = np.logical_or.reduce(walls_full)                 # union

walls_vis   = [wm & ~objects              for wm in walls_full]
floors_vis  = [fm & ~objects & ~rug       for fm in floors_full]
```

Then everything is written out, plus a colour overlay (each wall its own colour from
`WALL_COLORS`, floor green) and `time.txt` with per-stage seconds.

### What `run_pipeline()` returns

A dict with every intermediate — `img`, `empty`, `objects`, `rug`, `obj_boxes`,
`obj_labels`, `obj_tags`, `obj_prompt`, `wall_full`, `floor_full`, `walls_visible`,
`floors_visible`, `timings`, `moge_dir`. The visualization cells below it consume this
dict. Keeping intermediates in memory is deliberate: tuning means looking at stage
outputs, not at the final mask.

---

## 5. Running it

### Environment

```bash
conda create -n birefnet python=3.10 && conda activate birefnet
pip install -r requirements.txt
# install torch/torchvision for YOUR cuda: https://pytorch.org
gcloud auth application-default login
```

`.env` at the repo root:

```
GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
GOOGLE_CLOUD_LOCATION=us-central1
```

Assets that are **not** in git (too large) and must be fetched separately:

| Asset | Path |
| --- | --- |
| BiRefNet source repo | `hq-mat/BiRefNet` (added to `sys.path` at runtime) |
| BiRefNet Hypersim fine-tunes | `hq-mat/BiRefNet/ckpts/hypersim/epoch_*.pth` |
| RAM++ checkpoint | `weights/ram_plus_swin_large_14m.pth` |
| SDMatte | `weights/SDMatte/` + source in `hq-mat/SDMatte` |
| DiffMatte | `weights/DiffMatte/DiffMatte_ViTS_1024.pth` + source in `hq-mat/DiffMatte` |
| CLIP ViT-B/32 | `weights/clip/ViT-B-32.pt` |

> Note: `run.py` looks for RAM++ at the **repo root** (`ROOT/"ram_plus_swin_large_14m.pth"`)
> while the notebooks look in `weights/`. Symlink or copy if you use the CLI.

### Data layout

One folder per sample, named `user_N`, containing `original.jpg`. That is the only required
input. The pipeline writes all outputs back into the same folder. Each pipeline variant has
its own root so variants never overwrite each other:

```
users_gd/user_8/original.jpg          ← main_gd.ipynb reads this
users_new_gd/                         ← new_gd.ipynb
users_yolo/, users_yolo_nano/         ← YOLO variants
image2scene/                          ← run_api.py baseline
users_sd/, users_diff/                ← matting outputs
```

### Notebook (recommended)

Open `main_gd.ipynb`, run all cells top to bottom. `load_models()` runs at import. Then:

```python
SAMPLE = "user_8"
res = run_pipeline(OUTPUT_DIR / SAMPLE, thresh=0.5)
```

The cells below give you the tuning surface, in the order you should look at them when a
result is wrong:

1. **All DINO boxes + labels** — plus the RAM tag list and the assembled prompt. First
   place to look: were the right things even detected?
2. **Per-object BiRefNet matte grid** — one tile per DINO box, plus the full-image
   fallback, each with its padded box drawn. Shows exactly what BiRefNet returned inside
   each detection. `run_pipeline` unions all of these; here they are separate.
3. **3×3 amodal-vs-visible panel** — input, empty room, objects, rug, amodal wall, visible
   wall, amodal floor, visible floor, overlay.

The batch cell at the bottom loops every sample folder. It ships **commented out** on
purpose — each iteration bills.

### CLI

```bash
python run.py --sample living_23                    # one
python run.py --sample Bedroom --sample living_23   # several
python run.py --all                                 # every folder with downsample.jpg
python run.py --all --thresh 0.4                    # looser object threshold

python run_api.py --src users_gd --out image2scene --workers 5   # raw API baseline
python generate_trimap.py --input <masks> --output <trimaps>
```

> 💸 **Every empty-room run bills 1 Imagen call + 1 image2scene job + 1 MoGe job per
> sample.** There is no dry-run mode. Test on one sample before any batch.

Note `run.py` reads `downsample.jpg` from `output-test/`, while `main_gd.ipynb` reads
`original.jpg` from `users_gd/`. Historical drift — check which one you have.

---

## 6. Tuning guide — symptom to knob

| Symptom | Cause | Fix |
| --- | --- | --- |
| Object left in the wall/floor mask | DINO never boxed it | Check the tag list in viz cell 1. Lower `DINO_BOX_THRESH`, or add the class to `OBJECT_PROMPT`. |
| Flat wall item (poster, mirror) left in | BiRefNet is blind to flat items and DINO wasn't told to look | Add the class to the prompt — this is the known residual gap (§3). |
| Wall itself got erased and inpainted | A structural RAM tag reached DINO | Add it to `RAM_EXCLUDE` / `RAM_EXCLUDE_KW`. |
| Object halo / fringe around removed items | Matte too tight, or inpaint mask too tight | Raise `pad` (0.05 → 0.08) or `mask_dilation` (0.03 → 0.05). |
| Imagen *added* furniture | `EMPTY_INPAINT_PROMPT` is non-empty | Set it back to `""`. |
| Two real walls merged into one mask | The API returned one instance | Upstream limitation of the API — or use `main_normal.ipynb`, which splits by plane offset. |
| Sockets detected on blank wall | DINO hallucinating small classes | That's what `RAM_PLUG_KW` gating is for — verify RAM++ actually tagged one. |
| Mask edges blocky / miss soft boundaries | Binary mask, by design | Run the matting stage (§8). |
| `Missing key(s) ... bn ...` on checkpoint load | `batch_size` config race | See the `_BS1Config` workaround in §4. |

---

## 7. The variants, and what each one teaches

Every alternative pipeline in the repo is a controlled experiment. Keeping them is
deliberate — they document what was tried and why the main path won.

| Notebook | Changes one thing | Result |
| --- | --- | --- |
| `main_gd.ipynb` | — (baseline) | ⭐ **best** |
| `new_gd.ipynb` | BiRefNet `epoch_15` instead of `epoch_10_new` | checkpoint sweep |
| `main_yolo.ipynb` | YOLO-World + fixed class list instead of RAM++ → DINO | see 7.1 |
| `main_yolo_nano.ipynb` | Gemini 2.5 Flash Image instead of Imagen | see 7.2 |
| `main_normal.ipynb` | Drops Imagen + API entirely | see §9 |

### 7.1 RAM++ → DINO vs YOLO-World

YOLO-World is open-vocab with built-in NMS and is faster — one model instead of two. But
its class list is **fixed at config time**. Rooms are long-tailed: the class you didn't
list is the one in this photo. RAM++ solves the vocabulary problem *per image*, which is
worth the extra model. Keep the YOLO variant for speed-constrained deployments where you
control the furniture distribution.

### 7.2 Imagen vs Gemini "nano banana"

This is the clearest lesson in the repo.

- **Imagen** `EDIT_MODE_INPAINT_REMOVAL` takes a **mask** and rewrites only masked pixels.
  Output is pixel-registered with the input. Masks transfer back for free.
- **Gemini nano banana** (2.5 Flash Image) edits from a **prompt** — there is no mask
  channel. It may subtly move, recolour or re-render pixels *outside* the region you meant
  to change. Now the empty room is no longer pixel-aligned, and every mask coordinate
  computed on it is suspect. Recovering alignment needs a homography estimate, which
  introduces its own error.

**The capability that matters is not image quality — it is the guarantee about untouched
pixels.** A mask-based editor gives you a contract; a prompt-based editor does not.

---

## 8. The matting stage — SDMatte vs DiffMatte

Everything above produces **hard binary masks**. Edges are stair-stepped; soft boundaries
(a curtain edge, a plant fringe, motion blur) cannot be represented at all. The matting
stage refines a wall mask into a soft **alpha**.

### Trimap discipline

Both `main_sd.ipynb` and `main_diff.ipynb` import the **same** function,
`generate_trimap.wall_trimap()`. That is what makes the comparison valid: identical input,
so any difference isolates the network. Worth copying as a methodology habit — a shared
import cannot drift the way two copy-pasted preprocessing blocks will.

Trimap encoding: `1` = definite wall, `0` = definite background, `0.5` = unknown band,
dilated `TRIMAP_SIZE` (0.75 % of the mask's mean dimension, clamped to [3, 30] px)
**outward** from every boundary.

Source is `image2scene/<sample>/wall_mask.png` — written by `run_api.py` straight from
`original.jpg`. That is the *actually visible* wall, deliberately **not**
`users_gd/<s>/wall_mask_visible.png`, which the API derived from an object-removed image.
Matting must be trained on real photo edges; an inpainted edge is synthetic and the matting
network has no business seeing it.

`TRIMAP_CUT_OBJECTS=True` forces band pixels that fall on the BiRefNet object mask to
definite `0` — the network should never be asked to decide whether a sofa is wall.

### The two networks

| | SDMatte | DiffMatte |
| --- | --- | --- |
| prompt types | trimap / mask / box / points | **trimap only** |
| backbone | Stable Diffusion U-Net (~1.3 B) | ViT-S (29 M) + tiny diffusion decoder |
| iterations | 1 step, denoise in latent space | N steps, denoise the **alpha** directly |
| VRAM @1024² | ~5 GB (fp16) | ~1.6 GB (fp32) |

### The result (`compare_sd_diff/`, 50 samples, shared trimap, `alpha_thresh=0.5`)

| metric | value |
| --- | --- |
| mean IoU | **0.981** |
| pixel agreement | 0.995 |
| MAE | 0.0044 |
| unknown-band fraction | 1.98 % |
| **band** agreement | 0.740 |
| **band** MAE | 0.250 |

**How to read this.** Whole-image IoU of 0.98 is almost meaningless — 98 % of pixels are
definite fg/bg that the trimap already decided, so both networks trivially agree there. The
informative number is **band agreement: 0.740**. Inside the 2 % of pixels that are actually
ambiguous — the only pixels matting exists to resolve — the two networks disagree a
quarter of the time.

The engineering conclusion: **DiffMatte gets there with a 45× smaller backbone and ⅓ the
VRAM.** Unless the band disagreement resolves in SDMatte's favour on your data, the small
model wins. Per-sample numbers in `metrics.csv` / `metrics.json`; side-by-side panels in
`user_N.png`.

**Always define your metric over the region the method is responsible for.** A
whole-image average will hide the effect you are trying to measure.

### Practical notes

- SDMatte is built **config-only** (`load_weight=False` → diffusers `from_config` on the HF
  snapshot's `unet/vae/text_encoder/scheduler/tokenizer`), then `SDMatte.pth` is loaded on
  top. `weights_only=False` + `mmap=True` are **required** — the checkpoint is a
  `DetectionCheckpointer` dict pickling an omegaconf `ListConfig`.
- DiffMatte needs no detectron2; a small `d2shim` supplies the handful of symbols it imports.
- SDMatte's `points` mode cannot encode negatives, and signed blobs collapse the alpha to
  zero. Trimap is the only mode that works reliably.
- `PROMPT_MODE="trimap"` skips RAM++ and DINO entirely — ~4 GB VRAM and ~5 s/sample saved.
- EXIF `Orientation != 1` samples (`user_3`, `user_13`) are normalized **in memory** via
  `ImageOps.exif_transpose`, never on disk.

---

## 9. The structure-first alternative — `main_normal.ipynb`

A fundamentally different approach ([ADR 0008](adr/0008-structure-first-normal-clustering.md)),
motivated by two real problems with the main path: Imagen is slow and billed, and it
**ghosts** — hallucinating faint content into the emptied room, which the segmenter then
faithfully segments.

No RAM++, no DINO, no Imagen, no SAM, no external API:

1. **BiRefNet** (fine-tuned `epoch_10.pth`) segments *movable objects* → **invert** →
   structure mask = wall + floor + ceiling shell. Works because Hypersim's background class
   set teaches door/window/mirror/ceiling as *structure*, not foreground — the flat-occluder
   blind spot from §3 becomes an asset here.
2. **Metric3D-v2** — one feed-forward net giving per-pixel **metric depth + surface normal**.
   Chosen over Marigold: no diffusion latency (consistent with the "kill Imagen for speed"
   motive) and depth + normal from a single pass.
3. Backproject structure pixels to a point cloud (`fx = fy = 0.7·max(H,W)`), then split:
   - **floor / ceiling / wall** by `angle(normal, UP_CAM)` with a **30° cone** — these are
     ~90° apart, so the threshold is not delicate;
   - **wall instances** by **HDBSCAN** on `normal ⊕ W_OFFSET·(z-scored plane offset d)`
     where `d = point · n`;
   - **2-D connected components** for spatially disjoint same-plane pieces.

### Why depth is mandatory, not optional

The motivating case (`user_2`): two **jogged sub-walls facing the same direction**. They
have **identical normals**, and they can be 2-D connected. Neither normal angle nor
connectivity can separate them. Only the **plane offset `d = point · n`** — the signed
distance along the shared normal — distinguishes two parallel planes at different depths.
That single geometric fact is why a depth model is required rather than a normals-only one.

`W_OFFSET` trades the two cues off: `0` = pure-normal (same-faced walls merge), higher =
more aggressive offset-based splitting.

HDBSCAN is fit on a 30 k-pixel subsample and all wall pixels assigned by **nearest cluster
centroid** — `approximate_predict` on ~1 M pixels is too slow. Standard scaling trick,
worth remembering.

### Verified result (`user_2`)

`epoch_10.pth` structure fraction **0.995** — correctly leaves an empty hallway as
structure. Metric3D-v2 + HDBSCAN found 4 clusters and cleanly split left wall / right wall /
far wall / floor / ceiling → 8 instance masks after 2-D CC. The separation method is
validated.

### The checkpoint lesson

The later `epoch_10_new.pth` re-train scored a structure fraction of **0.34** on the same
empty hallway — it called 66 % of a bare room "object". Unusable, and the re-train was
cancelled. That number **is the measurement of the synth→real gap**: more epochs on
synthetic Hypersim data made the model *worse* on real photographs.

Note the asymmetry: `epoch_10_new` is fine in `main_gd.ipynb`, where BiRefNet only mattes
*inside a DINO box* and never has to judge empty wall. The same checkpoint is unusable in
`main_normal`, where the whole image is its responsibility. **A checkpoint is only good
relative to the job it is given** — always re-benchmark on the actual downstream task.

### Known limits

- **Visible-only.** Inversion cannot recover what is behind furniture. Every object leaves
  a hole. `*_mask_amodal` is not reproduced. This is the trade: speed and locality for
  amodality.
- **No safety net.** A BiRefNet miss leaves an object as fake surface; a false positive
  punches a hole in the wall. Unlike the main path, there is no second detector to catch
  the first one's errors. Quality rides entirely on one model.

Env: `mmengine` + `mmcv` shim (`Config` → mmengine; mmcv 1.x will not build on torch 2.12 /
cu130), `hdbscan` against sklearn 1.7.2.

---

## 10. Training — BiRefNet on Hypersim

**Goal:** teach BiRefNet that foreground = *everything that is not room structure*. The
stock model segments salient objects; this fine-tune redefines the foreground class.

**Data:** [Hypersim](https://github.com/apple/ml-hypersim) (Apple) — photorealistic
*synthetic* indoor scenes, **CC BY-SA 3.0 (commercial use OK)**, with pixel-perfect rendered
semantic labels. Labels are free and exact, which is precisely why synthetic data is
attractive here — and §9 is the bill for it.

| Notebook | Method |
| --- | --- |
| `trainer.ipynb` | Guided walkthrough of the first fine-tune: download (tonemap + semantic), prep, train. Heavy steps are flag-guarded and **OFF by default**, so *Run All* stays fast — a good pattern for a notebook that doubles as documentation. |
| `trainer_v2.ipynb` | Proper methodology (below). |

### `trainer_v2` methodology — worth imitating

- **Split by SCENE, 70/15/15.** Not by frame. Hypersim gives many frames per room; a
  frame-level split puts near-duplicate views of the same room in both train and test and
  inflates your metrics enormously. Splitting by scene is the only honest option.
- **Phase A** — train on TRAIN, monitor VAL every epoch, **early stop** on val mean-IoU
  (patience, max 50 epochs), keep the **best** checkpoint.
- **Phase B** — once VAL has served model selection, fold it into training for a few more
  epochs. The data has done its job; now it can be training signal.
- **TEST is looked at exactly once**, at the very end.
- Checkpoints + `history.json` written **every epoch** to `ckpts/hypersim_es/`, so an
  interrupted run is not a lost run.

Hardware target: RTX 5070, 12 GB → batch size 1 at 640². Which is exactly why the
`batch_size=1` BatchNorm→Identity swap bites at load time (§4).

Checkpoints: `hq-mat/BiRefNet/ckpts/hypersim/epoch_{5,10,15,20,25,30}.pth` and
`ckpts/hypersim_es/{best,final}.pth`. Prep docs: `hq-mat/BiRefNet/hypersim_prep/README.md`.

---

## 11. Transferable lessons

If you take nothing else from this repo:

1. **Reframe the impossible requirement.** "Segment the wall behind the sofa" is not
   solvable by a segmenter. "Remove the sofa, then segment" is. Move the hard part to a
   model that is good at it.
2. **A generative editor's contract about *untouched* pixels can matter more than its
   output quality.** Imagen's masked inpaint guarantees registration; that guarantee is the
   whole reason no homography is needed.
3. **Know your model's structural blind spot.** BiRefNet cannot see flat wall-anchored
   objects — not a bug, a consequence of what "salient foreground" means. Design the second
   detector around the first one's blind spot, not around its error rate.
4. **Generate the vocabulary per input.** Open-vocab detectors are closed-set per call.
   RAM++ → DINO turns a fixed list into an adaptive one.
5. **Measure where the method acts.** Whole-image IoU 0.98 hid a band agreement of 0.74.
6. **A checkpoint is good only relative to its job.** `epoch_10_new` is fine inside a box
   and catastrophic on a full image.
7. **Hide latency you cannot remove.** MoGe starts first, is collected last, costs ~0.
8. **Keep your ablations.** Every `main_*.ipynb` here is a controlled one-variable
   experiment, and the design record explains what each one proved.
9. **Share preprocessing by import, not by copy.** One `wall_trimap()` is why the
   SDMatte/DiffMatte comparison is trustworthy.
10. **Outputs are non-destructive.** New PNGs per sample dir; no input is ever modified.
    Cheap to do, and it means a bad run costs you nothing but money.

---

## 12. Open risks

- **VRAM.** BiRefNet + DINO + SAM/Metric3D co-resident on a 12 GB card may need sequential
  loading or CPU offload.
- **DINO false positives** — mirror ↔ window, framed art ↔ window. The confidence threshold
  needs per-deployment tuning.
- **Synth → real gap** in the Hypersim fine-tune, quantified by the `epoch_10_new`
  regression (§9).
- **Flat unnamed wall items** remain the acknowledged residual miss (§3).
- **Cost and non-determinism.** Imagen is billed and stochastic; the same input can yield
  slightly different empty rooms, hence slightly different amodal masks. Do not expect
  bit-reproducible outputs across runs.
