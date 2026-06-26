# Amodal surface pipeline

Recovers **amodal** wall/floor masks for a room photo — including the parts hidden
behind furniture — by removing the furniture with Imagen and re-detecting surfaces on
the resulting empty room.

See [`pipeline.html`](pipeline.html) for the full flow diagram.

## Flow

Per sample (`output-test/<SAMPLE>/`), starting from only `downsample.jpg`:

1. **RAM++** auto-tags the image → tags become the **GroundingDINO** open-vocab prompt
   → object bboxes + labels (class-agnostic NMS). **BiRefNet** mattes the foreground
   inside each box → object mask + **rug** mask (the things to remove).
2. **Imagen 3** (`imagen-3.0-capability-001`) inpaint-removes objects ∪ rug → a bare
   room, pixel-aligned to the original (inpaint only changes masked pixels, so no
   homography is needed).
3. **image2scene API** re-runs `auto_detect` on the empty room → amodal wall/floor
   masks (no parts hidden by furniture, since the furniture is gone).
4. **visible** = amodal surface − objects; floor also − rug.

No `auto_detect.json` *input* is required: objects + rug come from RAM++/GroundingDINO,
wall/floor from the API re-run. The API's own `auto_detect.json` is saved under
`from-api/`.

## Outputs

Written into each sample folder:

| File | Meaning |
| --- | --- |
| `object_mask_birefnet.png` | foreground objects (removed) |
| `rug_mask_birefnet.png` | rug / floor-covering (removed) |
| `empty_imagen.png` | Imagen empty room |
| `inpaint_mask.png` | mask sent to Imagen (objects ∪ rug) |
| `wall_mask_amodal.png` / `floor_mask_amodal.png` | full surfaces incl. hidden |
| `wall_mask_visible.png` / `floor_mask_visible.png` | surfaces minus objects (floor also minus rug) |
| `overlay.png` | input + wall (red) + floor (green) |
| `from-api/` | raw image2scene API result (incl. its `auto_detect.json`) |

## Setup

```bash
conda create -n birefnet python=3.10 && conda activate birefnet
pip install -r requirements.txt
# Install torch/torchvision for your CUDA: https://pytorch.org
```

Extra assets, not pip-installable:

- **BiRefNet** — local repo at `hq-mat/BiRefNet` (added to `sys.path` at runtime;
  weights auto-pulled from `zhengpeng7/birefnet` on the HF Hub).
- **RAM++** checkpoint — `ram_plus_swin_large_14m.pth` in the repo root.

### Vertex / Imagen auth

Imagen runs on Vertex AI via **ADC** (gcloud Application Default Credentials), not an
API key. Put your project in a `.env` file at the repo root:

```
GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
GOOGLE_CLOUD_LOCATION=us-central1
```

Then `gcloud auth application-default login`.

## Usage

```bash
python run.py --sample living_23                 # one sample
python run.py --sample Bedroom --sample living_23 # several
python run.py --all                               # every folder with downsample.jpg
```

Options: `--thresh` (BiRefNet object threshold, default `0.5`).

> Each run **bills 1 Imagen call + 1 image2scene API job per sample.**

`main_short.ipynb` runs the same code inline with visualizations (DINO boxes,
masks, amodal-vs-visible overlays) for tuning `OBJECT_PROMPT`, `RAM_EXCLUDE`,
`RUG_LABELS`, `DINO_BOX_THRESH`, and `NMS_IOU`.

## Tuning knobs

Top of `run.py`:

- `RAM_EXCLUDE` / `RAM_EXCLUDE_KW` — RAM tags never passed to DINO (structural/scene words).
- `OBJECT_PROMPT` — fallback DINO prompt when RAM returns nothing usable.
- `DINO_BOX_THRESH` / `DINO_TEXT_THRESH` / `NMS_IOU` — GroundingDINO detection.
- `RUG_LABELS` — DINO labels treated as rug/floor-covering.
