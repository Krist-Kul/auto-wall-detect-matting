#!/usr/bin/env python3
"""Full empty-room surface pipeline.

Flow per sample (output-test/<SAMPLE>/):
    1. RAM++     -> auto-tag image -> tags become the GroundingDINO prompt
       DINO      -> open-vocab object bboxes + labels (class-agnostic NMS)
       BiRefNet  -> object mask + rug mask (things to remove)
                    -> object_mask_birefnet.png, rug_mask_birefnet.png
    2. Imagen    -> inpaint-remove objects AND rug -> empty -> empty_imagen.png
    3. image2scene API -> auto_detect on the EMPTY room -> wall / floor masks
    4. visible   = surface(empty) - objects; floor also - rug  -> *_mask_visible.png
       amodal    = surface(empty)  (full, incl. hidden)        -> *_mask_amodal.png
    5. overlay   = input + wall(red) + floor(green)            -> overlay.png

No auto_detect.json input: objects + rug come from RAM++/GroundingDINO, wall/floor
from the image2scene API re-run on the empty room. Only downsample.jpg per sample.

The Imagen inpaint keeps the room geometry (only the masked pixels change), so the
empty room is aligned to the original and no homography is needed; the API masks are
resized back to the original resolution.

Usage:
    python run.py --sample living_23
    python run.py --sample Bedroom --sample living_23
    python run.py --all
"""
import os
import io
import sys
import json
import time
import base64
import zipfile
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import requests
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output-test"
BIREFNET_REPO = ROOT / "hq-mat" / "BiRefNet"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Lazily-loaded heavy models (filled by load_models()).
_bi = None            # BiRefNet
_bi_tf = None         # BiRefNet input transform
dino = dino_proc = None       # GroundingDINO
ram_model = ram_tf = None     # RAM++
genai_client = None

# RAM++ (Recognize Anything Plus) auto-tagging -> tags become the DINO prompt
RAM_CKPT = ROOT / "ram_plus_swin_large_14m.pth"
RAM_IMG_SIZE = 384
RAM_VIT = "swin_l"
# RAM tags we never want DINO to detect/remove (structural surfaces, scene words).
# Matched case-insensitively: exact set OR any tag containing one of RAM_EXCLUDE_KW.
RAM_EXCLUDE = {
    "window", "doorway", "window blind", "blind", "shutter",
    "hardwood", "tile", "carpet", "flooring", "wood floor", "curtain",
}
RAM_EXCLUDE_KW = ("wall", "floor", "ceiling", "room")  # substring match

# GroundingDINO open-vocab detection. Prompt is built from RAM tags at runtime;
# OBJECT_PROMPT is the fallback when RAM returns nothing usable.
DINO_ID = "IDEA-Research/grounding-dino-base"
OBJECT_PROMPT = ("furniture. chair. table. sofa. couch. bed. cabinet. shelf. "
                 "lamp. light. plant. rug. pillow. curtain. tv. picture. frame. "
                 "mirror. vase. appliance. box. decoration. object.")
DINO_BOX_THRESH = 0.25
DINO_TEXT_THRESH = 0.20
NMS_IOU = 0.5  # class-agnostic NMS: drop boxes overlapping > this IoU
# DINO labels treated as rug/floor-covering (own removal + floor-visible subtract)
RUG_LABELS = ("rug", "carpet", "mat", "runner")

# Distinct overlay colours, one per individual wall (cycled if more walls than colours)
WALL_COLORS = [(230, 25, 75), (245, 130, 48), (255, 225, 25), (0, 130, 200),
               (66, 212, 244), (145, 30, 180), (240, 50, 230), (170, 110, 40)]
FLOOR_COLOR = (60, 180, 75)

IMAGEN_EDIT_MODEL = "imagen-3.0-capability-001"
# Removal works best with an EMPTY prompt; a descriptive prompt makes Imagen *generate*
# new furniture instead of clearing the masked region.
EMPTY_INPAINT_PROMPT = ""

# image2scene API: re-run auto_detect on the Imagen empty room
API_BASE = "https://image2scene-dev.wedolabs.net"


# ----------------------------------------------------------------------------- IO
def load_sample(sample_dir: Path):
    """Load downsample.jpg for one sample folder (no auto_detect.json needed:
    objects+rug come from RAM++/GroundingDINO, wall/floor from the API re-run)."""
    return Image.open(sample_dir / "downsample.jpg").convert("RGB")


def mask_to_pil(mask) -> Image.Image:
    """Boolean/float mask -> 8-bit single-channel PIL image (255 = surface)."""
    return Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")


# ------------------------------------------------------------------ model loading
def load_models():
    """Load BiRefNet + RAM++ + GroundingDINO + the Vertex/Imagen client (once)."""
    global _bi, _bi_tf, dino, dino_proc, genai_client, ram_model, ram_tf

    # --- env / Vertex auth (gcloud ADC, not an API key) ---
    load_dotenv(override=True)
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and not os.path.exists(gac):       # drop a stale SA path -> fall back to ADC
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    assert project, "Put GOOGLE_CLOUD_PROJECT=<project id> in .env (Vertex uses ADC)"

    torch.set_float32_matmul_precision("high")

    # --- BiRefNet (local repo) ---
    if str(BIREFNET_REPO) not in sys.path:
        sys.path.insert(0, str(BIREFNET_REPO))
    from models.birefnet import BiRefNet
    _bi = BiRefNet.from_pretrained("zhengpeng7/birefnet").to(device).eval()
    _bi_tf = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- RAM++ (auto image tagging -> open-vocab labels for DINO) ---
    from ram.models import ram_plus
    from ram import get_transform
    ram_model = ram_plus(pretrained=str(RAM_CKPT), image_size=RAM_IMG_SIZE,
                         vit=RAM_VIT).eval().to(device)
    ram_tf = get_transform(image_size=RAM_IMG_SIZE)

    # --- GroundingDINO (open-vocab object detection) ---
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    dino_proc = AutoProcessor.from_pretrained(DINO_ID)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(device).eval()

    # --- Imagen (Vertex via google-genai) ---
    from google import genai
    genai_client = genai.Client(vertexai=True, project=project, location=location)

    print(f"models ready on {device} | project={project} loc={location}")


# ------------------------------------------------------------------- 1. BiRefNet
@torch.no_grad()
def _birefnet_crop(image: Image.Image, box):
    """BiRefNet foreground for one crop, pasted back onto a full-image canvas."""
    w, h = image.size
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    canvas = np.zeros((h, w), dtype=np.float32)
    if x2 <= x1 or y2 <= y1:
        return canvas
    pred = _bi(_bi_tf(image.crop((x1, y1, x2, y2))).unsqueeze(0).to(device))
    pred = pred[-1].sigmoid().cpu()[0, 0]
    canvas[y1:y2, x1:x2] = F.interpolate(pred[None, None], size=(y2 - y1, x2 - x1),
                                         mode="bilinear", align_corners=True)[0, 0].numpy()
    return canvas


def _filter_ram_tags(tags):
    """Drop structural/scene tags; lowercase + dedupe, preserve order."""
    out, seen = [], set()
    for t in tags:
        t = t.strip().lower()
        if (not t) or t in RAM_EXCLUDE or t in seen:
            continue
        if any(k in t for k in RAM_EXCLUDE_KW):
            continue
        seen.add(t)
        out.append(t)
    return out


@torch.no_grad()
def ram_autotag(image: Image.Image):
    """RAM++ auto-labels the image -> (DINO prompt string, tag list).
    Tags are filtered (RAM_EXCLUDE) and formatted '. '-separated for GroundingDINO."""
    from ram import inference_ram
    res = inference_ram(ram_tf(image).unsqueeze(0).to(device), ram_model)
    tags = _filter_ram_tags(res[0].split(" | "))
    prompt = (". ".join(tags) + ".") if tags else OBJECT_PROMPT
    return prompt, tags


@torch.no_grad()
def detect_objects(image: Image.Image, prompt=None,
                   box_thresh=DINO_BOX_THRESH, text_thresh=DINO_TEXT_THRESH,
                   nms_iou=NMS_IOU):
    """GroundingDINO open-vocab -> ([x1,y1,x2,y2] boxes, labels) for room objects.
    `prompt=None` -> RAM++ auto-tags the image and uses those tags as the prompt.
    Class-agnostic NMS drops redundant overlapping boxes (keep highest score)."""
    from torchvision.ops import nms
    if prompt is None:
        prompt, _ = ram_autotag(image)
    inp = dino_proc(images=image, text=prompt, return_tensors="pt").to(device)
    out = dino(**inp)
    res = dino_proc.post_process_grounded_object_detection(
        out, inp["input_ids"], threshold=box_thresh, text_threshold=text_thresh,
        target_sizes=[image.size[::-1]])[0]
    boxes, scores = res["boxes"], res["scores"]
    labels = list(res.get("text_labels", res.get("labels", [])))
    if nms_iou is not None and len(boxes):
        keep = nms(boxes, scores, nms_iou).tolist()       # sorted by score desc
        boxes = boxes[keep]
        labels = [labels[i] for i in keep]
    boxes = [[float(v) for v in b] for b in boxes.tolist()]
    return boxes, labels


def birefnet_objects(image: Image.Image, boxes, pad=0.05, full_fallback=True):
    """Object matte (float) = union of BiRefNet foreground inside each DINO box.
    `pad` grows each box a bit; `full_fallback` also unions a full-image pass to
    catch objects DINO may have missed."""
    w, h = image.size
    out = (_birefnet_crop(image, (0, 0, w, h)) if full_fallback
           else np.zeros((h, w), dtype=np.float32))
    for x1, y1, x2, y2 in (boxes or []):
        bw, bh = x2 - x1, y2 - y1
        out = np.maximum(out, _birefnet_crop(
            image, (x1 - pad*bw, y1 - pad*bh, x2 + pad*bw, y2 + pad*bh)))
    return out


def rug_boxes_from_labels(boxes, labels):
    """Pick the DINO boxes whose label is a rug/floor-covering (RUG_LABELS)."""
    return [b for b, l in zip(boxes, labels)
            if any(k in str(l).lower() for k in RUG_LABELS)]


def birefnet_rug(image: Image.Image, rug_bboxes, thresh=0.5):
    """Rug footprint = BiRefNet foreground inside each rug bbox."""
    w, h = image.size
    r = np.zeros((h, w), dtype=bool)
    for b in (rug_bboxes or []):
        r |= _birefnet_crop(image, b) >= thresh
    return r


# --------------------------------------------------------------------- 2. Imagen
def imagen_empty_room(sample_dir: Path, base_img, remove_mask, out_path):
    """Inpaint-remove the remove_mask region; save empty room resized to base size."""
    from google.genai import types
    w, h = base_img.size
    base_path = str(sample_dir / "downsample.jpg")
    mask_path = str(sample_dir / "inpaint_mask.png")
    mask_to_pil(remove_mask).save(mask_path)

    raw_ref = types.RawReferenceImage(
        reference_image=types.Image.from_file(location=base_path), reference_id=0)
    mask_ref = types.MaskReferenceImage(
        reference_id=1, reference_image=types.Image.from_file(location=mask_path),
        config=types.MaskReferenceConfig(mask_mode="MASK_MODE_USER_PROVIDED",
                                         mask_dilation=0.03))
    resp = genai_client.models.edit_image(
        model=IMAGEN_EDIT_MODEL, prompt=EMPTY_INPAINT_PROMPT,
        reference_images=[raw_ref, mask_ref],
        config=types.EditImageConfig(edit_mode="EDIT_MODE_INPAINT_REMOVAL",
                                     number_of_images=1))
    resp.generated_images[0].image.save(str(out_path))
    empty = Image.open(out_path).convert("RGB")
    if empty.size != (w, h):                  # align to original so API masks match
        empty = empty.resize((w, h), Image.LANCZOS)
        empty.save(out_path)
    return empty


# ---------------------------------------------------------- 3. image2scene API
def api_autodetect(empty_path, out_dir, do_masked_images=True, poll=2.0, timeout=300):
    """POST the empty room to image2scene /run, poll /jobs/{id}, download the result
    zip and extract EVERY file into out_dir ('from-api'). Returns
    (auto_detect dict, api downsample PIL.Image, job_id)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(empty_path, "rb") as f:
        r = requests.post(
            f"{API_BASE}/run",
            files={"input_image": (Path(empty_path).name, f, "image/png")},
            data={"do_masked_images": str(do_masked_images).lower()}, timeout=60)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    deadline = time.time() + timeout
    while True:
        j = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=30).json()
        if j["status"] == "completed":
            break
        if j["status"] == "failed":
            raise RuntimeError(f"image2scene job {job_id} failed: {j.get('message')}")
        if time.time() > deadline:
            raise TimeoutError(f"image2scene job {job_id} timed out (last={j['status']})")
        time.sleep(poll)
    zb = requests.get(j["result_zip_url"], timeout=120).content
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        z.extractall(out_dir)
    with open(out_dir / "auto_detect.json") as f:
        meta = json.load(f)
    api_img = Image.open(out_dir / "downsample.jpg").convert("RGB")
    print(f"  api job {job_id} -> {out_dir} ({len(list(out_dir.rglob('*')))} files)")
    return meta, api_img, job_id


def moge_start(image_path):
    """Kick off a MoGe (monocular geometry) job on `image_path`; return job_id.
    Async: does NOT wait. Call moge_fetch(job_id, ...) later to collect the result.
    Run on the ORIGINAL downsample.jpg (not the empty room)."""
    mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{API_BASE}/run_moge",
            files={"input_image": (Path(image_path).name, f, mime)}, timeout=60)
    r.raise_for_status()
    return r.json()["job_id"]


def moge_fetch(job_id, out_dir, poll=2.0, timeout=300):
    """Poll the MoGe job started by moge_start(); download + extract its result zip
    into out_dir. Returns out_dir."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    while True:
        j = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=30).json()
        if j["status"] == "completed":
            break
        if j["status"] == "failed":
            raise RuntimeError(f"moge job {job_id} failed: {j.get('message')}")
        if time.time() > deadline:
            raise TimeoutError(f"moge job {job_id} timed out (last={j['status']})")
        time.sleep(poll)
    zb = requests.get(j["result_zip_url"], timeout=120).content
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        z.extractall(out_dir)
    print(f"  moge job {job_id} -> {out_dir} ({len(list(out_dir.rglob('*')))} files)")
    return out_dir


def _decode_masks_list(b64_list, dst_size):
    """Decode API base64 masks individually -> list of bool arrays at dst_size (W, H).
    Masks come at the API downsample size; resize (nearest) to dst."""
    W, H = dst_size
    out = []
    for b in (b64_list or []):
        raw = b.split(",", 1)[1] if b.startswith("data:") else b
        m = Image.open(io.BytesIO(base64.b64decode(raw))).convert("L")
        if m.size != (W, H):
            m = m.resize((W, H), Image.NEAREST)
        out.append(np.asarray(m) > 127)
    return out


def _decode_masks(b64_list, dst_size):
    """Union of API base64 masks -> bool array at dst_size (W, H)."""
    masks = _decode_masks_list(b64_list, dst_size)
    if not masks:
        return np.zeros(dst_size[::-1], dtype=bool)
    return np.logical_or.reduce(masks)


# -------------------------------------------------------------------- pipeline
def run_pipeline(sample_dir: Path, thresh=0.5):
    """RAM++/DINO/BiRefNet -> Imagen empty room -> image2scene API auto_detect
    (wall/floor masks) -> subtract objects + rug. API output saved under
    sample_dir/from-api/."""
    img = load_sample(sample_dir)
    w, h = img.size

    timings = {}                                  # stage name -> seconds
    t0 = time.time()

    def _t(name, since):
        timings[name] = time.time() - since
        return time.time()

    # 0. kick off MoGe on the ORIGINAL image (async; collected at the last step,
    #    so it runs in parallel with BiRefNet/Imagen/auto_detect)
    start = time.time()
    moge_job = moge_start(sample_dir / "downsample.jpg")
    start = _t("moge_start", start)

    # 1. RAM++ auto-tags the image -> tags drive GroundingDINO -> BiRefNet matte
    obj_prompt, obj_tags = ram_autotag(img)
    obj_boxes, obj_labels = detect_objects(img, prompt=obj_prompt)
    rug_boxes = rug_boxes_from_labels(obj_boxes, obj_labels)
    objects = birefnet_objects(img, obj_boxes) >= thresh
    rug = birefnet_rug(img, rug_boxes, thresh=thresh)
    remove = objects | rug
    start = _t("ram_dino_birefnet", start)

    # 2. empty room (Imagen inpaint-removal of objects AND rug)
    empty = imagen_empty_room(sample_dir, img, remove,
                              sample_dir / "empty_imagen.png")
    start = _t("imagen_empty_room", start)

    # 3. auto_detect on the empty room via image2scene -> wall/floor masks
    api_meta, _, _ = api_autodetect(sample_dir / "empty_imagen.png",
                                    sample_dir / "from-api")
    start = _t("api_autodetect", start)

    # 3b. amodal wall/floor = API masks (resized to original pixel space).
    #     walls AND floors kept INDIVIDUALLY (one mask per surface).
    walls_full = _decode_masks_list(api_meta.get("wall_masks"), (w, h))
    floors_full = _decode_masks_list(api_meta.get("floor_masks"), (w, h))
    wall_full = (np.logical_or.reduce(walls_full) if walls_full
                 else np.zeros((h, w), dtype=bool))
    floor_full = (np.logical_or.reduce(floors_full) if floors_full
                  else np.zeros((h, w), dtype=bool))

    # 4. visible = full surface (from empty room) minus objects; floor also minus rug
    walls_vis = [wm & ~objects for wm in walls_full]          # per-wall visible
    floors_vis = [fm & ~objects & ~rug for fm in floors_full]  # per-floor visible
    wall_vis = wall_full & ~objects                           # union visible
    floor_vis = floor_full & ~objects & ~rug
    start = _t("decode_visible", start)

    out = {
        "object_mask_birefnet.png": objects,
        "rug_mask_birefnet.png": rug,
        "wall_mask_amodal.png": wall_full,
        "floor_mask_amodal.png": floor_full,
        "wall_mask_visible.png": wall_vis,
        "floor_mask_visible.png": floor_vis,
    }
    for name, m in out.items():
        mask_to_pil(m).save(sample_dir / name)

    # independent (per-surface) visible masks -> masked_images/{wall,floor}_mask_visible_NN.png
    mi_dir = sample_dir / "masked_images"
    mi_dir.mkdir(parents=True, exist_ok=True)
    for i, wm in enumerate(walls_vis, 1):
        mask_to_pil(wm).save(mi_dir / f"wall_mask_visible_{i:02d}.png")
    for i, fm in enumerate(floors_vis, 1):
        mask_to_pil(fm).save(mi_dir / f"floor_mask_visible_{i:02d}.png")

    # overlay: input + each wall in its own colour + floor (green)
    overlay = np.asarray(img).astype(np.float32)
    layers = [(wm, WALL_COLORS[i % len(WALL_COLORS)]) for i, wm in enumerate(walls_vis)]
    layers.append((floor_vis, FLOOR_COLOR))
    for m, rgb in layers:
        sel = (np.asarray(m) > 0)[..., None]
        overlay = np.where(sel, 0.5 * overlay + 0.5 * np.array(rgb), overlay)
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(sample_dir / "overlay.png")
    start = _t("save_masks_overlay", start)

    # last step: collect the MoGe geometry started at step 0
    moge_dir = moge_fetch(moge_job, sample_dir / "moge")
    start = _t("moge_fetch", start)

    # per-stage + total timing -> time.txt
    timings["total"] = time.time() - t0
    lines = [f"{name:<20s} {secs:8.2f}s" for name, secs in timings.items()]
    (sample_dir / "time.txt").write_text("\n".join(lines) + "\n")
    print("  timings: " + ", ".join(f"{k}={v:.1f}s" for k, v in timings.items()))

    return {"img": img, "empty": empty, "objects": objects, "rug": rug, "moge_dir": moge_dir,
            "timings": timings,
            "obj_boxes": obj_boxes, "obj_labels": obj_labels, "rug_boxes": rug_boxes,
            "obj_prompt": obj_prompt, "obj_tags": obj_tags,
            "wall_full": wall_full, "floor_full": floor_full,
            "walls_visible": walls_vis, "floors_visible": floors_vis,
            "wall_visible": wall_vis, "floor_visible": floor_vis}


# ------------------------------------------------------------------------- CLI
def _iter_samples(args):
    if args.all:
        return sorted(p for p in OUTPUT_DIR.iterdir()
                      if (p / "downsample.jpg").exists())
    return [OUTPUT_DIR / s for s in args.sample]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="append", default=[],
                    help="sample folder name under output-test/ (repeatable)")
    ap.add_argument("--all", action="store_true", help="process every sample")
    ap.add_argument("--thresh", type=float, default=0.5, help="BiRefNet object threshold")
    args = ap.parse_args()

    samples = _iter_samples(args)
    if not samples:
        ap.error("nothing to do: pass --sample <name> or --all")

    load_models()
    for d in samples:
        if not (d / "downsample.jpg").exists():
            print(f"skip {d.name}: no downsample.jpg")
            continue
        print(f"\n=== {d.name} ===")
        try:
            run_pipeline(d, thresh=args.thresh)
            print(f"done {d.name}: wrote object/wall/floor masks + empty_imagen.png")
        except Exception as e:
            print(f"FAIL {d.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
