#!/usr/bin/env python3
"""Baseline: call the image2scene API directly on original.jpg (no RAM++/DINO/
BiRefNet/Imagen object-removal preprocessing that run.py does), and keep the raw
wall/floor masks it returns.

For every user_N/original.jpg under the source dir:
    1. POST original.jpg to image2scene /run, poll until done.
    2. Decode wall/floor masks from auto_detect.json.
    3. Write them to image2scene/user_N/ as PNGs, plus the same overlay style as
       run.py (per-wall colour + green floor) drawn on original.jpg.

Unlike users_gd/<s>/wall_mask_visible.png, wall_mask.png here is the *raw* union of
the API's wall masks -- no BiRefNet object subtraction. main_sd.ipynb reads it as
its trimap source.

Layout written per sample:
    image2scene/user_N/original.jpg      EXIF-normalized copy (what was POSTed)
                       wall_mask.png     union of wall_masks (255 = wall)
                       floor_mask.png    union of floor_masks
                       walls/wall_00.png per-wall masks, in API order
                       sam_refine.png    overlay preview
                       auto_detect.json  API metadata, base64 masks stripped

Usage:
    python run_api.py                       # users_gd -> image2scene/
    python run_api.py --src users_yolo      # different source of original.jpg
    python run_api.py --out image2scene_v2  # different destination
    python run_api.py --workers 8           # more parallel API jobs
    python run_api.py --overwrite           # redo samples that already have wall_mask.png
"""
import argparse
import base64
import io
import json
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
API_BASE = "https://image2scene-dev.wedolabs.net"

# same overlay colours as run.py, for a visually consistent comparison
WALL_COLORS = [(230, 25, 75), (245, 130, 48), (255, 225, 25), (0, 130, 200),
               (66, 212, 244), (145, 30, 180), (240, 50, 230), (170, 110, 40)]
FLOOR_COLOR = (60, 180, 75)

MASK_KEYS = ("wall_masks", "floor_masks")   # base64 blobs -> written as PNGs instead


def api_autodetect(image_path, poll=2.0, timeout=300):
    """POST image_path to image2scene /run, poll /jobs/{id}, return the decoded
    auto_detect dict (result_zip's auto_detect.json)."""
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{API_BASE}/run",
            files={"input_image": (Path(image_path).name, f, "image/jpeg")},
            data={"do_masked_images": "true"}, timeout=60)
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
        with z.open("auto_detect.json") as f:
            meta = json.load(f)
    return meta, job_id


def _decode_masks_list(b64_list, dst_size):
    """Decode API base64 masks -> list of bool arrays at dst_size (W, H)."""
    W, H = dst_size
    out = []
    for b in (b64_list or []):
        raw = b.split(",", 1)[1] if b.startswith("data:") else b
        m = Image.open(io.BytesIO(base64.b64decode(raw))).convert("L")
        if m.size != (W, H):
            m = m.resize((W, H), Image.NEAREST)
        out.append(np.asarray(m) > 127)
    return out


def _save_mask(mask, path):
    """Bool array -> 8-bit L PNG (255 = set)."""
    Image.fromarray(np.asarray(mask).astype(np.uint8) * 255, mode="L").save(path)


def _stage_original(src_jpg: Path, dst_jpg: Path):
    """Copy original.jpg to dst, applying EXIF orientation so that every downstream
    consumer (API masks, notebook) agrees on pixel coordinates. Bytes are copied
    verbatim when there is nothing to rotate."""
    raw = Image.open(src_jpg)
    if raw.getexif().get(274, 1) in (1, None):
        raw.close()
        shutil.copy2(src_jpg, dst_jpg)
    else:
        ImageOps.exif_transpose(raw).convert("RGB").save(dst_jpg, quality=95)


def process_one(src_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    original = out_dir / "original.jpg"
    _stage_original(src_dir / "original.jpg", original)

    img = Image.open(original).convert("RGB")
    w, h = img.size

    meta, job_id = api_autodetect(original)
    walls = _decode_masks_list(meta.get("wall_masks"), (w, h))
    floors = _decode_masks_list(meta.get("floor_masks"), (w, h))
    empty = np.zeros((h, w), dtype=bool)
    wall = np.logical_or.reduce(walls) if walls else empty
    floor = np.logical_or.reduce(floors) if floors else empty

    _save_mask(wall, out_dir / "wall_mask.png")
    _save_mask(floor, out_dir / "floor_mask.png")
    walls_dir = out_dir / "walls"
    walls_dir.mkdir(exist_ok=True)
    for i, wm in enumerate(walls):
        _save_mask(wm, walls_dir / f"wall_{i:02d}.png")

    (out_dir / "auto_detect.json").write_text(json.dumps(
        {**{k: v for k, v in meta.items() if k not in MASK_KEYS},
         "job_id": job_id, "image_size": [w, h],
         "n_wall_masks": len(walls), "n_floor_masks": len(floors)}, indent=1))

    overlay = np.asarray(img).astype(np.float32)
    layers = [(wm, WALL_COLORS[i % len(WALL_COLORS)]) for i, wm in enumerate(walls)]
    layers.append((floor, FLOOR_COLOR))
    for m, rgb in layers:
        sel = m[..., None]
        overlay = np.where(sel, 0.5 * overlay + 0.5 * np.array(rgb), overlay)
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(out_dir / "sam_refine.png")
    return job_id, len(walls)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="users_gd", help="dir holding user_*/original.jpg")
    ap.add_argument("--out", default="image2scene", help="destination dir")
    ap.add_argument("--workers", type=int, default=5, help="parallel API jobs")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo samples that already have wall_mask.png")
    args = ap.parse_args()

    src_root, out_root = ROOT / args.src, ROOT / args.out
    samples = sorted(p.parent for p in src_root.glob("user_*/original.jpg"))
    if not args.overwrite:
        samples = [s for s in samples if not (out_root / s.name / "wall_mask.png").exists()]
    if not samples:
        print("nothing to do")
        return

    print(f"processing {len(samples)} samples with {args.workers} workers -> {out_root}")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, s, out_root / s.name): s for s in samples}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                job_id, n_walls = fut.result()
                ok += 1
                print(f"done {s.name} ({n_walls} walls, job {job_id}) [{ok+fail}/{len(samples)}]")
            except Exception as e:
                fail += 1
                print(f"FAIL {s.name}: {type(e).__name__}: {e} [{ok+fail}/{len(samples)}]")

    print(f"\n{ok} ok, {fail} failed")


if __name__ == "__main__":
    main()
