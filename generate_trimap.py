#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Trimap Generator
=======================
Scans an input directory of binary masks, generates trimaps with a specified
boundary size and pre-scaling method, and saves them to a target output directory.

Also the single source of truth for trimap generation in the matting notebooks: `main_sd.ipynb`
and `main_diff.ipynb` both import `make_trimap()` below, so SDMatte and DiffMatte are always fed
the identical trimap and any comparison between them isolates the network.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Union

import numpy as np

# Add trimap_generator directory to python path
sys.path.append(str(Path(__file__).parent / "trimap_generator"))

try:
    from trimap_generator import extract_image, check_image, trimap, Erosion, Dilation
except ImportError:
    print("Error: Could not import trimap_generator. Make sure the trimap_generator.py file exists.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BatchTrimap")


# ==========================================
# DEFAULTS -- shared by the CLI batch below AND by main_sd.ipynb / main_diff.ipynb
# ==========================================
# Boundary size: an int (fixed px) or a percentage string ("0.75%" = 0.75% of the average of the
# mask's two dimensions, clamped to [3, 30] px).
SIZE = "0.75%"
# Foreground adjustment applied BEFORE the band is dilated: "standard", "erosion", or "dilation".
METHOD = "standard"
# Iterations for that adjustment: an int, or a percentage string clamped to [1, 15].
ITERATIONS = "0%"

# Percentage clamps -- named so make_trimap() and process_batch() cannot drift apart.
SIZE_MIN_PX, SIZE_MAX_PX = 3, 30
ITER_MIN, ITER_MAX = 1, 15

_METHODS = {"standard": None, "erosion": Erosion, "dilation": Dilation}


def resolve_size(size: Union[int, str], shape) -> int:
    """Boundary size -> px. A "N%" string scales with the mask's average dimension."""
    if isinstance(size, str) and size.endswith("%"):
        try:
            percentage = float(size.rstrip("%")) / 100.0
        except ValueError:
            logger.error(f"  -> Invalid percentage format: {size}. Defaulting to 10px.")
            return 10
        h, w = shape[:2]
        avg_dim = (h + w) / 2.0
        return max(SIZE_MIN_PX, min(SIZE_MAX_PX, int(round(avg_dim * percentage))))
    return int(size)


def resolve_iterations(iterations: Union[int, str], shape) -> int:
    """Foreground-adjustment iterations -> int. A "N%" string scales with the mask's dimensions."""
    if isinstance(iterations, str) and iterations.endswith("%"):
        try:
            percentage = float(iterations.rstrip("%")) / 100.0
        except ValueError:
            logger.error(f"  -> Invalid percentage format for iterations: {iterations}. "
                         f"Defaulting to 1.")
            return 1
        h, w = shape[:2]
        avg_dim = (h + w) / 2.0
        return max(ITER_MIN, min(ITER_MAX, int(round(avg_dim * percentage))))
    return int(iterations)


def make_trimap(mask: np.ndarray, size: Union[int, str] = SIZE, method: str = METHOD,
                iterations: Union[int, str] = ITERATIONS, name: str = "mask") -> np.ndarray:
    """Binary (0/255) mask array -> trimap array (0 = bg, 127 = unknown, 255 = fg).

    The in-memory entry point used by the matting notebooks; `process_batch` is the same thing
    over a directory. `size` and `iterations` are resolved against the mask's OWN resolution, so
    resize the mask to the resolution you will run the matting network at BEFORE calling this --
    otherwise a "0.75%" band is computed on the full-size photo and then shrinks when the trimap
    is downscaled.

    Raises ValueError if the mask is not strictly binary (all-black, all-white, or grayscale)."""
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}, expected one of {sorted(_METHODS)}")

    mask = np.asarray(mask)
    if mask.ndim > 2:
        raise ValueError("mask must be single-channel")

    check_image(mask)                                    # explicit: fail before any morphology
    px = resolve_size(size, mask.shape)
    it = resolve_iterations(iterations, mask.shape)

    return trimap(
        image=mask,
        name=name,
        size=px,
        number="trimap",
        DEFG=_METHODS[method],
        num_iter=it,
        output_dir=None,                                 # in-memory only
    )


# ==========================================
# WALL-MATTING WRAPPER
# Shared by main_sd.ipynb and main_diff.ipynb so the two networks cannot be fed different trimaps.
# ==========================================
# Defaults are the settled config (2026-07-19): band the wall boundary outward only, keep objects
# in definite BACKGROUND, never in the 0.5 unknown class. Unknown = a thin buffer at true wall
# boundaries and nothing else.
FULL_RES = True          # Build the trimap at the mask's NATIVE resolution and take ONE
                         # INTER_NEAREST downsample of the finished 3-level trimap at the end.
                         # Load-bearing: `size="0.75%"` is resolution-relative, so downsampling the
                         # mask FIRST makes it resolve to a constant 8 px at 1024 -- degenerating
                         # into a fixed `dilate size=8` and throwing away the per-sample scaling
                         # (native gives 14/26/9 px on user_8/3/14). It also stops thin structure
                         # from being destroyed before the band is grown.
INNER_SIZE = 0           # px @1024 of band eaten INWARD off the wall. 0 = outward-only (upstream
                         # behaviour). Raising it makes the network re-litigate wall it was already
                         # given, and it UNDER-fills: raw-wall recall @ inner 0/4/8/16/24 =
                         # 99.8/97.3/96.2/92.8/62.1% on user_8 -- graceful to 16, cliff past it.
                         # Keep 0: eating wall into unknown violates the task framing.
FILL_HOLES = False       # True puts background fully ENCLOSED by wall (a TV, a picture frame) into
                         # the 0.5 unknown class. Off: that is an object, and objects belong in
                         # definite 0. Measured 7-14% of the band landing on objects when on.
CUT_OBJECTS = True       # Force band pixels that fall on the BiRefNet object mask back to definite
                         # 0, so the buffer stays on true wall edges instead of running over
                         # furniture standing against the wall.
CUT_DILATE = 2           # px @1024 to dilate the object mask before cutting (covers matting slop
                         # at the object's own edge).

# Px-denominated knobs above are calibrated at 1024. When FULL_RES builds at native resolution they
# are scaled by max(H, W) / REF_SIZE. `size` ("0.75%") self-scales -- do not scale it too.
REF_SIZE = 1024


def wall_trimap(mask: np.ndarray, out_size: int = REF_SIZE, size: Union[int, str] = SIZE,
                method: str = METHOD, iterations: Union[int, str] = ITERATIONS,
                full_res: bool = None, inner_size: int = None, fill_holes: bool = None,
                object_mask: np.ndarray = None, cut_dilate: int = None,
                name: str = "mask") -> np.ndarray:
    """Binary wall mask -> float trimap at (out_size, out_size): 0 = bg, 0.5 = unknown, 1 = wall.

    `make_trimap` supplies the band; everything else here is wall-specific policy. Returns float32
    with values in exactly {0, 0.5, 1}.

    Args:
        mask: binary (0/255) wall mask at any resolution.
        out_size: side length of the square output (the matting network's input resolution).
        object_mask: binary object mask (BiRefNet), same resolution as `mask`. Required by
            CUT_OBJECTS; if None, the cut is skipped.
    """
    import cv2  # local: keep the module importable without cv2 for the pure-config paths

    full_res = FULL_RES if full_res is None else full_res
    inner_size = INNER_SIZE if inner_size is None else inner_size
    fill_holes = FILL_HOLES if fill_holes is None else fill_holes
    cut_dilate = CUT_DILATE if cut_dilate is None else cut_dilate

    mask = np.asarray(mask)
    if mask.ndim > 2:
        raise ValueError("wall mask must be single-channel")

    if not full_res:
        mask = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        if object_mask is not None:
            object_mask = cv2.resize(np.asarray(object_mask), (out_size, out_size),
                                     interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    # px knobs are calibrated @REF_SIZE; at native resolution they must grow with the image
    scale = max(mask.shape[:2]) / float(REF_SIZE) if full_res else 1.0
    inner_px = int(round(inner_size * scale))
    cut_px = int(round(cut_dilate * scale))

    tri8 = make_trimap(mask, size=size, method=method, iterations=iterations, name=name)

    out = np.zeros(tri8.shape, np.float32)          # 0 / 127 / 255  ->  0 / 0.5 / 1
    out[tri8 == 127] = 0.5
    out[tri8 == 255] = 1.0

    fg = mask > 127

    if inner_px > 0:
        # band also eats inward: only the eroded core stays definite wall
        k = np.ones((2 * inner_px + 1, 2 * inner_px + 1), np.uint8)
        core = cv2.erode(fg.astype(np.uint8), k, iterations=1).astype(bool)
        out[fg & ~core] = 0.5

    if fill_holes:
        from scipy.ndimage import binary_fill_holes
        holes = binary_fill_holes(fg) & ~fg
        out[holes] = 0.5

    if object_mask is not None:
        obj = np.asarray(object_mask) > 127
        if cut_px > 0:
            k = np.ones((2 * cut_px + 1, 2 * cut_px + 1), np.uint8)
            obj = cv2.dilate(obj.astype(np.uint8), k, iterations=1).astype(bool)
        out[(out == 0.5) & obj] = 0.0               # band on furniture -> definite background

    if out.shape[:2] != (out_size, out_size):
        # MUST stay INTER_NEAREST: any interpolation invents values between 0/0.5/1 and corrupts
        # the three classes.
        out = cv2.resize(out, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    levels = set(np.unique(out).tolist())
    assert levels <= {0.0, 0.5, 1.0}, f"trimap corrupted, got levels {sorted(levels)}"
    return out


def process_batch(masks_dir: str, output_dir: str, size: Union[int, str], method: str, iterations: Union[int, str]):
    inp_path = Path(masks_dir)
    out_path = Path(output_dir)

    if not inp_path.exists() or not inp_path.is_dir():
        logger.error(f"Input masks directory does not exist: {masks_dir}")
        sys.exit(1)

    out_path.mkdir(parents=True, exist_ok=True)

    # Valid image extensions
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    mask_files = [p for p in inp_path.iterdir() if p.suffix.lower() in valid_exts]

    if not mask_files:
        logger.warning(f"No mask images found in: {masks_dir}")
        return

    logger.info(f"Found {len(mask_files)} mask files in '{masks_dir}'.")
    logger.info(f"Configuration: boundary_size={size} | method={method} | iterations={iterations}")

    # Map method name to DEFG class
    if method not in _METHODS:
        logger.error(f"Unknown method: {method}. Expected one of {sorted(_METHODS)}")
        sys.exit(1)
    defg_class = _METHODS[method]

    success_count = 0
    skip_count = 0

    for idx, mask_file in enumerate(mask_files, 1):
        logger.info(f"[{idx}/{len(mask_files)}] Processing: {mask_file.name}")
        
        try:
            # 1. Read image
            mask_data = extract_image(mask_file)
            
            # 2. Validate binary constraints
            check_image(mask_data)
            
            # Resolve dynamic size / iterations (shared with make_trimap)
            h, w = mask_data.shape[:2]
            calculated_size = resolve_size(size, mask_data.shape)
            calculated_iter = resolve_iterations(iterations, mask_data.shape)
            logger.info(f"  -> size {calculated_size}px, {calculated_iter} iter(s) "
                        f"(for {w}x{h} image)")

            # 3. Generate and save trimap
            trimap(
                image=mask_data,
                name=mask_file.stem,
                size=calculated_size,
                number="trimap",
                DEFG=defg_class,
                num_iter=calculated_iter,
                output_dir=out_path,
                filename=f"{mask_file.stem}.png"
            )
            success_count += 1
            
        except ValueError as ve:
            # Catch binary check errors and log warnings to skip rather than crashing the batch
            logger.warning(f"  ✗ Skipped {mask_file.name} (Validation Error): {ve}")
            skip_count += 1
        except Exception as e:
            logger.error(f"  ✗ Failed to process {mask_file.name}: {e}")
            skip_count += 1

    print("\n" + "=" * 50)
    print("BATCH TRIMAP GENERATION COMPLETED")
    print("=" * 50)
    print(f"Total masks found:  {len(mask_files)}")
    print(f"Successfully saved: {success_count}")
    print(f"Skipped / Failed:   {skip_count}")
    print(f"Outputs saved to:   {out_path.resolve()}/")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # ==========================================
    # CONFIGURATION VARIABLES
    # Change these values directly before running the script:
    # ==========================================
    
    # MASKS_DIR = "/Users/monie/Fabby/sandbox/workspace/image_matting/SAM1"       # Folder containing your input binary SAM masks
    # OUTPUT_DIR = "/Users/monie/Fabby/sandbox/workspace/image_matting/SAM1-trimap"     # Folder where the generated trimaps will be saved
    
    MASKS_DIR = "/Users/monie/Fabby/sandbox/workspace/image_matting/SAM-Refined"       # Folder containing your input binary SAM masks
    OUTPUT_DIR = "/Users/monie/Fabby/sandbox/workspace/image_matting/SAM-Refined-trimap"     # Folder where the generated trimaps will be saved

    # SIZE / METHOD / ITERATIONS are the module-level defaults defined near the top of this file
    # (shared with make_trimap, which the notebooks call). Override them here for a one-off batch.

    # Run batch processing
    process_batch(
        masks_dir=MASKS_DIR,
        output_dir=OUTPUT_DIR,
        size=SIZE,
        method=METHOD,
        iterations=ITERATIONS
    )
