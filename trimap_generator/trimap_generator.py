#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Trimap generation by dilation -- vendored from lnugraha/trimap_generator (`trimap_class.py`).

`generate_trimap.py` imports `extract_image`, `check_image`, `trimap`, `Erosion`, `Dilation` from
here. The upstream module is kept faithful with four deliberate changes, all marked LOCAL below:

1. **snake_case aliases** (`extract_image` / `check_image`) next to the upstream camelCase names.
2. **`check_image` raises `ValueError`** instead of `print` + `sys.exit()`. Upstream kills the
   interpreter on an all-black mask -- fine for a CLI, fatal in a notebook (it tears down the
   kernel). `generate_trimap.py`'s batch loop already catches `ValueError` to skip a bad mask.
3. **`trimap()` takes `output_dir` / `filename` and RETURNS the array.** Upstream hardcodes
   `./images/results/` and returns None, so it cannot be used as a library function.
4. **The final 3-level clamp is vectorised.** Upstream does a per-pixel Python double loop; at
   1024x1024 that is ~1M iterations per call. `np.where` is identical in effect -- it maps every
   value that is not 0 and not 255 to 127.

Algorithm (unchanged): optionally erode/dilate the foreground by `num_iter` with a 3x3 kernel,
then dilate by a `(2*size+1)` square kernel. The dilated ring becomes 127 (unknown) and the
original foreground stays 255. **The band grows OUTWARD only** -- the foreground is never eaten,
unlike a symmetric erode-both-sides trimap.
"""
import os
import sys
from abc import ABC, abstractmethod

import cv2
import numpy as np

__all__ = ["extractImage", "extract_image", "checkImage", "check_image",
           "trimap", "FGScale", "Erosion", "Dilation", "Toolbox"]


def extractImage(path):
    """Read an image off disk as single-channel grayscale."""
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def checkImage(image):
    """Verify the input really is a binary (0/255) single-channel mask.

    LOCAL: raises ValueError rather than sys.exit() so a bad mask skips instead of killing the
    process/kernel."""
    if image is None:
        raise ValueError("image could not be read")
    if len(image.shape) > 2:
        raise ValueError("non-binary image (RGB)")

    smallest = image.min(axis=0).min(axis=0)   # lowest pixel value; should be 0 (black)
    largest = image.max(axis=0).max(axis=0)    # highest pixel value; should be 255 (white)

    if smallest == 0 and largest == 0:
        raise ValueError("non-binary image (all black)")
    elif smallest == 255 and largest == 255:
        raise ValueError("non-binary image (all white)")
    elif smallest > 0 or largest < 255:
        raise ValueError("non-binary image (grayscale)")
    return True


# LOCAL: snake_case aliases -- the names generate_trimap.py imports.
extract_image = extractImage
check_image = checkImage


class FGScale(ABC):
    """Abstract base for foreground erosion/dilation applied BEFORE the trimap dilation."""

    def __init__(self, image):
        self.image = image

    @abstractmethod
    def scaling(self, image, iteration):
        pass


class Erosion(FGScale):
    """Shrink the foreground by `erosion` iterations of a 3x3 kernel."""

    def scaling(self, image, erosion):
        erosion = int(erosion)
        kernel = np.ones((3, 3), np.uint8)
        image = cv2.erode(image, kernel, iterations=erosion)
        image = np.where(image > 0, 255, image)          # any gray pixel becomes white (smoothing)
        if cv2.countNonZero(image) == 0:
            raise ValueError("foreground has been entirely eroded")
        return image


class Dilation(FGScale):
    """Grow the foreground by `dilation` iterations of a 3x3 kernel."""

    def scaling(self, image, dilation):
        dilation = int(dilation)
        kernel = np.ones((3, 3), np.uint8)
        image = cv2.dilate(image, kernel, iterations=dilation)
        image = np.where(image > 0, 255, image)
        if np.sum(image == 255) == image.shape[0] * image.shape[1]:
            raise ValueError("foreground has been entirely expanded")
        return image


class Toolbox:
    """Upstream helper kept for parity; unused by generate_trimap.py."""

    def __init__(self, image):
        self.image = image

    def saveImage(self, title, extension):
        cv2.imwrite("{}.{}".format(title, extension), self.image)

    def morph_open(self, image, kernel):
        """Remove white speckles outside the mask."""
        return cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)

    def morph_close(self, image, kernel):
        """Remove black speckles inside the mask."""
        return cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)


def trimap(image, name, size, number, DEFG=None, num_iter=0, output_dir=None, filename=None):
    """Binary mask -> trimap (0 = background, 127 = unknown, 255 = foreground).

    Args:
        image: binary (0/255) single-channel mask.
        name, number: used only to build the upstream-style filename when `filename` is None.
        size: half-width of the dilation kernel in px -- the unknown band grows `size` px OUTWARD.
        DEFG: None, `Erosion`, or `Dilation` -- adjust the foreground before dilating.
        num_iter: iterations for DEFG.
        output_dir, filename: LOCAL. Where to write. Pass `output_dir=None` to skip writing.

    Returns:
        LOCAL: the trimap as a uint8 array (upstream returned None).
    """
    checkImage(image)

    pixels = 2 * size + 1                              # odd-sized kernel
    kernel = np.ones((pixels, pixels), np.uint8)

    if DEFG is None:
        pass
    elif DEFG is Dilation:
        image = Dilation(image).scaling(image, num_iter)
    elif DEFG is Erosion:
        image = Erosion(image).scaling(image, num_iter)
    else:
        raise ValueError("unspecified foreground dilation or erosion method")

    dilation = cv2.dilate(image, kernel, iterations=1)

    dilation = np.where(dilation == 255, 127, dilation)   # WHITE to GRAY
    remake = np.where(dilation != 127, 0, dilation)       # smoothing
    remake = np.where(image > 127, 200, dilation)         # mark the foreground inside the GRAY

    remake = np.where(remake < 127, 0, remake)
    remake = np.where(remake > 200, 0, remake)
    remake = np.where(remake == 200, 255, remake)

    # LOCAL: vectorised replacement for upstream's per-pixel double loop. Same effect --
    # anything that is neither 0 nor 255 is unknown.
    remake = np.where((remake != 0) & (remake != 255), 127, remake).astype(np.uint8)

    if output_dir is not None:
        new_name = filename or "{}px_{}_{}.png".format(size, name, number)
        os.makedirs(str(output_dir), exist_ok=True)
        cv2.imwrite(os.path.join(str(output_dir), new_name), remake)

    return remake
