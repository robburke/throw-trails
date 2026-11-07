"""A plate made only from user-selected, genuinely empty background frames."""

import numpy as np

from .data import capture, decode, image_write


def plate(video, frames, output):
    if not frames or len(set(frames)) != len(frames) or min(frames) < 0:
        raise ValueError("Choose one or more distinct subject-absent frames")
    with capture(video) as cap:
        images = np.stack([decode(cap, n) for n in frames])
    # Lower median is an ACTUAL pixel from a donor, even for an even frame count.
    # Whole RGB triplets are selected together; channel medians can invent colours.
    brightness = images.astype(float).mean(axis=3)
    order = np.argsort(brightness, axis=0)
    selected = order[(len(images) - 1) // 2]
    yy, xx = np.indices(selected.shape)
    result = images[selected, yy, xx]
    image_write(output, result)
    return result
