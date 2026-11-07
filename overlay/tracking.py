"""Colour proposals, with a predictive search gate but no predicted output points."""

import cv2
import numpy as np

from .data import Point, capture, metadata, write_points


def colour_mask(image, settings=None):
    settings = settings or {}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    result = np.zeros(image.shape[:2], np.uint8)
    for lo, hi in settings.get("hues", [[0, 12], [168, 179]]):
        result |= cv2.inRange(hsv, (int(lo), int(settings.get("saturation", 100)),
                                   int(settings.get("value", 70))), (int(hi), 255, 255))
    return result


def track(video, output, start=0, end=None, seed=None, window=70, settings=None,
          min_area=8, max_area=2500, roi=None):
    fps, count, size = metadata(video)
    end = count - 1 if end is None else end
    if not 0 <= start <= end < count or window <= 0 or min_area <= 0 or max_area < min_area:
        raise ValueError("Invalid tracking range, window or area limits")
    if roi and not (0 <= roi[0] < roi[2] <= size[0] and 0 <= roi[1] < roi[3] <= size[1]):
        raise ValueError("ROI must lie within the decoded frame")
    predicted = np.array(seed, dtype=float) if seed is not None else None
    velocity = np.zeros(2)
    last = None
    last_frame = None
    rows = []
    with capture(video) as cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for index in range(start, end + 1):
            ok, image = cap.read()
            if not ok:
                raise ValueError("Video ended inside tracking window")
            mask = colour_mask(image, settings)
            if roi:
                keep = np.zeros_like(mask)
                x0, y0, x1, y1 = roi
                keep[y0:y1, x0:x1] = 255
                mask &= keep
            if predicted is not None:
                yy, xx = np.indices(mask.shape)
                mask[(xx - predicted[0]) ** 2 + (yy - predicted[1]) ** 2 > window ** 2] = 0
            count_components, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
            candidates = [i for i in range(1, count_components)
                          if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]
            if candidates:
                selected = min(candidates, key=lambda i: np.linalg.norm(centres[i] - predicted)) if predicted is not None else max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
                point = centres[selected]
                rows.append(Point(index, index / fps, *map(float, point), "detected"))
                if last is not None:
                    velocity = (point - last) / (index - last_frame)
                last, last_frame = point.copy(), index
                predicted = point + velocity
            else:
                rows.append(Point(index, index / fps, None, None, "occluded"))
                # Prediction controls only the next search window. It is NEVER exported.
                if predicted is not None:
                    predicted += velocity
    write_points(output, rows)
    return rows
