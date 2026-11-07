"""Observed-only persistent trails and actual source-pixel cutouts."""

import colorsys
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .data import (OBSERVED, Scene, capture, decode, frame, image_read,
                   image_write, metadata, project, read_json, transforms, write_json)
from .tracking import colour_mask


def palette(n):
    """N evenly spaced colours, returned in OpenCV BGR order."""
    return [tuple(round(255 * v) for v in colorsys.hsv_to_rgb((0.48 + i / n) % 1, 0.68, 1)[::-1])
            for i in range(n)]


def trail_intervals(groups, through):
    """Only complete adjacent observed intervals; pin EVERY playback cutoff.

    An unfinished run uses its raw chords. This prevents a full-run spline's
    intermediate residual from moving the current endpoint ahead of reality.
    Completed runs may use the bounded cubic; all their endpoints are pinned.
    """
    result = []
    for group in groups:
        for i, interval in enumerate(group["intervals"]):
            if interval["to"] > through:
                break
            if through < group["frames"][-1]:
                result.append({"from": interval["from"], "to": interval["to"],
                               "xy": [group["raw_xy"][i], group["raw_xy"][i + 1]]})
            else:
                result.append(interval)
    return result


def paint_trail(image, intervals, colour, first, last, width=5.0):
    for interval in intervals:
        xy = np.asarray(interval["xy"], dtype=float)
        count = len(xy) - 1
        for j, (a, b) in enumerate(zip(xy, xy[1:])):
            delta = b - a
            length = np.linalg.norm(delta)
            if length < 1e-8:
                continue
            normal = np.array([-delta[1], delta[0]]) / length
            fa = interval["from"] + (interval["to"] - interval["from"]) * j / count
            fb = interval["from"] + (interval["to"] - interval["from"]) * (j + 1) / count
            wa = (0.25 + 0.75 * (fa - first) / max(1, last - first)) * width / 2
            wb = (0.25 + 0.75 * (fb - first) / max(1, last - first)) * width / 2
            # Butt caps: the trail's centreline stops exactly at the measured end.
            polygon = np.rint([a + normal * wa, b + normal * wb,
                               b - normal * wb, a - normal * wa]).astype(np.int32)
            cv2.fillConvexPoly(image, polygon, colour, cv2.LINE_AA)


def dashed_ring(image, xy, colour, radius=11):
    centre = tuple(np.rint(xy).astype(int))
    for angle in range(0, 360, 45):
        cv2.ellipse(image, centre, (radius, radius), 0, angle, angle + 24, colour, 1, cv2.LINE_AA)


def crop_4_5(image, centre_x=None):
    height, width = image.shape[:2]
    target_width, target_height = min(width, int(height * 4 / 5)), height
    # Exact integer 4:5 output, with no stretch or invented border.
    units = min(target_width // 4, target_height // 5)
    if units < 1:
        raise ValueError("Canvas is too small for a 4:5 crop")
    target_width, target_height = units * 4, units * 5
    cx = width / 2 if centre_x is None else centre_x
    left = max(0, min(width - target_width, round(cx - target_width / 2)))
    top = (height - target_height) // 2
    return image[top:top + target_height, left:left + target_width].copy()


@dataclass
class Patch:
    x: int
    y: int
    rgb: np.ndarray  # BGR, already multiplied by alpha
    alpha: np.ndarray


def cutout(source, row, h, size, settings, radius=24, matte=None):
    x0, y0 = max(0, math.floor(row.x - radius)), max(0, math.floor(row.y - radius))
    x1, y1 = min(source.shape[1], math.ceil(row.x + radius + 1)), min(source.shape[0], math.ceil(row.y + radius + 1))
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Measured point is outside its source frame")
    crop = source[y0:y1, x0:x1]
    if matte is not None:
        if matte.shape != source.shape[:2]:
            raise ValueError("Cutout matte must match the decoded source size")
        alpha = matte[y0:y1, x0:x1].astype(np.float32) / 255
    else:
        mask = colour_mask(crop, settings)
        count, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
        candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 3]
        if not candidates:
            raise ValueError("No real cutout at observed frame " + str(row.frame) + "; correct the point, colour settings, or supply a matte")
        centre = np.array([row.x - x0, row.y - y0])
        chosen = min(candidates, key=lambda i: np.linalg.norm(centres[i] - centre))
        if np.linalg.norm(centres[chosen] - centre) > radius / 2:
            raise ValueError("Cutout component is too far from the measured centre")
        # Do not fill holes or reconstruct hidden portions of the object.
        alpha = (labels == chosen).astype(np.float32)
    if not alpha.any():
        raise ValueError("Empty cutout matte at observed frame " + str(row.frame))
    local = np.asarray(h) @ np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=float)
    corners = project([[0, 0], [x1 - x0, 0], [x1 - x0, y1 - y0], [0, y1 - y0]], local)
    lo = np.maximum(0, np.floor(corners.min(axis=0)).astype(int))
    hi = np.minimum(size, np.ceil(corners.max(axis=0)).astype(int))
    if np.any(hi <= lo):
        return None
    local = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]]) @ local
    dimensions = tuple(map(int, hi - lo))
    alpha_warp = cv2.warpPerspective(alpha, local, dimensions)
    colour_warp = cv2.warpPerspective(crop.astype(np.float32) * alpha[:, :, None], local, dimensions)
    return Patch(int(lo[0]), int(lo[1]), colour_warp, alpha_warp)


def composite(image, patch):
    if patch is None:
        return
    height, width = patch.alpha.shape
    region = image[patch.y:patch.y + height, patch.x:patch.x + width]
    region[:] = np.rint(region * (1 - patch.alpha[:, :, None]) + patch.rgb).clip(0, 255).astype(np.uint8)


class Renderer:
    def __init__(self, scene_path, alignment_path, smooth_path):
        self.scene = Scene(scene_path)
        self.alignment = transforms(self.scene, alignment_path)
        self.smoothing = read_json(smooth_path)
        if self.smoothing["signature"] != self.scene.signature(self.alignment):
            raise ValueError("Measurements or alignment changed: rerun smooth")
        cfg = self.scene.config
        ref = cfg["reference"]
        self.base = image_read(self.scene.file(cfg["base"])) if cfg.get("base") else frame(self.scene.file(ref["video"]), ref["frame"])
        self.size = tuple(self.alignment["size"])
        if self.base.shape[1::-1] != self.size:
            raise ValueError("Base must have the reference frame dimensions")
        self.colours = palette(len(self.scene.flights))
        self.rows = {}
        self.patches = {}
        self.fps = {}
        for flight in self.scene.flights:
            name = flight["id"]
            fps, count, source_size = metadata(self.scene.file(flight["video"]))
            self.fps[name] = fps
            rows = self.scene.rows(flight)
            if flight["end"] >= count or any(r.x is not None and not (0 <= r.x < source_size[0] and 0 <= r.y < source_size[1]) for r in rows):
                raise ValueError("Flight range or point lies outside its source video")
            if any(abs(r.time - r.frame / fps) > 0.5 / fps for r in rows):
                raise ValueError("CSV times disagree with video frame rate")
            self.rows[name] = rows
            h = self.alignment["clips"][flight["video"]]["H"]
            self.patches[name] = {}
            with capture(self.scene.file(flight["video"])) as cap:
                for row in rows:
                    if row.status not in OBSERVED:
                        continue
                    source = decode(cap, row.frame)
                    matte_name = flight.get("mattes", {}).get(str(row.frame))
                    matte = image_read(self.scene.file(matte_name), gray=True) if matte_name else None
                    self.patches[name][row.frame] = cutout(source, row, h, self.size,
                                                         flight.get("colour", {}),
                                                         flight.get("cutout_radius", 24), matte)

    def draw(self, through=None, every=6, crop=False, centre_x=None):
        if every < 1:
            raise ValueError("Cutout spacing must be at least one frame")
        image = self.base.copy()
        for flight, colour in zip(self.scene.flights, self.colours):
            name = flight["id"]
            limit = flight["end"] if through is None else through[name]
            rows = self.rows[name]
            seen = [r for r in rows if r.status in OBSERVED and r.frame <= limit]
            if not seen:
                continue
            groups = self.smoothing["flights"][name]
            intervals = trail_intervals(groups, limit)
            paint_trail(image, intervals, colour, flight["release"], flight["end"])
            chosen = [r for r in seen if (r.frame - flight["release"]) % every == 0]
            if seen[-1] not in chosen:
                chosen.append(seen[-1])
            for row in chosen:
                composite(image, self.patches[name][row.frame])
            # Rings are at known boundary observations, NEVER at guessed hidden ends.
            h = self.alignment["clips"][flight["video"]]["H"]
            observed_frames = {r.frame for r in rows if r.status in OBSERVED}
            for row in seen:
                missing_before = row.frame == seen[0].frame and row.frame > flight["release"]
                missing_after = row.frame + 1 <= min(limit, flight["end"]) and row.frame + 1 not in observed_frames
                if missing_before or missing_after:
                    dashed_ring(image, project([[row.x, row.y]], h)[0], colour)
        return crop_4_5(image, centre_x) if crop else image


def overlay(scene, alignment, smoothing, output, every=6, crop=False, centre_x=None):
    renderer = Renderer(scene, alignment, smoothing)
    result = renderer.draw(every=every, crop=crop, centre_x=centre_x)
    image_write(output, result)
    return result


def schedule(flights, fps_by_id, mode="common", duration=None, stagger=0.7):
    """Release frames anchor timing; end is the declared finish or unseen cutoff."""
    if mode not in {"common", "staggered"} or stagger < 0 or (duration is not None and duration <= 0):
        raise ValueError("Invalid playback timing")
    native = [(f["end"] - f["release"]) / fps_by_id[f["id"]] for f in flights]
    common = max(native) if duration is None else duration
    return {f["id"]: {"start_s": 0.35 + (i * stagger if mode == "staggered" else 0),
                        "duration_s": common if mode == "common" else length,
                        "release": f["release"], "end": f["end"],
                        "speed": length / common if mode == "common" else 1.0}
            for i, (f, length) in enumerate(zip(flights, native))}


def source_frame_at(t, item):
    if t < item["start_s"]:
        return item["release"] - 1
    fraction = min(1.0, (t - item["start_s"]) / item["duration_s"])
    return item["release"] + math.floor(fraction * (item["end"] - item["release"]) + 1e-9)


def video(scene, alignment, smoothing, output, mode="common", duration=None,
          stagger=0.7, fps=24, every=6, crop=False, centre_x=None):
    if fps <= 0:
        raise ValueError("Output FPS must be positive")
    renderer = Renderer(scene, alignment, smoothing)
    timing = schedule(renderer.scene.flights, renderer.fps, mode, duration, stagger)
    last = max(item["start_s"] + item["duration_s"] for item in timing.values()) + 0.6
    sample = renderer.draw({f["id"]: -1 for f in renderer.scene.flights}, every, crop, centre_x)
    size = sample.shape[1::-1]
    if any(n % 2 for n in size):
        raise ValueError("Video requires even image dimensions; choose an even-sized base")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    codec = "MJPG" if Path(output).suffix.lower() == ".avi" else "mp4v"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, size)
    if not writer.isOpened():
        raise ValueError("Cannot open video writer; try an .avi output")
    try:
        for index in range(math.ceil(last * fps) + 1):
            through = {name: source_frame_at(index / fps, item) for name, item in timing.items()}
            writer.write(renderer.draw(through, every, crop, centre_x))
    finally:
        writer.release()
    write_json(Path(output).with_suffix(".timing.json"),
               {"mode": mode, "fps": fps, "flights": timing, "audio": "none",
                "note": "Separate launches; common mode changes each flight speed. Unseen end times are cutoffs, not claimed landings."})
    return timing
