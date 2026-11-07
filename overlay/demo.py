"""Deterministic synthetic footage and an end-to-end offline demonstration."""

import time
from pathlib import Path

import cv2
import numpy as np

from .alignment import align
from .correction import correct
from .data import OBSERVED, Scene, image_write, project, read_points, write_json
from .plate import plate
from .rendering import overlay, video
from .smoothing import smooth
from .tracking import track


def generate(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    width, height, fps, count = 480, 320, 24, 56
    yy, xx = np.indices((height, width))
    texture = 105 + (xx // 24 + yy // 24) % 2 * 9
    background = np.stack((texture + 24, texture + 11, texture), axis=2).astype(np.uint8)
    for i in range(1100):
        x, y = map(int, rng.integers([3, 3], [width - 3, height - 3]))
        shade = int(rng.integers(50, 195))
        cv2.circle(background, (x, y), int(rng.integers(1, 3)), (shade, shade, shade), -1)
    flights, truth = [], {}
    for i, (release, end) in enumerate(((6, 40), (8, 44), (10, 42))):
        name = f"launch-{i + 1}"
        angle = 0 if i == 0 else (0.45 if i == 1 else -0.35)
        camera = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        camera[:, 2] += [0, 0] if i == 0 else ([4, -3] if i == 1 else [-3, 2])
        h = np.vstack((camera, [0, 0, 1]))
        filename = name + ".avi"
        writer = cv2.VideoWriter(str(folder / filename), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
        if not writer.isOpened():
            raise ValueError("Demo requires the OpenCV MJPG video writer")
        expected = []
        try:
            for n in range(count):
                image = background.copy()
                visible = release <= n <= end
                hidden = i == 1 and 24 <= n <= 28
                point = None
                if visible:
                    u = (n - release) / (end - release)
                    xy = np.array([128 + 224 * u + 3 * i,
                                   235 - (112 + 13 * i) * 4 * u * (1 - u) - 12 * i * u])
                    point = np.rint(project([xy], h)[0]).tolist()
                    cv2.circle(image, tuple(np.rint(xy).astype(int)), 7, (28, 35, 235), -1, cv2.LINE_AA)
                    cv2.line(image, tuple(np.rint(xy + [-3, -2]).astype(int)),
                             tuple(np.rint(xy + [3, 2]).astype(int)), (50, 65, 250), 2)
                if hidden:
                    # Actual foreground occluder, not a missing annotation trick.
                    cv2.rectangle(image, (215, 65), (270, 270), (92, 95, 98), -1)
                image = cv2.warpPerspective(image, h, (width, height), borderMode=cv2.BORDER_REFLECT)
                writer.write(image)
                expected.append({"frame": n, "status": "detected" if visible and not hidden else "occluded",
                                 "xy": point if visible and not hidden else None})
        finally:
            writer.release()
        flights.append({"id": name, "video": filename, "points": name + ".csv",
                        "release": release, "end": end,
                        "alignment": {"frame": 0, "exclude": [[15, 55, 465, 275]]},
                        "colour": {"hues": [[0, 12], [168, 179]], "saturation": 100, "value": 70},
                        "cutout_radius": 18})
        truth[name] = expected
    # Make the declared end later than the last visible observation on launch 3.
    flights[2]["end"] += 2
    scene = {"reference": {"video": flights[0]["video"], "frame": 0,
                            "exclude": [[15, 55, 465, 275]]},
             "base": "plate.png", "flights": flights}
    write_json(folder / "scene.json", scene)
    write_json(folder / "expected.json", {"fps": fps, "size": [width, height], "flights": truth,
                                           "occlusion": {"id": "launch-2", "frames": [24, 25, 26, 27, 28]},
                                           "invariants": ["No coordinates for occluded frames", "No interval crosses a gap", "Last trail centre equals the last observation"]})
    return folder / "scene.json"


def demo(output="demo-output"):
    started = time.perf_counter()
    folder = Path(output)
    # Never recursively delete an existing directory. Files owned by the demo are replaced.
    scene_path = generate(folder)
    scene = Scene(scene_path)
    for flight in scene.flights:
        track(scene.file(flight["video"]), scene.file(flight["points"]),
              start=flight["release"], end=flight["end"], seed=(128, 235), window=50,
              settings=flight["colour"], min_area=25, max_area=400)
    # Exercise the exact editor model without opening a window. The synthetic
    # generator supplies a known centre for a genuinely visible frame.
    first = scene.flights[0]
    edit_frame = first["release"]
    edits_path = folder / "edits.json"
    write_json(edits_path, [{"frame": edit_frame, "x": 128.0, "y": 235.0}])
    correct(scene.file(first["video"]), scene.file(first["points"]), edits=edits_path)
    plate(scene.file(first["video"]), [0, 1, 2], scene.file("plate.png"))
    alignments = folder / "alignment.json"
    smoothed = folder / "smoothed.json"
    align(scene_path, alignments)
    smooth(scene_path, alignments, smoothed, folder / "residuals.png")
    overlay(scene_path, alignments, smoothed, folder / "overlay.png", every=7)
    overlay(scene_path, alignments, smoothed, folder / "portrait.png", every=7, crop=True)
    video(scene_path, alignments, smoothed, folder / "common.avi", mode="common", every=7)
    video(scene_path, alignments, smoothed, folder / "staggered.avi", mode="staggered", stagger=0.7, every=7)
    statuses = {}
    for flight in scene.flights:
        rows = read_points(scene.file(flight["points"]))
        statuses[flight["id"]] = {s: sum(r.status == s for r in rows) for s in ("detected", "corrected", "occluded")}
        if not any(r.status in OBSERVED for r in rows):
            raise AssertionError("Demo produced no detections")
    result = {"elapsed_s": round(time.perf_counter() - started, 3), "statuses": statuses,
              "image": "overlay.png", "synthetic_only": True}
    write_json(folder / "result.json", result)
    print(f"Synthetic demo: {result['elapsed_s']:.3f}s; image: {folder / 'overlay.png'}")
    return result
