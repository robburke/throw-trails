"""Small explicit file formats; coordinates are zero-based decoded pixels."""

import csv
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

OBSERVED = frozenset({"detected", "corrected"})
STATUSES = OBSERVED | {"occluded"}
FIELDS = ("frame", "time", "x", "y", "status")


@dataclass(frozen=True)
class Point:
    frame: int
    time: float
    x: float | None
    y: float | None
    status: str

    def __post_init__(self):
        if self.frame < 0 or not math.isfinite(self.time) or self.time < 0:
            raise ValueError("Invalid frame or time")
        if self.status not in STATUSES:
            raise ValueError("Unknown point status")
        if self.status == "occluded":
            if self.x is not None or self.y is not None:
                raise ValueError("Occluded rows must have blank coordinates")
        elif self.x is None or self.y is None or not all(map(math.isfinite, (self.x, self.y))):
            raise ValueError("Observed rows need finite coordinates")


def read_points(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("CSV columns must be frame,time,x,y,status")
        rows = [Point(int(r["frame"]), float(r["time"]),
                      float(r["x"]) if r["x"] else None,
                      float(r["y"]) if r["y"] else None, r["status"]) for r in reader]
    if not rows or any(b.frame <= a.frame or b.time <= a.time for a, b in zip(rows, rows[1:])):
        raise ValueError("CSV must contain increasing frames and times")
    return rows


def write_points(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + chr(10), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def capture(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open video: " + Path(path).name)
        yield cap
    finally:
        cap.release()


def metadata(path):
    with capture(path) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if not math.isfinite(fps) or fps <= 0 or count < 1 or min(size) < 1:
        raise ValueError("Video has invalid metadata")
    return fps, count, size


def decode(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = cap.read()
    if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
        raise ValueError("Cannot decode requested frame " + str(index))
    return image


def frame(path, index):
    with capture(path) as cap:
        return decode(cap, index)


def image_read(path, gray=False):
    result = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError("Cannot read image: " + Path(path).name)
    return result


def image_write(path, image):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ValueError("Cannot write image")


def project(xy, transform):
    xy = np.asarray(xy, dtype=float).reshape(-1, 1, 2)
    result = cv2.perspectiveTransform(xy, np.asarray(transform, dtype=float)).reshape(-1, 2)
    if not np.isfinite(result).all():
        raise ValueError("Invalid projected coordinates")
    return result


def runs(rows):
    """Split on ANY missing row, frame jump, or occlusion. Never bridge gaps."""
    group = []
    for row in rows:
        if row.status not in OBSERVED or (group and row.frame != group[-1].frame + 1):
            if group:
                yield group
            group = []
        if row.status in OBSERVED:
            group.append(row)
    if group:
        yield group


class Scene:
    def __init__(self, path):
        self.path = Path(path)
        self.root = self.path.parent
        self.config = read_json(path)
        self.flights = self.config["flights"]
        if not self.flights or len({f["id"] for f in self.flights}) != len(self.flights):
            raise ValueError("Scene needs flights with unique ids")
        for flight in self.flights:
            if flight["release"] < 0 or flight["end"] <= flight["release"]:
                raise ValueError("Flight end must follow release")

    def file(self, name):
        return self.root / name

    def rows(self, flight):
        rows = read_points(self.file(flight["points"]))
        rows = [r for r in rows if flight["release"] <= r.frame <= flight["end"]]
        if not any(r.status in OBSERVED for r in rows):
            raise ValueError("Flight has no observations: " + flight["id"])
        return rows

    def signature(self, alignments):
        """Detect stale geometry after correction or alignment changes."""
        payload = {"scene": self.config, "alignment": alignments,
                   "points": {f["id"]: digest(self.file(f["points"])) for f in self.flights}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def transforms(scene, path):
    record = read_json(path)
    if record["scene_sha256"] != digest(scene.path):
        raise ValueError("Scene changed: rerun align and smooth")
    return record
