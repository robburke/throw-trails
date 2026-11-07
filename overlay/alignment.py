"""One masked background homography per clip, never per-trajectory shifts."""

import cv2
import numpy as np

from .data import Scene, digest, frame, image_read, project, write_json


def static_mask(scene, spec, shape):
    if spec.get("mask"):
        mask = image_read(scene.file(spec["mask"]), gray=True)
        if mask.shape != shape:
            raise ValueError("Alignment mask must have the source frame dimensions")
        mask = (mask > 0).astype(np.uint8) * 255
    else:
        mask = np.full(shape, 255, np.uint8)
    for x0, y0, x1, y1 in spec.get("exclude", []):
        if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
            raise ValueError("Excluded moving region must lie inside its frame")
        mask[y0:y1, x0:x1] = 0
    if not spec.get("mask") and not spec.get("exclude"):
        raise ValueError("Supply a static-feature mask or excluded moving-region boxes")
    return mask


def estimate(source, reference, source_mask, reference_mask):
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=8)
    sk, sd = orb.detectAndCompute(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), source_mask)
    rk, rd = orb.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), reference_mask)
    if sd is None or rd is None:
        raise ValueError("Too few static features; add background texture or improve masks")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(sd, rd, k=2)
    matches = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
    if len(matches) < 8:
        raise ValueError("Too few background matches for a trustworthy homography")
    src = np.float64([sk[m.queryIdx].pt for m in matches])
    dst = np.float64([rk[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(42)
    h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if h is None or inliers is None:
        raise ValueError("Homography fit failed")
    keep = inliers.ravel().astype(bool)
    residual = np.linalg.norm(project(src, h) - dst, axis=1)[keep]
    if keep.sum() < 8 or keep.mean() < 0.35 or np.percentile(residual, 95) > 3:
        raise ValueError("Background alignment failed quality checks")
    for pts, mask in ((src[keep], source_mask), (dst[keep], reference_mask)):
        hull_area = cv2.contourArea(cv2.convexHull(pts.astype(np.float32)))
        if hull_area < 0.03 * np.count_nonzero(mask):
            raise ValueError("Background matches are too concentrated; use spread-out fixed features")
    corners = np.float64([[0, 0], [source.shape[1] - 1, 0],
                          [source.shape[1] - 1, source.shape[0] - 1], [0, source.shape[0] - 1]])
    warped = project(corners, h)
    if not cv2.isContourConvex(warped.astype(np.float32)):
        raise ValueError("Invalid homography geometry")
    area_ratio = abs(cv2.contourArea(warped.astype(np.float32))) / (source.shape[0] * source.shape[1])
    if not 0.25 < area_ratio < 4:
        raise ValueError("Implausible fixed-camera scale change")
    return h / h[2, 2], {"matches": len(matches), "inliers": int(keep.sum()),
                         "median_px": float(np.median(residual)),
                         "p95_px": float(np.percentile(residual, 95))}


def align(scene_path, output):
    scene = Scene(scene_path)
    reference_spec = scene.config["reference"]
    reference = frame(scene.file(reference_spec["video"]), reference_spec["frame"])
    reference_mask = static_mask(scene, reference_spec, reference.shape[:2])
    clips = {}
    for flight in scene.flights:
        video = flight["video"]
        spec = flight.get("alignment", {})
        sample = int(spec.get("frame", flight["release"]))
        if video in clips:
            if clips[video]["alignment_spec"] != spec:
                raise ValueError("All flights in a clip must share one alignment specification")
            continue
        source = frame(scene.file(video), sample)
        source_mask = static_mask(scene, spec, source.shape[:2])
        if video == reference_spec["video"] and sample == reference_spec["frame"]:
            h, stats = np.eye(3), {"method": "reference identity"}
        else:
            h, stats = estimate(source, reference, source_mask, reference_mask)
        clips[video] = {"H": h.tolist(), "frame": sample, "alignment_spec": spec, **stats}
    record = {"scene_sha256": digest(scene.path), "size": list(reference.shape[1::-1]), "clips": clips}
    write_json(output, record)
    return record
