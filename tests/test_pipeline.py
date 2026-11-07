"""Offline integration and adversarial checks. Run: python -B -m unittest discover -s tests -v"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from overlay.correction import Edits, correct
from overlay.data import OBSERVED, Point, Scene, image_read, project, read_points, runs, write_points
from overlay.rendering import Renderer, palette, schedule, source_frame_at, trail_intervals
from overlay.smoothing import fit_run

ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = ROOT / "test-output"
        parent.mkdir(exist_ok=True)
        cls.folder = Path(tempfile.mkdtemp(prefix="pipeline-", dir=parent))
        result = subprocess.run([sys.executable, "-B", "-m", "overlay", "demo", "--output", str(cls.folder)],
                                cwd=ROOT, capture_output=True, text=True, timeout=55)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.scene = Scene(cls.folder / "scene.json")
        cls.smoothed = json.loads((cls.folder / "smoothed.json").read_text())
        cls.alignment = json.loads((cls.folder / "alignment.json").read_text())

    @classmethod
    def tearDownClass(cls):
        # Only this test's resolved child directory is ever deleted.
        if cls.folder.resolve().is_relative_to((ROOT / "test-output").resolve()):
            shutil.rmtree(cls.folder)

    def test_demo_statuses_and_synthetic_truth(self):
        result = json.loads((self.folder / "result.json").read_text())
        self.assertLess(result["elapsed_s"], 60)
        truth = json.loads((self.folder / "expected.json").read_text())
        statuses = set()
        for flight in self.scene.flights:
            expected = {r["frame"]: r for r in truth["flights"][flight["id"]]}
            for row in self.scene.rows(flight):
                statuses.add(row.status)
                if row.status == "occluded":
                    self.assertIsNone(row.x)
                    self.assertIsNone(row.y)
                    self.assertEqual(expected[row.frame]["status"], "occluded")
                else:
                    self.assertEqual(expected[row.frame]["status"], "detected")
                    self.assertLess(np.linalg.norm(np.array([row.x, row.y]) - expected[row.frame]["xy"]), 2)
        self.assertEqual(statuses, {"detected", "corrected", "occluded"})
        second = self.scene.rows(self.scene.flights[1])
        self.assertEqual([r.frame for r in second if r.status == "occluded"], [24, 25, 26, 27, 28])

    def test_no_interpolation_and_trails_stop_at_observation(self):
        for flight in self.scene.flights:
            observed = {r.frame: r for r in self.scene.rows(flight) if r.status in OBSERVED}
            groups = self.smoothed["flights"][flight["id"]]
            h = self.alignment["clips"][flight["video"]]["H"]
            for group in groups:
                np.testing.assert_array_equal(np.asarray(group["raw_xy"])[[0, -1]],
                                              np.asarray(group["fitted_xy"])[[0, -1]])
                self.assertLessEqual(max(group["residuals_px"]), group["budget_px"] + 1e-8)
                for interval in group["intervals"]:
                    self.assertIn(interval["from"], observed)
                    self.assertIn(interval["to"], observed)
                    self.assertEqual(interval["to"], interval["from"] + 1)
            for limit in range(flight["release"] - 1, flight["end"] + 5):
                intervals = trail_intervals(groups, limit)
                self.assertTrue(all(i["to"] <= limit for i in intervals))
                if intervals:
                    last = intervals[-1]
                    expected = project([[observed[last["to"]].x, observed[last["to"]].y]], h)[0]
                    np.testing.assert_array_equal(last["xy"][-1], expected)
            last = trail_intervals(groups, flight["end"] + 10)[-1]
            self.assertEqual(last["to"], max(observed))

    def test_output_dimensions_and_video_decodes(self):
        portrait = image_read(self.folder / "portrait.png")
        self.assertEqual(portrait.shape[1] * 5, portrait.shape[0] * 4)
        self.assertEqual(image_read(self.folder / "overlay.png").shape, (320, 480, 3))
        for name in ("common.avi", "staggered.avi"):
            cap = cv2.VideoCapture(str(self.folder / name))
            count = 0
            try:
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    self.assertEqual(image.shape, (320, 480, 3))
                    count += 1
            finally:
                cap.release()
            self.assertGreater(count, 40)

    def test_signature_rejects_stale_corrections(self):
        path = self.scene.file(self.scene.flights[0]["points"])
        old = path.read_bytes()
        try:
            rows = read_points(path)
            changed = rows[1]
            rows[1] = Point(changed.frame, changed.time, changed.x + 0.5, changed.y, "corrected")
            write_points(path, rows)
            with self.assertRaisesRegex(ValueError, "rerun smooth"):
                Renderer(self.folder / "scene.json", self.folder / "alignment.json", self.folder / "smoothed.json")
        finally:
            path.write_bytes(old)


class GeometryTest(unittest.TestCase):
    def test_missing_rows_and_occlusions_split_runs(self):
        rows = [Point(0, 0, 0, 0, "detected"), Point(1, 1, None, None, "occluded"),
                Point(2, 2, 2, 2, "corrected"), Point(4, 4, 4, 4, "detected")]
        self.assertEqual([[r.frame for r in g] for g in runs(rows)], [[0], [2], [4]])
        for group in runs(rows):
            self.assertEqual(fit_run(group, np.eye(3))["intervals"], [])

    def test_pinned_cubic_is_bounded_continuously(self):
        rng = np.random.default_rng(1)
        rows = [Point(i, i / 30, float(8 * i), float(200 - 3 * i + rng.normal(0, 0.3)), "detected")
                for i in range(30)]
        result = fit_run(rows, np.eye(3))
        self.assertEqual(result["method"], "cubic")
        self.assertGreater(max(result["residuals_px"]), 0.001)
        self.assertLessEqual(result["max_control_deviation_px"], result["budget_px"] + 1e-8)
        np.testing.assert_array_equal(np.asarray(result["raw_xy"])[[0, -1]],
                                      np.asarray(result["fitted_xy"])[[0, -1]])
        xy = np.asarray(result["raw_xy"])
        for j, interval in enumerate(result["intervals"]):
            curve = np.asarray(interval["xy"])
            delta = xy[j + 1] - xy[j]
            u = ((curve - xy[j]) @ delta) / (delta @ delta)
            nearest = xy[j] + np.clip(u, 0, 1)[:, None] * delta
            self.assertLessEqual(np.linalg.norm(curve - nearest, axis=1).max(), result["budget_px"] + 1e-8)
            if j == len(result["intervals"]) - 1:
                self.assertLessEqual(u.max(), 1 + 1e-9)
        for limit in range(1, 30):
            prefix = trail_intervals([result], limit)
            np.testing.assert_array_equal(prefix[-1]["xy"][-1], [rows[limit].x, rows[limit].y])

    def test_stationary_interval_falls_back_to_raw(self):
        rows = [Point(i, i / 24, float(i), float(i % 3), "detected") for i in range(12)]
        rows[6] = Point(6, 6 / 24, rows[5].x, rows[5].y, "corrected")
        result = fit_run(rows, np.eye(3))
        self.assertEqual(result["method"], "raw")
        np.testing.assert_array_equal(result["raw_xy"], result["fitted_xy"])
        np.testing.assert_array_equal(result["intervals"][5]["xy"], np.repeat([[5.0, 2.0]], 17, axis=0))

    def test_correction_undo_and_occlusion(self):
        row = Point(3, 3 / 24, 4, 5, "detected")
        model = Edits([row], 24, (100, 100))
        model.set(3, 12, 15)
        self.assertEqual(model.rows[3].status, "corrected")
        model.set(3)
        self.assertIsNone(model.rows[3].x)
        model.undo()
        self.assertEqual(model.rows[3].x, 12)
        model.undo()
        self.assertEqual(model.rows[3], row)
        with self.assertRaises(ValueError):
            model.set(3, 101, 15)

    def test_headless_cleanup_preserves_desktop_diagnostic(self):
        opening_error = cv2.error("No display available")
        cleanup_error = cv2.error("destroyAllWindows unavailable")
        row = Point(3, 3 / 24, 4, 5, "detected")
        with (
            patch("overlay.correction.metadata", return_value=(24, 10, (100, 100))),
            patch("overlay.correction.read_points", return_value=[row]),
            patch("overlay.correction.cv2.namedWindow", side_effect=opening_error),
            patch("overlay.correction.cv2.destroyAllWindows", side_effect=cleanup_error) as cleanup,
        ):
            with self.assertRaisesRegex(ValueError, "desktop and GUI-enabled") as raised:
                correct("clip.avi", "points.csv")
        self.assertIs(raised.exception.__cause__, opening_error)
        cleanup.assert_called_once_with()

    def test_release_and_finish_timing(self):
        flights = [{"id": "a", "release": 5, "end": 35}, {"id": "b", "release": 20, "end": 35}]
        common = schedule(flights, {"a": 30, "b": 30}, "common", duration=1.2)
        for item in common.values():
            self.assertEqual(source_frame_at(item["start_s"] - 0.01, item), item["release"] - 1)
            self.assertEqual(source_frame_at(item["start_s"], item), item["release"])
            self.assertEqual(source_frame_at(item["start_s"] + 1.2, item), item["end"])
        stagger = schedule(flights, {"a": 30, "b": 30}, "staggered", stagger=0.8)
        self.assertAlmostEqual(stagger["b"]["start_s"] - stagger["a"]["start_s"], 0.8)
        self.assertEqual(stagger["a"]["speed"], 1)
        self.assertEqual(len(set(palette(13))), 13)


if __name__ == "__main__":
    unittest.main()
