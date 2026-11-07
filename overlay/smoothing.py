"""Endpoint-pinned natural cubic smoothing, with a conservative geometric bound.

The exact integrated squared second-derivative penalty is solved with NumPy.
This keeps the small solver inspectable and also works without SciPy installed.
"""

import numpy as np
from PIL import Image, ImageDraw

from .data import Scene, project, runs, transforms, write_json

SUBDIVISIONS = 16


def penalty(t):
    n = len(t)
    h = np.diff(t)
    q = np.zeros((n, n - 2))
    r = np.zeros((n - 2, n - 2))
    for j in range(n - 2):
        q[j, j] = 1 / h[j]
        q[j + 1, j] = -1 / h[j] - 1 / h[j + 1]
        q[j + 2, j] = 1 / h[j + 1]
        r[j, j] = (h[j] + h[j + 1]) / 3
        if j < n - 3:
            r[j, j + 1] = r[j + 1, j] = h[j + 1] / 6
    return q @ np.linalg.solve(r, q.T)


def controls(t, z):
    """Natural cubic spline, expressed as exact cubic Bezier control polygons."""
    n = len(t)
    h = np.diff(t)
    second = np.zeros_like(z)
    a = np.diag(2 * (h[:-1] + h[1:]))
    for i in range(n - 3):
        a[i, i + 1] = a[i + 1, i] = h[i + 1]
    second[1:-1] = np.linalg.solve(a, 6 * np.diff(np.diff(z, axis=0) / h[:, None], axis=0))
    slope = np.diff(z, axis=0) / h[:, None]
    left = slope - h[:, None] * (2 * second[:-1] + second[1:]) / 6
    right = slope + h[:, None] * (second[:-1] + 2 * second[1:]) / 6
    return np.stack((z[:-1], z[:-1] + h[:, None] * left / 3,
                     z[1:] - h[:, None] * right / 3, z[1:]), axis=1)


def sample_cubics(cp):
    u = np.linspace(0, 1, SUBDIVISIONS + 1)
    weights = np.stack(((1 - u) ** 3, 3 * u * (1 - u) ** 2,
                        3 * u ** 2 * (1 - u), u ** 3), axis=1)
    pieces = np.einsum("uk,ikd->iud", weights, cp)
    pieces[:, 0] = cp[:, 0]
    pieces[:, -1] = cp[:, -1]
    return pieces


def within_bound(cp, xy, budget):
    """A convex capsule contains all controls => it contains the WHOLE cubic.

    End controls must also stay behind the measured terminal cross-sections.
    This is stricter than checking a finite sample of a curve.
    """
    a, b = xy[:-1, None], xy[1:, None]
    v = b - a
    u = np.sum((cp - a) * v, axis=2) / np.maximum(1e-20, np.sum(v * v, axis=2))
    nearest = a + np.clip(u, 0, 1)[:, :, None] * v
    deviation = float(np.linalg.norm(cp - nearest, axis=2).max())
    ends_safe = u[0].min() >= -1e-9 and u[-1].max() <= 1 + 1e-9
    return deviation <= budget + 1e-9 and ends_safe, deviation


def fit_run(rows, transform):
    xy = project([[r.x, r.y] for r in rows], transform)
    frames = [r.frame for r in rows]
    # Equal frame spacing is required by the third-difference noise model.
    t = np.array([r.time for r in rows])
    t -= t[0]
    third = np.diff(xy, n=3, axis=0)
    sigma = (1.4826 * np.median(np.abs(third - np.median(third, axis=0)), axis=0) / np.sqrt(20)
             if len(third) >= 5 else np.zeros(2))
    budget = min(6.0, 2.5 * float(np.linalg.norm(sigma)))
    pins = {0, len(rows) - 1}
    for axis in range(2):
        for op in (np.argmin, np.argmax):
            i = int(op(xy[:, axis]))
            pins.update(range(max(0, i - 1), min(len(rows), i + 2)))
    z = xy.copy()
    pieces = np.array([np.linspace(a, b, SUBDIVISIONS + 1) for a, b in zip(xy, xy[1:])])
    method = "raw"
    reason = "Too few samples, zero jitter, stationary interval, or no bounded cubic"
    parameter, control_deviation = 0.0, 0.0
    equally_spaced = len(t) < 3 or np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-4)
    eligible = len(rows) >= 8 and budget > 1e-8 and equally_spaced and not np.any(np.linalg.norm(np.diff(xy, axis=0), axis=1) < 1e-10)
    if eligible:
        # Normalize time to frames for conditioning; lambda is in this time basis.
        basis = t / np.median(np.diff(t))
        k = penalty(basis)
        fixed = np.array(sorted(pins), dtype=int)
        free = np.array([i for i in range(len(rows)) if i not in pins], dtype=int)
        for lam in np.logspace(-4, 5, 24):
            system = np.eye(len(rows)) + lam * k
            trial = xy.copy()
            if len(free):
                trial[free] = np.linalg.solve(system[np.ix_(free, free)],
                                              xy[free] - system[np.ix_(free, fixed)] @ xy[fixed])
            cp = controls(basis, trial)
            allowed, deviation = within_bound(cp, xy, budget)
            if allowed and np.linalg.norm(trial - xy, axis=1).max() <= budget + 1e-9:
                z, pieces = trial, sample_cubics(cp)
                method, reason, parameter, control_deviation = "cubic", None, float(lam), deviation
    residuals = np.linalg.norm(z - xy, axis=1)
    intervals = [{"from": a.frame, "to": b.frame, "xy": piece.tolist()}
                 for a, b, piece in zip(rows, rows[1:], pieces)]
    return {"frames": frames, "raw_xy": xy.tolist(), "fitted_xy": z.tolist(),
            "intervals": intervals, "method": method, "fallback": reason,
            "sigma_xy": sigma.tolist(), "budget_px": budget, "lambda_frames": parameter,
            "max_control_deviation_px": control_deviation, "residuals_px": residuals.tolist(),
            "pinned_frames": [frames[i] for i in sorted(pins)]}


def residual_plot(path, flights):
    image = Image.new("RGB", (900, max(220, len(flights) * 200)), (24, 27, 32))
    draw = ImageDraw.Draw(image)
    for i, (name, groups) in enumerate(flights.items()):
        top = i * 200
        draw.text((16, top + 8), name + " | residual pixels; orange = bound", fill="white")
        all_frames = [f for run in groups for f in run["frames"]]
        lo, hi = min(all_frames), max(all_frames)
        ymax = max([g["budget_px"] for g in groups] + [1.0]) * 1.15
        draw.line((55, top + 160, 870, top + 160), fill=(120, 120, 120))
        for group in groups:
            xs = [55 + (f - lo) * 815 / max(1, hi - lo) for f in group["frames"]]
            ys = [top + 160 - r * 115 / ymax for r in group["residuals_px"]]
            by = top + 160 - group["budget_px"] * 115 / ymax
            draw.line((xs[0], by, xs[-1], by), fill=(255, 170, 60), width=2)
            if len(xs) > 1:
                draw.line(list(zip(xs, ys)), fill=(90, 220, 250), width=2)
            for x, y in zip(xs, ys):
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="white")
            draw.text((xs[0], top + 175), f"f{group['frames'][0]} {group['method']} B={group['budget_px']:.3f}", fill="white")
        draw.text((10, top + 45), f"{ymax:.2f}", fill="white")
        draw.text((10, top + 148), "0", fill="white")
    image.save(path)


def smooth(scene_path, alignment_path, output, plot):
    scene = Scene(scene_path)
    alignment = transforms(scene, alignment_path)
    flights = {}
    for flight in scene.flights:
        h = alignment["clips"][flight["video"]]["H"]
        flights[flight["id"]] = [fit_run(group, h) for group in runs(scene.rows(flight))]
    record = {"signature": scene.signature(alignment), "flights": flights,
              "jitter_note": "Third differences reject local quadratics, not true jerk or correlated errors. Inspect the raw measurements."}
    write_json(output, record)
    from pathlib import Path
    Path(plot).parent.mkdir(parents=True, exist_ok=True)
    residual_plot(plot, flights)
    return record
