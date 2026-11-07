"""One command-line entry point. No network services or external executables."""

import argparse
import sys

import cv2


def parser():
    p = argparse.ArgumentParser(prog="python -m overlay", description="Observed-only thrown-object overlays")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Generate synthetic clips and run every pipeline stage")
    d.add_argument("--output", default="demo-output")
    t = sub.add_parser("track", help="Detect a bright coloured object")
    t.add_argument("video")
    t.add_argument("--output", required=True)
    t.add_argument("--start", type=int, default=0)
    t.add_argument("--end", type=int)
    t.add_argument("--seed", type=float, nargs=2, metavar=("X", "Y"))
    t.add_argument("--window", type=float, default=70)
    t.add_argument("--hue", type=int, nargs=2, action="append", metavar=("LOW", "HIGH"))
    t.add_argument("--saturation", type=int, default=100)
    t.add_argument("--value", type=int, default=70)
    t.add_argument("--min-area", type=int, default=8)
    t.add_argument("--max-area", type=int, default=2500)
    t.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    c = sub.add_parser("correct", help="Seek, step, click, mark unseen, undo and save")
    c.add_argument("video")
    c.add_argument("points")
    c.add_argument("--output", help="Defaults to replacing the input CSV on explicit save")
    c.add_argument("--edits", help="Replay a JSON edit list without a desktop")
    a = sub.add_parser("align", help="Estimate one masked background transform per clip")
    a.add_argument("scene")
    a.add_argument("--output", required=True)
    s = sub.add_parser("smooth", help="Fit jitter-bounded cubic splines, with raw fallback")
    s.add_argument("scene")
    s.add_argument("--alignment", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--plot", required=True)
    for name, help_text in (("overlay", "Render a persistent-trail still"), ("video", "Render silent release-aligned playback")):
        r = sub.add_parser(name, help=help_text)
        r.add_argument("scene")
        r.add_argument("--alignment", required=True)
        r.add_argument("--smoothing", required=True)
        r.add_argument("--output", required=True)
        r.add_argument("--every", type=int, default=6, help="Retain a real cutout every N frames and at the last observation")
        r.add_argument("--crop", choices=["4:5"])
        r.add_argument("--centre-x", type=float)
        if name == "video":
            r.add_argument("--mode", choices=["common", "staggered"], default="common")
            r.add_argument("--duration", type=float, help="Common presented duration in seconds (common mode only)")
            r.add_argument("--stagger", type=float, default=0.7)
            r.add_argument("--fps", type=float, default=24)
    b = sub.add_parser("plate", help="Build a plate from selected subject-absent frames")
    b.add_argument("video")
    b.add_argument("--frames", type=int, nargs="+", required=True)
    b.add_argument("--output", required=True)
    return p


def main(argv=None):
    p = parser()
    a = vars(p.parse_args(argv))
    command = a.pop("command")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    try:
        if command == "demo":
            from .demo import demo
            demo(**a)
        elif command == "track":
            from .tracking import track
            hues = a.pop("hue") or [[0, 12], [168, 179]]
            settings = {"hues": hues, "saturation": a.pop("saturation"), "value": a.pop("value")}
            if any(not 0 <= lo <= hi <= 179 for lo, hi in hues):
                raise ValueError("Hue bounds must be between 0 and 179")
            track(settings=settings, **a)
        elif command == "correct":
            from .correction import correct
            correct(**a)
        elif command == "align":
            from .alignment import align
            align(a["scene"], a["output"])
        elif command == "smooth":
            from .smoothing import smooth
            smooth(a["scene"], a["alignment"], a["output"], a["plot"])
        elif command == "plate":
            from .plate import plate
            plate(**a)
        else:
            from .rendering import overlay, video
            a["crop"] = a["crop"] is not None
            if command == "video" and a["mode"] == "staggered" and a["duration"] is not None:
                raise ValueError("--duration is only valid for common playback")
            {"overlay": overlay, "video": video}[command](**a)
    except (ValueError, OSError, KeyError, cv2.error) as exc:
        p.exit(2, "error: " + str(exc) + chr(10))


if __name__ == "__main__":
    main(sys.argv[1:])
