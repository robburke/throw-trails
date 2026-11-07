# Flight overlays

Make one picture of several thrown-object launches, aligned against a fixed background, with persistent tapered trails and real object cutouts. This small Python tool tracks a bright object, lets you correct its measured positions, and renders a still or a silent video with simultaneous or staggered releases. It includes synthetic footage generated entirely in code, so you can try the complete workflow without supplying a recording.

## Try it

Use **Python 3.11+**, from this directory. Install the dependencies into your usual virtual environment:

```sh
python -m pip install -r requirements.txt
python -m overlay demo
```

The demo generates three short textured-background clips with a red disc, a five-frame occlusion, slightly different camera positions, and an unseen ending. It runs tracking, scripted correction, plate creation, masked alignment, bounded smoothing, still rendering and both video modes. Expect a run under a minute on a typical laptop; codec support and machine speed matter.

Open **`demo-output/overlay.png`** for the output image, **`demo-output/portrait.png`** for the 4:5 crop, and `demo-output/common.avi` or `demo-output/staggered.avi` for playback. These are generated files, not included assets. `demo-output/expected.json` contains the synthetic ground truth and expected invariants; `result.json` records the measured runtime and status counts. Rerunning the demo replaces its named output files.

## Commands

These examples all use the generated sample. Run them in order after the demo, or adapt the paths and scene description for your footage. `python -m overlay COMMAND --help` lists the options.

```sh
# Bright-red detection in a bounded search window; frames are zero-based.
python -m overlay track demo-output/launch-1.avi --start 6 --end 40 --seed 128 235 --window 50 --min-area 25 --max-area 400 --output demo-output/launch-1.csv

# Desktop scrubber: seek with the slider, A/D or arrows to step, J/L for ten.
# Click the object centre, O for unseen, U to undo, S to save, Q to quit.
python -m overlay correct demo-output/launch-1.avi demo-output/launch-1.csv

# Without a desktop, replay the demo's known synthetic correction instead.
python -m overlay correct demo-output/launch-1.avi demo-output/launch-1.csv --edits demo-output/edits.json

# Choose frames where the subject and object are genuinely absent.
python -m overlay plate demo-output/launch-1.avi --frames 0 1 2 --output demo-output/plate.png

# One background homography per clip, with moving regions masked out.
python -m overlay align demo-output/scene.json --output demo-output/alignment.json

# Per-contiguous-run cubic fit; raw fallback and residual chart are automatic.
python -m overlay smooth demo-output/scene.json --alignment demo-output/alignment.json --output demo-output/smoothed.json --plot demo-output/residuals.png

# Persistent tapered trails; cutouts every seventh source frame that was observed.
python -m overlay overlay demo-output/scene.json --alignment demo-output/alignment.json --smoothing demo-output/smoothed.json --every 7 --output demo-output/overlay.png

# Optional 4:5 crop. Shift its horizontal centre with --centre-x PIXELS.
python -m overlay overlay demo-output/scene.json --alignment demo-output/alignment.json --smoothing demo-output/smoothed.json --every 7 --crop 4:5 --output demo-output/portrait.png

# Release-aligned, per-flight speed warp to a common presented finish/cutoff.
python -m overlay video demo-output/scene.json --alignment demo-output/alignment.json --smoothing demo-output/smoothed.json --mode common --duration 1.5 --output demo-output/common.avi

# Each launch starts later, at its native playback speed. No audio is added.
python -m overlay video demo-output/scene.json --alignment demo-output/alignment.json --smoothing demo-output/smoothed.json --mode staggered --stagger 0.7 --output demo-output/staggered.avi
```

`track` accepts repeated `--hue LOW HIGH` ranges in OpenCV's 0-179 hue scale, saturation/value thresholds, component area limits, and `--roi X0 Y0 X1 Y1`. Red is the default. The seed and prediction guide the search; they never become observations. Correct false detections before rendering. The OpenCV correction window needs a desktop and GUI-enabled `opencv-python`; a headless build can run the rest of the pipeline and scripted edits.

All CSVs have exactly `frame,time,x,y,status`. Time is seconds from clip start at the decoded constant frame rate, coordinates are native source pixels, and status is `detected`, `corrected`, or `occluded`; occluded coordinates are empty. Scripted edits are a JSON list such as `[{"frame": 6, "x": 128, "y": 235}, {"frame": 7, "status": "occluded"}, {"action": "undo"}]`. GUI saving replaces the CSV unless `--output` names a different file. Quitting does not save.

## Describe your own scene

Copy the structure of generated `demo-output/scene.json`. Paths in a scene are relative to that scene file. A minimal one-launch scene looks like this:

```json
{
  "reference": {
    "video": "launch.avi", "frame": 0,
    "exclude": [[15, 55, 465, 275]]
  },
  "base": "plate.png",
  "flights": [{
    "id": "launch-1", "video": "launch.avi", "points": "launch.csv",
    "release": 6, "end": 40,
    "alignment": {"frame": 0, "exclude": [[15, 55, 465, 275]]},
    "colour": {"hues": [[0, 12], [168, 179]], "saturation": 100, "value": 70},
    "cutout_radius": 18
  }]
}
```

Replace those example frame numbers and boxes. `release` is the frame of separation you identify by inspection, not automatically the first detection; `end` is the final frame you choose to consider, and may be an unseen cutoff. The optional base must already be in reference coordinates; without it the reference frame is used directly. A plate from another clip must first be registered to the reference outside this command.

Alignment excludes rectangles `[left, top, right, bottom]`, with right/bottom exclusive. Alternatively supply `"mask": "static.png"` in each alignment specification and in the reference: white pixels permit fixed background features, black pixels exclude moving regions. Masks are native source-sized images. Use the same alignment specification for every flight in a shared clip. ORB matches and RANSAC residuals are recorded, and weak or concentrated matches fail with an error. A constant homography assumes a steady camera and limited parallax; inspect aligned background edges yourself.

Automatic cutouts keep only the visible local colour component, including real holes; they do not infer a multicoloured body from coloured tips. For a complex object supply optional flight `"mattes": {"6": "object-0006.png"}` entries: each grayscale alpha matte is source-sized and must isolate only visible object pixels. Rendering prepares cutouts for every observed frame, including frames skipped by `--every`; each needs usable automatic segmentation or a supplied matte. `cutout_radius` must enclose the whole object. Cutouts are warped from their actual measured frames and positions, independently of trail smoothing. A missing usable cutout causes an error rather than a substitute shape.

Videos use repeated or skipped source observations, without synthesized in-between object poses. Common mode changes per-flight speed; staggered mode keeps native flight duration. Timing sidecars disclose those changes. An active run uses raw measured chords so its current endpoint is exact; a completed run may use its bounded spline. A crop can remove part of a trail, so inspect the uncropped output too. Use constant-frame-rate video; transcode variable-frame-rate phone footage first with your preferred video tool.

## Honesty rules

- Observed points only. A guessed search position is never exported as a measurement.
- Never interpolate across a frame gap or an occlusion. Each gap splits the trail and smoothing fit.
- Never extend a trail past its last observation, including at every playback cutoff. Line width and endpoint rings are visual symbols, not additional measured positions.
- Keep cutouts at the raw measured positions. Do not move them onto a smoothed path or reconstruct hidden pixels.
- Smooth only within the measured jitter estimate. Run endpoints are pinned; short, stationary, zero-budget or unsuccessful fits retain raw geometry.
- Dashed rings mark the last known position before an unseen section, or the first observation after an unseen release. They do not locate the hidden endpoint or imply a landing.
- Keep launch positions different when they were different. Background registration is shared per clip; never shift individual trajectories to make their starts coincide.
- Disclose separate launches, any playback speed changes, unseen endings, and uncertainty in the release frame.

The jitter estimate uses a robust third-difference noise diagnostic and caps the radial budget at six reference pixels. Every cubic Bezier control lies within that budget's convex capsule around its raw segment, so the entire continuous curve is bounded, not just the plotted samples. This is a diagnostic rather than proof of measurement accuracy: true jerk and correlated errors can affect it. Inspect `residuals.png` and each run's budget, residuals, pins, method and fallback in `smoothed.json`.

Read [RECIPE.md](RECIPE.md) for shooting advice, caption wording, the parent shortcut, and a coding-agent starter prompt.

## Checks and limits

```sh
python -B -m unittest discover -s tests -v
```

The tests generate their own sample under an ignored local directory, check truth/statuses and occlusions, refuse stale geometry after corrections, decode both videos, and test bounded smoothing and exact trail ends at every playback cutoff. No real footage, tracked coordinates, masks or caches are distributed. Runtime products are ignored by default; review the directory itself before sharing it.

The detector is a starting point for bright objects against distinguishable backgrounds. It does not estimate 3D motion, infer hidden flight, automatically identify release/landing, or guarantee segmentation on arbitrary footage. No network, model weights, FFmpeg executable or cloud account is needed at runtime. Dependency installation is separate.

Licence: MIT (see LICENSE).
