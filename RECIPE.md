# Make several flights visible at once

The idea is simple: film a few launches from one spot, mark where the object really appears, and lay those separate flights onto one background. A curved trail makes the motion easier to compare; a few real cutouts help the viewer see what was flying. This example grew out of a foam glider from a **CrunchLabs** box. The **Magnus effect**, a force on a spinning object moving through air, is the physics that makes that glider curve.

## 1. Give the camera an easy job

Prop the phone somewhere solid and leave it there for every launch. Lock exposure and focus if the camera app allows it. Launch across the frame rather than toward the camera, and leave space around the whole flight. A plain background with a few fixed features around its edges helps both object detection and background alignment.

Record a practice flight, then step through it frame by frame. Can you see a sharp object in most frames? If it is a smear, try brighter light or a shorter exposure before recording the real takes. A higher frame rate helps only if each frame still has enough light. Keep all takes at the same frame rate; convert variable-frame-rate phone clips to constant frame rate before tracking.

Film a few empty seconds before or after the launches, with everyone and the object out of frame. Those empty frames can become the background plate. If you ever want sound, record ten seconds of ambience too; this tool's output is silent, so add sound separately later.

## 2. Track the object

**First run.** Use Python 3.11+ and run these commands from the package directory, in your usual virtual environment. Interactive `correct` and the frame viewer below need a desktop window and GUI-enabled `opencv-python`, not a headless session.

```sh
python -m pip install -r requirements.txt
python -m overlay demo
```

Open `demo-output/overlay.png`. For your own clip, create an `own/` folder here and copy the original phone recording into it by USB or your usual file transfer, as `phone.mov`. With ffmpeg available locally, convert it to constant 30 fps before choosing frames:

```sh
ffmpeg -i own/phone.mov -vf fps=30 -fps_mode cfr -c:v mjpeg -q:v 2 -an own/launch.avi
```

**Choose a flight.** Frame numbers start at zero in the converted clip. This viewer shows native pixels: click the object to print `(x, y)` and its HSV value; press a key to close. Change `6` and rerun to inspect other frames, identify release/end and check that frames 0, 1 and 2 are empty. Pixel `(0, 0)` is top left; x increases right and y down.

```sh
python -c "import cv2; from overlay.data import frame; image = frame('own/launch.avi', 6); hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV); cv2.namedWindow('Frame', cv2.WINDOW_AUTOSIZE); cv2.setMouseCallback('Frame', lambda event, x, y, flags, param: print('x,y:', x, y, 'HSV:', hsv[y, x].tolist()) if event == cv2.EVENT_LBUTTONDOWN else None); cv2.imshow('Frame', image); cv2.waitKey(0); cv2.destroyAllWindows()"
```

The worked example below assumes a small red object, release at frame 6 near `(128, 235)`, end at 40, and a 480 by 320 clip. Replace those values with yours. Sample several visible object pixels: OpenCV hue is 0-179, saturation/value are 0-255. Choose hue intervals covering the samples (red wraps across zero, so use two); put saturation/value minimums below the object's samples while excluding the background. Keep these same settings in the scene. Choose a search radius large enough for the next position and area limits around the object's visible pixel area.

```sh
python -m overlay track own/launch.avi --start 6 --end 40 --seed 128 235 --window 50 --hue 0 12 --hue 168 179 --saturation 100 --value 70 --min-area 8 --max-area 2500 --output own/launch.csv
```

The tracker writes one CSV row per requested frame. A failed detection gets `occluded` with empty coordinates. Its internal motion guess only helps find the next real detection.

## 3. Correct what the machine misunderstood

**Track, click.** After tracking, open the clip and CSV:

```sh
python -m overlay correct own/launch.avi own/launch.csv
```

Seek with the slider; A/D or arrow keys step one frame, and J/L step ten. Click the visible centre to mark a correction. Press O when the object cannot be seen, U to undo, and S to save. Quit with Q after saving.

Inspect every proposed point, especially near a hand, a similarly coloured background object, an occluder, or the frame edge. Leave invisible frames empty. Do not draw a plausible flight through an absence. Identify the release frame by stepping through separation from the hand or launcher; if it is hidden, record that uncertainty in the caption. You can still compare the visible portions.

## 4. Put the backgrounds together

**Write one scene.** Save this as `own/scene.json`, substituting your inspected values. All paths inside it are relative to the scene file, so use `launch.avi`, not `own/launch.avi`. The reference and alignment both use empty frame 0 of the same clip, giving identity alignment for this one-launch example.

```json
{
  "reference": {"video": "launch.avi", "frame": 0, "exclude": [[15, 55, 465, 275]]},
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

Make the moving-region mask by inspecting the whole clip and replacing `exclude` with rectangles covering every person, object and other moving region across all frames. Coordinates are `[left, top, right, bottom]`, right/bottom exclusive, inside the native frame. The tool makes these boxes black (excluded) and the remainder white (fixed features allowed). Leave useful background texture unmasked. For more launches, add flight entries and use the same alignment specification for flights sharing a clip.

Use `plate` on genuinely empty frames from the reference clip, then `align`. The plate's pixel-selection median uses actual donor RGB pixels; it cannot remove somebody present in every selected frame.

```sh
python -m overlay plate own/launch.avi --frames 0 1 2 --output own/plate.png
python -m overlay align own/scene.json --output own/alignment.json
```

Give `align` masks that allow fixed background features and exclude the entire area containing people, the flying object, moving foliage and other motion. You can use black/white mask images or excluded rectangles. Alignment fits one homography per clip, shared by its flights. Check the recorded residuals and the alignment of fixed edges. A moving camera, nearby objects at different depths, or a featureless wall may need a new shoot rather than more fitting. Never drag an individual path until its release sits on another launch's release.

## 5. Smooth only what the measurements permit

Run `smooth`, then inspect `own/residuals.png`:

```sh
python -m overlay smooth own/scene.json --alignment own/alignment.json --output own/smoothed.json --plot own/residuals.png
```

Third differences largely remove a locally quadratic flight curve, leaving a robust estimate of pixel jitter; real jerk and correlated tracking mistakes can contaminate it, so check the frames too. Each uninterrupted run gets an endpoint-pinned cubic smoothing spline only when the whole curve fits within that jitter budget; otherwise the tool keeps its raw measured segments.

The budget, chosen fit, endpoint pins and any fallback are saved alongside the plot. A missing frame always splits the run. Smoothing is a restrained presentation choice, not evidence of an unseen position.

## 6. Render and explain it

Run `overlay` to draw the persistent tapered trails and real cutouts onto the base. Start with one cutout every six or seven source frames, skipping unseen ones. Inspect the outline of each object: the automatic mask is deliberately simple, and a multicoloured glider may need hand-drawn alpha mattes. Keep only visible object pixels, preserve holes, and never paint a hidden part back in. The cutouts stay at their measured positions even when the trail is smoothed.

For a supplied matte, save the relevant decoded frame as a PNG (`cv2.imwrite('own/frame-0006.png', image)` in the viewer command), open it in an image editor, and paint a source-sized grayscale mask: white on the visible object, black everywhere else, preserving holes. Save as `own/object-0006.png` without resizing and add `"mattes": {"6": "object-0006.png"}` to the flight. Repeat for every observed frame where colour segmentation is unusable, including frames skipped by `--every`. Set `cutout_radius` to enclose the whole object. After changing points or scene settings, rerun `align` and `smooth` before rendering.

```sh
python -m overlay overlay own/scene.json --alignment own/alignment.json --smoothing own/smoothed.json --every 7 --output own/overlay.png
```

Render the full frame first. Try `--crop 4:5` for a portrait version, moving the horizontal crop with `--centre-x` if needed. For the optional "pew pew pew" sequence, `video --mode staggered --stagger 0.7` introduces each launch in turn and leaves its trail visible. `video --mode common` aligns release times and changes each flight's speed to reach its declared finish or cutoff together. Both modes are silent and use real observed object poses. Active trails use raw segments to keep every current endpoint exact; completed runs can use the approved spline. Dashed rings mean visibility ended, not that an impact was observed.

```sh
python -m overlay video own/scene.json --alignment own/alignment.json --smoothing own/smoothed.json --mode common --duration 1.5 --output own/common.avi
python -m overlay video own/scene.json --alignment own/alignment.json --smoothing own/smoothed.json --mode staggered --stagger 0.7 --output own/staggered.avi
```

A useful caption is:

> Separate launches filmed from a fixed camera and aligned by the background. Release positions differ. Trails join consecutive observed frames only; gaps are left blank, cutouts come from measured frames, and smoothing stays within estimated pixel jitter. Nothing drawn represents an unseen flight position. Dashed rings mark visibility limits.

Add whether playback was sped up or slowed down, whether release timing was uncertain, and whether an ending was out of view. Avoid saying "same throw" or "same release point" when the recording does not support it.

## Starter prompt for a coding agent

> Help me make an observed-only overlay of separate throws from a fixed camera. Read README.md and run the synthetic demo first. Track the bright object, let me correct every frame, and align clips using masked fixed background features. Split paths at every gap. Pin endpoints and allow smoothing only within measured jitter. Keep real cutouts at measured positions, never extend a trail, and mark unseen ends. Render a still and optional silent staggered video. Ask me to identify release frames and check masks before final rendering.

## The one-launch version for a parent

Prop the phone, choose a plain background and film one launch across the screen. Check a few frames for a sharp object. Follow the worked example above in order: install, demo, transfer and convert, inspect frames, `track`, `correct`, write the scene, `plate`, `align`, `smooth`, then `overlay`. Use the reference clip's empty frame as the base and its own reference-frame identity for alignment. Skip the video and portrait crop until the still tells the story. If a frame is unclear, leave it unseen; a short honest trail is enough.

## Inspiration and credits

Broadcast pitch overlays are the prior art: [Pitching Ninja's Paul Skenes four-pitch overlay](https://www.youtube.com/shorts/VcFRmNfdIbI) and [u/chonyyy's pitch-overlay project in r/learnmachinelearning](https://old.reddit.com/r/learnmachinelearning/comments/l3db4k/updated_automatically_overlaying_baseball_pitch/). No footage or code from either is included here.

This recipe combines persistent trails whose width encodes elapsed source time, release-aligned comparison, optional staggered launches, and observed-only geometry.

Built by Rob Burke with Claude (orchestrating), Codex Sol (building) and Codex Astra (critique), November 2026.
