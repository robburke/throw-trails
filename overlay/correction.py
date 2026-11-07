"""OpenCV scrubber and the same editing model for deterministic headless replay."""

import cv2

from .data import Point, capture, decode, metadata, read_json, read_points, write_points


class Edits:
    def __init__(self, rows, fps, size):
        self.rows = {r.frame: r for r in rows}
        self.fps, self.size = fps, size
        self.history = []
        self.dirty = False

    def set(self, frame, x=None, y=None):
        if frame not in self.rows:
            raise ValueError("Correction frame is outside the tracked CSV")
        if x is not None and not (y is not None and 0 <= x < self.size[0] and 0 <= y < self.size[1]):
            raise ValueError("Click must lie inside the source frame")
        point = Point(frame, frame / self.fps, x, y, "corrected" if x is not None else "occluded")
        self.history.append((frame, self.rows[frame]))
        self.rows[frame] = point
        self.dirty = True

    def undo(self):
        if self.history:
            frame, previous = self.history.pop()
            self.rows[frame] = previous
            self.dirty = True

    def save(self, path):
        write_points(path, sorted(self.rows.values(), key=lambda r: r.frame))
        self.dirty = False


def correct(video, points, output=None, edits=None):
    fps, count, size = metadata(video)
    rows = read_points(points)
    if rows[-1].frame >= count:
        raise ValueError("CSV extends beyond the video")
    model = Edits(rows, fps, size)
    output = output or points
    if edits:
        for edit in read_json(edits):
            if edit.get("action") == "undo":
                model.undo()
            elif edit.get("status") == "occluded":
                model.set(int(edit["frame"]))
            else:
                model.set(int(edit["frame"]), float(edit["x"]), float(edit["y"]))
        model.save(output)
        return model
    title = "Flight correction"
    index = rows[0].frame
    first, last = rows[0].frame, rows[-1].frame
    scale = min(1.0, 1200 / size[0], 760 / size[1])
    display_size = (round(size[0] * scale), round(size[1] * scale))
    message = "Click centre | A/D step | J/L ten | O unseen | U undo | S save | Q quit"

    def seek(value):
        nonlocal index
        index = max(first, min(last, value))

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            model.set(index, min(size[0] - 1, x * size[0] / display_size[0]),
                      min(size[1] - 1, y * size[1] / display_size[1]))

    try:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("frame", title, index, last, seek)
        cv2.setMouseCallback(title, click)
        with capture(video) as cap:
            while True:
                image = cv2.resize(decode(cap, index), display_size)
                row = model.rows.get(index)
                if row and row.x is not None:
                    cv2.drawMarker(image, (round(row.x * display_size[0] / size[0]),
                                           round(row.y * display_size[1] / size[1])),
                                   (0, 255, 255), cv2.MARKER_CROSS, 17, 1)
                status = row.status if row else "outside CSV"
                cv2.putText(image, f"{index} {status} | {message}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                cv2.imshow(title, image)
                key = cv2.waitKeyEx(25)
                if key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    if model.dirty:
                        print("Unsaved corrections discarded; use S to save before quitting.")
                    break
                if key in (ord("a"), 2424832, 65361):
                    seek(index - 1)
                elif key in (ord("d"), 2555904, 65363):
                    seek(index + 1)
                elif key == ord("j"):
                    seek(index - 10)
                elif key == ord("l"):
                    seek(index + 10)
                elif key == ord("o") and index in model.rows:
                    model.set(index)
                elif key == ord("u"):
                    model.undo()
                elif key == ord("s"):
                    model.save(output)
                    message = "Saved | A/D step | O unseen | U undo | S save | Q quit"
                cv2.setTrackbarPos("frame", title, index)
    except cv2.error as exc:
        raise ValueError("Correction needs a desktop and GUI-enabled opencv-python; headless replay uses --edits") from exc
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return model
