import os

from ultralytics import YOLO

from models.tracked_object import TrackedObject
from utils.debug import FrameTimer, is_debug, log, print_detections

# Go up one folder from core/ to find the project root where fire.pt lives
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Detector:
    """
    Wraps YOLO and keeps track of everything it has seen.
    One instance is shared across all camera sources so the log
    doesn't reset when you switch cameras.
    """

    def __init__(self, model_path: str = "fire.pt", classes: list[int] | None = None):
        full_path = os.path.join(BASE_DIR, model_path)
        log("DETECTOR", f"Loading model: {full_path}")

        self.model = YOLO(full_path)
        self.classes = classes  # None means detect everything the model knows about

        self.tracked_objects: dict[int, TrackedObject] = {}
        self._last_frame_ids: set[int] = set()
        self._timer = FrameTimer(window=30)

        log("DETECTOR", f"Ready. Classes: {self.model.names}")

    def process_frame(self, frame) -> tuple:
        """
        Run the model on one frame. Returns three things:
          - the frame with boxes drawn on it
          - list of IDs that showed up for the first time
          - list of IDs we've seen before that are still there
        """
        self._timer.start()

        results = self.model.track(
            frame,
            persist=True,
            classes=self.classes,
            verbose=False,
            conf=0.4,   # only accept detections the model is at least 90% sure about
            iou=0.5,    # merge boxes that overlap a lot (avoids double-counting)
            tracker="bytetrack.yaml",
        )

        elapsed = self._timer.stop()
        print_detections(results, self.model.names, self._timer.frame_count)

        if is_debug() and self._timer.frame_count % 30 == 0:
            log(
                "PERF",
                f"Frame {self._timer.frame_count:>6} | "
                f"inference: {elapsed:>6.1f}ms | "
                f"avg: {self._timer.avg_ms:>6.1f}ms | "
                f"fps: {self._timer.avg_fps:>5.1f}",
            )

        annotated = results[0].plot()
        boxes = results[0].boxes

        new_ids: list[int] = []
        updated_ids: list[int] = []
        self._last_frame_ids = set()

        if boxes is not None and boxes.id is not None:
            for track_id, class_id in zip(
                boxes.id.int().tolist(),
                boxes.cls.int().tolist(),
            ):
                label = self.model.names[class_id]
                self._last_frame_ids.add(track_id)

                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = TrackedObject(
                        track_id=track_id,
                        label=label,
                    )
                    log("TRACK", f"New  ID #{track_id}  ({label})")
                    new_ids.append(track_id)
                else:
                    self.tracked_objects[track_id].update()
                    updated_ids.append(track_id)

        return annotated, new_ids, updated_ids

    def clear(self):
        """Forget everything. Called when the user clears the log."""
        self.tracked_objects.clear()
        self._last_frame_ids.clear()
        log("DETECTOR", "Cleared")

    @property
    def total_tracked(self) -> int:
        return len(self.tracked_objects)

    @property
    def current_count(self) -> int:
        return len(self._last_frame_ids)
