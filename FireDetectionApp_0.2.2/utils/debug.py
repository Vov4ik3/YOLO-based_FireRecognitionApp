"""
Debug mode for FireDetectionApp.

Pass --debug when launching to turn this on:
    python main.py --debug

Prints per-frame detections, confidence scores, inference time, and memory usage.
In normal mode none of this runs so there's no performance hit.
"""

import sys
import time

DEBUG: bool = "--debug" in sys.argv


def is_debug() -> bool:
    return DEBUG


def log(section: str, message: str):
    """Prints [SECTION] message to console. Does nothing outside debug mode."""
    if DEBUG:
        print(f"[{section}] {message}")


class FrameTimer:
    """
    Tracks how long each frame takes and keeps a rolling average.
    Good for spotting if something is slowing down inference.
    """

    def __init__(self, window: int = 30):
        self._window = window
        self._times: list[float] = []
        self._frame_start: float = 0.0
        self.frame_count: int = 0

    def start(self):
        self._frame_start = time.perf_counter()

    def stop(self) -> float:
        """Returns how long this frame took in milliseconds."""
        elapsed_ms = (time.perf_counter() - self._frame_start) * 1000
        self._times.append(elapsed_ms)
        if len(self._times) > self._window:
            self._times.pop(0)
        self.frame_count += 1
        return elapsed_ms

    @property
    def avg_ms(self) -> float:
        return sum(self._times) / len(self._times) if self._times else 0.0

    @property
    def avg_fps(self) -> float:
        return 1000 / self.avg_ms if self.avg_ms > 0 else 0.0


def print_detections(results, model_names: dict, frame_count: int):
    """
    Print what got detected this frame. Shows class, confidence as a percentage,
    a little ASCII bar, and the pixel coordinates of the bounding box.
    Only fires when something was actually detected.
    """
    if not DEBUG:
        return

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return

    print(f"\n  Frame {frame_count}")

    confs = boxes.conf.tolist()
    cls_ids = boxes.cls.int().tolist()
    xyxy = boxes.xyxy.tolist()
    ids = (
        boxes.id.int().tolist()
        if boxes.id is not None
        else [None] * len(confs)
    )

    for track_id, cls_id, conf, box in zip(ids, cls_ids, confs, xyxy):
        label = model_names[cls_id]
        x1, y1, x2, y2 = [int(v) for v in box]
        id_str = f"#{track_id}" if track_id is not None else "#?"
        print(
            f"    {id_str:>4}  {label:<12}  conf: {conf:.2%}  {_conf_bar(conf)}  "
            f"box: [{x1},{y1} to {x2},{y2}]"
        )


def _conf_bar(conf: float, width: int = 10) -> str:
    """Makes a tiny bar like [████████░░] from a 0.0-1.0 confidence value."""
    filled = round(conf * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def print_memory():
    """
    Shows current RAM usage. Needs psutil (pip install psutil).
    Quietly skips if it's not installed.
    """
    if not DEBUG:
        return
    try:
        import os
        import psutil
        mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"[MEM] {mb:.1f} MB")
    except ImportError:
        pass


def print_banner(model_path: str, camera_source, fps: int, resolution: tuple):
    """Prints a quick summary at the top when you start a feed in debug mode."""
    if not DEBUG:
        return
    w, h = resolution
    print("=" * 52)
    print("  FireDetectionApp  DEBUG MODE")
    print("=" * 52)
    print(f"  Model:      {model_path}")
    print(f"  Source:     {camera_source}")
    print(f"  FPS:        {fps}")
    print(f"  Resolution: {w}x{h}")
    print("=" * 52)
