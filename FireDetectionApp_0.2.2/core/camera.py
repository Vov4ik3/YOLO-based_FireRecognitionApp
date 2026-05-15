import time

import cv2
import numpy as np


class Camera:
    """
    One video source. Could be a webcam, an IP camera, a video file, or a still image.
    The rest of the app doesn't need to care which one it is, they all work the same way.
    """

    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

    def __init__(self, source: int | str = 0, label: str = ""):
        self.source = source
        self.label = label or self._default_label(source)

        self._cap: cv2.VideoCapture | None = None
        self._measured_fps: int | None = None

        self._static_frame: np.ndarray | None = None

        # Figure out what kind of source this is upfront
        src_lower = source.lower() if isinstance(source, str) else ""
        self._is_image = src_lower.endswith(self.IMAGE_EXTENSIONS)
        self._is_video_file = src_lower.endswith(self.VIDEO_EXTENSIONS)

    def open(self) -> bool:
        """Open the source. Returns True if it worked."""
        if self._is_image:
            self._static_frame = cv2.imread(self.source)
            return self._static_frame is not None

        self._cap = cv2.VideoCapture(self.source)
        return self._cap.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Get the next frame. Images just return the same frame every time so the
        rest of the app doesn't need special cases for them.
        """
        if self._is_image:
            if self._static_frame is not None:
                return True, self._static_frame.copy()
            return False, None

        if self._cap is None or not self._cap.isOpened():
            return False, None

        return self._cap.read()

    def release(self):
        """Let go of the camera or file handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._static_frame = None
        self._measured_fps = None

    @property
    def is_open(self) -> bool:
        if self._is_image:
            return self._static_frame is not None
        return self._cap is not None and self._cap.isOpened()

    @property
    def is_image(self) -> bool:
        return self._is_image

    @property
    def is_video_file(self) -> bool:
        return self._is_video_file

    @property
    def fps(self) -> int:
        """
        The FPS for this source. For video files this reads the value stored
        in the file itself so playback stays at the right speed. For live cameras
        it uses whatever was measured (or falls back to 30 if we don't know yet).
        """
        if self._is_image:
            return 1
        if not self.is_open:
            return 30

        # For video files always trust the file's own FPS value, not a measured one.
        # Measuring would just tell us how fast we can decode, not the intended speed.
        if self._is_video_file:
            reported = self._cap.get(cv2.CAP_PROP_FPS)
            if reported > 0:
                return int(reported)
            return 30

        # Live camera: use measured value if we have one, otherwise ask OpenCV
        if self._measured_fps is not None:
            return self._measured_fps
        reported = self._cap.get(cv2.CAP_PROP_FPS)
        if reported <= 0 or reported > 120:
            return 30
        return int(reported)

    def measure_fps(self, sample_frames: int = 30) -> int:
        """
        Time how fast the camera actually delivers frames by reading a bunch
        and dividing by how long it took. Takes about 1 second.
        Only useful for live cameras, skipped for everything else.
        """
        if self._is_image or self._is_video_file:
            return self.fps
        if not self.is_open:
            return 30
        start = time.time()
        for _ in range(sample_frames):
            self._cap.read()
        elapsed = time.time() - start
        fps = int(sample_frames / elapsed)
        self._measured_fps = max(1, min(fps, 120))
        return self._measured_fps

    @property
    def resolution(self) -> tuple[int, int]:
        """Width and height of the source."""
        if self._is_image and self._static_frame is not None:
            h, w = self._static_frame.shape[:2]
            return (w, h)
        if not self.is_open:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @staticmethod
    def _default_label(source: int | str) -> str:
        """Come up with a readable name if the caller didn't give one."""
        if isinstance(source, int):
            return f"Webcam {source}"
        if isinstance(source, str):
            if source.startswith("rtsp://") or source.startswith("http://"):
                return f"IP Camera ({source[:30]}...)"
            return source.split("/")[-1].split("\\")[-1]
        return str(source)
