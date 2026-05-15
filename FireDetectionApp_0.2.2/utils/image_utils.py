import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel


def frame_to_qimage(frame: np.ndarray) -> QImage:
    """Turn an OpenCV BGR frame into something Qt can display."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)


def fit_pixmap_to_label(image: QImage, label: QLabel) -> QPixmap:
    """Scale the image to fit the label without stretching it."""
    return QPixmap.fromImage(image).scaled(
        label.size(),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
