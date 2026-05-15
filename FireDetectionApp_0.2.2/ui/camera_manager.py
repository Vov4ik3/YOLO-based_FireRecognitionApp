"""
Camera manager panel shown in the left sidebar above the controls.

The user can:
  - See all currently added camera sources in a list
  - Switch to any of them with a single click
  - Add a new webcam by index (0, 1, 2 ...)
  - Add an IP camera by typing an RTSP URL
  - Import a video file (.mp4, .avi, .mov, .mkv) via file browser
  - Import an image file (.png, .jpg etc.) via file browser
  - Remove a source they no longer need

When the user picks a source, this widget emits the `source_selected` signal
with the Camera object. MainWindow listens for that and switches the feed.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.camera import Camera
from utils.debug import log


# Supported file types for the file browser dialogs
VIDEO_FILTER = "Video files (*.mp4 *.avi *.mov *.mkv *.wmv);;All files (*)"
IMAGE_FILTER = "Image files (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All files (*)"


class CameraManager(QWidget):
    """
    A panel that shows the list of camera sources and lets the user manage them.
    """

    # Fired when the user clicks a source in the list. Carries the Camera object.
    source_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        # List of Camera objects the user has added
        self._cameras: list[Camera] = []

        self._setup_ui()

        # Add the default webcam after the UI exists so source_list is ready
        self._add_camera(Camera(source=0, label="Webcam 0 (default)"))

    def _setup_ui(self):
        title = QLabel("Camera Sources")
        title.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #4ecca3;"
            "border-bottom: 1px solid #4ecca3; padding-bottom: 4px;"
        )

        # The list shows all added sources. Clicking one selects it.
        self.source_list = QListWidget()
        self.source_list.setMaximumHeight(120)
        self.source_list.setStyleSheet("""
            QListWidget {
                background-color: #0f0f1a;
                border: 1px solid #333;
                border-radius: 6px;
                font-size: 12px;
                color: #ccc;
            }
            QListWidget::item { padding: 4px 6px; }
            QListWidget::item:selected { background-color: #1e3a2a; color: #4ecca3; }
            QListWidget::item:hover { background-color: #1a2a1a; }
        """)
        self.source_list.itemClicked.connect(self._on_source_clicked)

        # Buttons for adding sources
        btn_webcam = QPushButton("+ Webcam")
        btn_ip = QPushButton("+ IP Camera")
        btn_video = QPushButton("+ Video file")
        btn_image = QPushButton("+ Image")
        btn_remove = QPushButton("Remove")

        for btn in [btn_webcam, btn_ip, btn_video, btn_image]:
            btn.setStyleSheet(self._add_btn_style())

        btn_remove.setStyleSheet(
            "QPushButton { background-color: #1a1a2e; color: #e84545; border: 1px solid #e84545;"
            "border-radius: 4px; padding: 4px 10px; font-size: 11px; }"
            "QPushButton:hover { background-color: #e84545; color: white; }"
        )

        btn_webcam.clicked.connect(self._add_webcam_dialog)
        btn_ip.clicked.connect(self._add_ip_dialog)
        btn_video.clicked.connect(self._add_video_dialog)
        btn_image.clicked.connect(self._add_image_dialog)
        btn_remove.clicked.connect(self._remove_selected)

        add_row = QHBoxLayout()
        add_row.setSpacing(4)
        for btn in [btn_webcam, btn_ip, btn_video, btn_image, btn_remove]:
            add_row.addWidget(btn)

        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(title)
        layout.addWidget(self.source_list)
        layout.addLayout(add_row)
        self.setLayout(layout)

        # Populate the list with the default camera
        self._refresh_list()

    def _add_btn_style(self) -> str:
        return (
            "QPushButton { background-color: #1a1a2e; color: #4ecca3; border: 1px solid #4ecca3;"
            "border-radius: 4px; padding: 4px 10px; font-size: 11px; }"
            "QPushButton:hover { background-color: #4ecca3; color: #1a1a2e; }"
        )

    def _add_webcam_dialog(self):
        """Ask the user for a webcam index (0, 1, 2 ...) and add it."""
        index, ok = QInputDialog.getInt(
            self,
            "Add Webcam",
            "Enter camera index (0 = built-in, 1 = first USB etc.):",
            value=0,
            min=0,
            max=10,
        )
        if ok:
            cam = Camera(source=index, label=f"Webcam {index}")
            self._add_camera(cam)
            log("CAMERA_MGR", f"Added webcam index {index}")

    def _add_ip_dialog(self):
        """Ask the user for an RTSP URL and add it as an IP camera."""
        url, ok = QInputDialog.getText(
            self,
            "Add IP Camera",
            "Enter RTSP or HTTP stream URL:\n(e.g. rtsp://admin:pass@192.168.1.10/stream)",
        )
        if ok and url.strip():
            url = url.strip()
            cam = Camera(source=url, label=f"IP: {url[:25]}...")
            self._add_camera(cam)
            log("CAMERA_MGR", f"Added IP camera: {url}")

    def _add_video_dialog(self):
        """Open a file browser and let the user pick a video file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            VIDEO_FILTER,
        )
        if path:
            cam = Camera(source=path)
            self._add_camera(cam)
            log("CAMERA_MGR", f"Added video file: {path}")

    def _add_image_dialog(self):
        """Open a file browser and let the user pick a static image."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            IMAGE_FILTER,
        )
        if path:
            cam = Camera(source=path)
            self._add_camera(cam)
            log("CAMERA_MGR", f"Added image file: {path}")

    def _remove_selected(self):
        """Remove whichever source is currently highlighted in the list."""
        row = self.source_list.currentRow()
        if row < 0:
            return
        if len(self._cameras) <= 1:
            QMessageBox.warning(self, "Cannot Remove", "You need at least one camera source.")
            return
        removed = self._cameras.pop(row)
        log("CAMERA_MGR", f"Removed source: {removed.label}")
        self._refresh_list()

    def _add_camera(self, cam: Camera):
        """Add a Camera object to the internal list and update the UI."""
        self._cameras.append(cam)
        self._refresh_list()

    def _refresh_list(self):
        """Rebuild the QListWidget from the internal cameras list."""
        self.source_list.clear()
        for cam in self._cameras:
            self.source_list.addItem(QListWidgetItem(cam.label))
        # Auto-select the first item
        if self.source_list.count() > 0:
            self.source_list.setCurrentRow(0)

    def _on_source_clicked(self, item: QListWidgetItem):
        """User clicked a source in the list. Emit it so MainWindow can switch."""
        row = self.source_list.row(item)
        if 0 <= row < len(self._cameras):
            cam = self._cameras[row]
            log("CAMERA_MGR", f"Source selected: {cam.label}")
            self.source_selected.emit(cam)

    def get_selected_camera(self) -> Camera | None:
        """Return the currently highlighted Camera, or None if the list is empty."""
        row = self.source_list.currentRow()
        if 0 <= row < len(self._cameras):
            return self._cameras[row]
        return None
