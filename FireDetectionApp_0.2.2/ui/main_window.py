from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from core.camera import Camera
from core.detector import Detector
from ui.camera_manager import CameraManager
from ui.log_panel import LogPanel
from utils.debug import log, print_banner, print_memory
from utils.image_utils import fit_pixmap_to_label, frame_to_qimage


class MainWindow(QMainWindow):
    """
    The main window. Owns the camera manager, video display, log panel,
    and the detector. Basically wires everything together.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fire Detection System")
        self.setGeometry(100, 100, 1400, 720)
        self.setStyleSheet("background-color: #1a1a2e; color: white;")

        # Whichever camera is currently running. None until the user hits Start.
        self.active_camera: Camera | None = None

        # Single detector shared across all sources so the log persists when switching
        self.detector = Detector(model_path="fire.pt")

        self._flash_timer = QTimer()
        self._flash_timer.timeout.connect(self._do_flash)
        self._flash_count = 0

        self._setup_ui()

        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._update_frame)

        log("APP", "MainWindow ready")

    def _setup_ui(self):
        self.video_label = QLabel("Pick a source above and press Start")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #0f0f1a; border: 2px solid #ff6b35;"
            "border-radius: 8px; font-size: 15px; color: #ff6b35;"
        )
        self.video_label.setMinimumSize(800, 500)

        self.info_label = QLabel("Fire detected: 0")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("font-size: 17px; color: #ff6b35; padding: 6px;")

        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet(self._btn_style("#4ecca3"))
        self.start_btn.clicked.connect(self.start_feed)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(self._btn_style("#e84545"))
        self.stop_btn.clicked.connect(self.stop_feed)
        self.stop_btn.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()

        self.camera_manager = CameraManager()
        self.camera_manager.source_selected.connect(self._on_source_selected)

        left_col = QVBoxLayout()
        left_col.setSpacing(10)
        left_col.addWidget(self.camera_manager)
        left_col.addWidget(self.video_label)
        left_col.addWidget(self.info_label)
        left_col.addLayout(btn_row)

        self.log_panel = LogPanel()
        self.log_panel.cleared.connect(self._on_log_cleared)
        self.log_panel.fire_detected.connect(self._start_taskbar_flash)

        root = QHBoxLayout()
        root.setContentsMargins(16, 16, 16, 10)
        root.setSpacing(16)
        root.addLayout(left_col)
        root.addWidget(self.log_panel)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.status = QStatusBar()
        self.status.setStyleSheet("color: #888; font-size: 12px;")
        self.setStatusBar(self.status)
        self.status.showMessage("Ready  |  model: fire.pt  |  select a source and press Start")

    def _btn_style(self, color: str) -> str:
        bg = "#1a1a2e"
        return (
            f"QPushButton {{ background-color: {bg}; color: {color}; border: 2px solid {color};"
            f"border-radius: 6px; padding: 8px 24px; font-size: 13px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: {color}; color: {bg}; }}"
            f"QPushButton:disabled {{ border-color: #444; color: #444; }}"
        )

    # source switching

    def _on_source_selected(self, camera: Camera):
        # If something is already playing, stop it first before switching over
        if self.frame_timer.isActive():
            self.stop_feed()
        self.active_camera = camera
        self.status.showMessage(f"Selected: {camera.label}  |  press Start to begin")
        log("APP", f"Switched to: {camera.label}")

    # camera control

    def start_feed(self):
        if self.active_camera is None:
            self.active_camera = self.camera_manager.get_selected_camera()

        if self.active_camera is None:
            self.status.showMessage("No source selected. Add one above.")
            return

        self.status.showMessage(f"Opening {self.active_camera.label} ...")
        if not self.active_camera.open():
            self.status.showMessage(f"ERROR: Could not open {self.active_camera.label}")
            log("CAMERA", f"Failed to open: {self.active_camera.label}")
            return

        if self.active_camera.is_image:
            # Static image, just run the detector at 1 tick per second
            fps = 1
        elif self.active_camera.is_video_file:
            # Read the FPS that's stored in the video file itself so it plays
            # at the right speed rather than as fast as the GPU can go
            fps = self.active_camera.fps
            log("CAMERA", f"Video file FPS from file metadata: {fps}")
        else:
            # Live camera, actually measure how fast it delivers frames
            self.status.showMessage("Measuring camera FPS ...")
            fps = self.active_camera.measure_fps()

        interval_ms = int(1000 / fps)
        self.frame_timer.start(interval_ms)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        w, h = self.active_camera.resolution
        self.status.showMessage(
            f"Running: {self.active_camera.label}  |  {fps} FPS  |  {w}x{h}"
        )
        print_banner("fire.pt", self.active_camera.source, fps, (w, h))
        log("CAMERA", f"Feed started: {self.active_camera.label}")

    def stop_feed(self):
        self.frame_timer.stop()
        self._flash_timer.stop()

        if self.active_camera is not None:
            self.active_camera.release()

        self.video_label.setText("Pick a source above and press Start")
        self.info_label.setText("Fire detected: 0")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.showMessage(
            f"Stopped  |  {self.detector.total_tracked} fire events logged this session"
        )
        log("CAMERA", "Feed stopped")
        print_memory()

    # frame loop

    def _update_frame(self):
        if self.active_camera is None:
            return

        ok, frame = self.active_camera.read()
        if not ok:
            # End of video file or camera disconnected
            self.status.showMessage("Feed ended or file finished.")
            log("CAMERA", "Read failed or end of file")
            self.stop_feed()
            return

        annotated, new_ids, updated_ids = self.detector.process_frame(frame)

        for tid in new_ids:
            self.log_panel.add_entry(self.detector.tracked_objects[tid])

        # Update last-seen time on existing entries, but not every single frame
        for tid in updated_ids:
            obj = self.detector.tracked_objects[tid]
            if obj.frames % 30 == 0:
                self.log_panel.update_entry(obj)

        in_frame = len(new_ids) + len(updated_ids)
        self.info_label.setText(
            f"Fire in frame: {in_frame}   |   Total events: {self.detector.total_tracked}"
        )

        qt_image = frame_to_qimage(annotated)
        self.video_label.setPixmap(fit_pixmap_to_label(qt_image, self.video_label))

    # taskbar flash

    def _start_taskbar_flash(self):
        if not self._flash_timer.isActive():
            self._flash_count = 0
            self._flash_timer.start(500)
            log("ALERT", "Taskbar flash started")

    def _do_flash(self):
        QApplication.alert(self, 0)
        self._flash_count += 1
        if self._flash_count >= 10:
            self._flash_timer.stop()
            log("ALERT", "Taskbar flash ended")

    # slots

    def _on_log_cleared(self):
        self.detector.clear()
        self.info_label.setText("Fire detected: 0")
        self.status.showMessage("Log cleared")
        log("APP", "Log cleared by user")

    def closeEvent(self, event):
        log("APP", "Closing, cleaning up")
        self.stop_feed()
        event.accept()
