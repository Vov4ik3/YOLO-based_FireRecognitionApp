from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from models.tracked_object import TrackedObject


class LogPanel(QWidget):
    """
    Sidebar widget that logs every fire detection event with a full timecode.
    Each entry is a timestamped row - one per new detection.

    Signals:
        fire_detected - emitted when a new fire appears (used to flash taskbar)
        cleared       - emitted when the user wipes the log
    """

    cleared = pyqtSignal()
    fire_detected = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self._setup_ui()

    def _setup_ui(self):
        title = QLabel("🔥 Fire Detection Log")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #ff6b35;"
            "border-bottom: 1px solid #ff6b35; padding-bottom: 6px;"
        )

        self.count_label = QLabel("No events recorded")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.count_label.setStyleSheet("font-size: 11px; color: #888; padding: 2px;")

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #0f0f1a;
                border: 2px solid #ff6b35;
                border-radius: 8px;
                font-size: 12px;
                color: #ccc;
                padding: 4px;
            }
            QListWidget::item {
                padding: 8px 4px;
                border-bottom: 1px solid #1e1e3a;
            }
            QListWidget::item:selected {
                background-color: #2a1a0a;
            }
        """)

        clear_btn = QPushButton("🗑  Clear Log")
        clear_btn.setStyleSheet(
            "QPushButton { background-color: #1a1a2e; color: #888; border: 2px solid #888;"
            "border-radius: 6px; padding: 8px 20px; font-size: 13px; font-weight: bold; }"
            "QPushButton:hover { background-color: #888; color: #1a1a2e; }"
        )
        clear_btn.clicked.connect(self._on_clear)

        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.addWidget(title)
        layout.addWidget(self.count_label)
        layout.addWidget(self.list_widget)
        layout.addWidget(clear_btn)
        self.setLayout(layout)

    def add_entry(self, obj: TrackedObject):
        """
        Log a new fire detection event with full PC timecode.
        Each call produces one timestamped row in the log.
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S.%f")[:-3]  # millisecond precision

        text = (
            f"🔥  {obj.label.capitalize()} detected\n"
            f"    {date_str}  {time_str}\n"
            f"    Track ID: #{obj.track_id}"
        )

        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, obj.track_id)
        item.setForeground(QColor("#ff9966"))
        self.list_widget.insertItem(0, item)  # newest at top

        self._update_count()
        self.fire_detected.emit()

    def update_entry(self, obj: TrackedObject):
        """Update the last-seen time on an existing log entry."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S.%f")[:-3]
        first_str = obj.first_seen.strftime("%H:%M:%S.%f")[:-3]

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == obj.track_id:
                item.setText(
                    f"🔥  {obj.label.capitalize()} detected\n"
                    f"    First: {date_str}  {first_str}\n"
                    f"    Last:  {date_str}  {time_str}  |  Frames: {obj.frames}"
                )
                break

    def clear(self):
        """Clear all log entries."""
        self.list_widget.clear()
        self.count_label.setText("No events recorded")

    def _update_count(self):
        count = self.list_widget.count()
        self.count_label.setText(f"{count} event{'s' if count != 1 else ''} recorded")

    def _on_clear(self):
        self.clear()
        self.cleared.emit()
