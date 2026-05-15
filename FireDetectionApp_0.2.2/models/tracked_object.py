from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TrackedObject:
    """Everything we know about one detected object."""

    track_id: int
    label: str
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    frames: int = 0

    def update(self):
        """Call this every frame the object is still visible."""
        self.last_seen = datetime.now()
        self.frames += 1

    @property
    def duration_seconds(self) -> float:
        """How long this object has been on screen in total."""
        return (self.last_seen - self.first_seen).total_seconds()

    def __str__(self) -> str:
        return (
            f"{self.label} #{self.track_id} | "
            f"First: {self.first_seen.strftime('%H:%M:%S')} | "
            f"Last: {self.last_seen.strftime('%H:%M:%S')} | "
            f"Frames: {self.frames}"
        )
