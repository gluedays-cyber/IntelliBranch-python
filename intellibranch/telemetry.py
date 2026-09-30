import threading
import time
from dataclasses import dataclass
from typing import List


@dataclass
class TelemetryEvent:
    input_text: str
    predicted_label: str
    secondary_label: str = ""
    confidence: float = 0.0
    entropy: float = 0.0
    is_ambiguous: bool = False
    is_pipeline: bool = False
    is_fallback: bool = False
    timestamp_nano: int = 0


class TelemetryRingBuffer:
    """Thread-safe bounded ring buffer for asynchronous telemetry feedback."""

    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            capacity = 1024
        self.capacity: int = capacity
        self.events: List[TelemetryEvent | None] = [None] * capacity
        self.head: int = 0
        self.tail: int = 0
        self._count: int = 0
        self._lock = threading.Lock()

    def push(self, event: TelemetryEvent) -> None:
        """Records an event into the ring buffer, overwriting the oldest entry if capacity is reached."""
        with self._lock:
            event.timestamp_nano = time.time_ns()
            self.events[self.head] = event
            self.head = (self.head + 1) % self.capacity

            if self._count < self.capacity:
                self._count += 1
            else:
                self.tail = (self.tail + 1) % self.capacity

    def drain(self) -> List[TelemetryEvent]:
        """Extracts and clears all recorded events in FIFO order for retraining and drift analysis."""
        with self._lock:
            if self._count == 0:
                return []
            result: List[TelemetryEvent] = []
            for i in range(self._count):
                idx = (self.tail + i) % self.capacity
                ev = self.events[idx]
                if ev is not None:
                    result.append(ev)
                self.events[idx] = None

            self.head = 0
            self.tail = 0
            self._count = 0
            return result

    def count(self) -> int:
        """Returns the number of currently buffered events."""
        with self._lock:
            return self._count
