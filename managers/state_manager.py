from enum import Enum
import threading
import logging
logger = logging.getLogger("Coordinator")

class SystemState(Enum):
    IDLE = 0
    ACTIVE = 1

class GlobalStateManager:
    def __init__(self):
        self.state = SystemState.IDLE
        self._lock = threading.Lock()

    def activate(self):
        with self._lock:
            self.state = SystemState.ACTIVE
        logger.info("[STATE] System activated (entrance trigger)")

    def deactivate(self):
        with self._lock:
            self.state = SystemState.IDLE
        logger.info("[STATE] System deactivated (exit trigger)")

    def is_active(self):
        with self._lock:
            return self.state == SystemState.ACTIVE