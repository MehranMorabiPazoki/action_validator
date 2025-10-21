from enum import Enum
import threading
import logging

logger = logging.getLogger("State Manager")

class SystemState(Enum):
    IDLE = 0
    ACTIVE = 1

class GlobalStateManager:
    def __init__(self):
        self.state = SystemState.IDLE
        self._lock = threading.Lock()
        self.id = ""
    def activate(self,uuid,timestamp):
        with self._lock:
            self.state = SystemState.ACTIVE
        logger.info("System activated (entrance trigger)")
        self.id = f"{uuid}_{timestamp}"
    def deactivate(self):
        with self._lock:
            self.state = SystemState.IDLE
        logger.info("System deactivated (exit trigger)")
    def get_id(self):
        return self.id
    def is_active(self):
        with self._lock:
            return self.state == SystemState.ACTIVE