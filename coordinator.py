from actionrecognition_inference import mmaction_inference
import threading
import time
import logging
from enum import Enum

logger = logging.getLogger("Coordinator")

class CoordinatorState(Enum):
    IDLE = 0
    ACTIVE = 1

class Coordinator(threading.Thread):
    def __init__(self, camera_listeners, state_manager, poll_interval=0.05):
        super().__init__(daemon=True)
        self.camera_listeners = camera_listeners
        self.state_manager = state_manager
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()

        # Buffers initialized when ACTIVE
        self.cam_data = {}
        
        self.active = False

    def run(self):
        logger.info("Coordinator started.")
        while not self._stop_event.is_set():
            current_active = self.state_manager.is_active()

            # --- Handle ACTIVE/IDLE transitions ---
            if self.active == False:
                if current_active:
                    self._enter_active_state()
            else :
                # --- Collect data when ACTIVE ---
                self._collect_camera_data()
                # --- Trigger inference if all data ready ---
                if all(v is not None for v in self.cam_data.values()):
                    self._run_inference()
                    self._enter_idle_state()

            time.sleep(self.poll_interval)

        logger.info("Coordinator stopped.")

    def _enter_active_state(self):
        """Transition to ACTIVE: init buffers."""
        self.cam_data = {listener.bind_addr: None for listener in self.camera_listeners}
        
        self.active = True
        logger.info("System ACTIVE → buffers initialized")

    def _enter_idle_state(self):
        """Transition to IDLE: clear buffers."""
        self.cam_data.clear()
        
        self.active = False
        logger.info("System IDLE → buffers cleared")

    def _collect_camera_data(self):
        """Collect one sample from each camera listener if available."""
        for listener in self.camera_listeners:
            if self.cam_data[listener.bind_addr] is None:
                try:
                    data = listener.get_data()
                    if data:
                        self.cam_data[listener.bind_addr] = data
                        logger.info("📷 Camera %s provided data: %s",
                                    listener.bind_addr, str(data))
                except Exception as e:
                    logger.warning("Failed to get data from %s: %s", listener.bind_addr, e)

    def _run_inference(self):
        """Run MMAction2 inference when all camera inputs are ready."""
        logger.info("✅ All camera inputs ready → running action recognition")

        try:
            # Pass dict {camera_id: clip_path_or_dir} to inference
            results = mmaction_inference(dict(self.cam_data),delete_videos=False)
            logger.info(f"mmaction_inference  result → results= {results}, ")
        except Exception as e:
            logger.exception("Error running auth pipeline: %s", e)


    def stop(self):
        """Signal thread to stop gracefully."""
        self._stop_event.set()
