import logging
import yaml
import threading
from managers import AnchorManager, GlobalStateManager, create_anchor_managers_from_config
from listener import CameraListener, FingerprintListener,start_trigger_listener
from coordinator import Coordinator

"""Configure the root logger."""
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def load_config(path="config.yml"):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    logger.info("Loading configuration...")
    config = load_config("config.yml")

    logger.info("Creating anchor managers...")
    anchor_managers = create_anchor_managers_from_config(config)

    logger.info("Initializing state manager...")
    state_manager = GlobalStateManager()

    trigger_thread = threading.Thread(
        target=start_trigger_listener,
        args=(state_manager, anchor_managers,"tcp://entry_detection:5560"),
        daemon=True,
    )
    trigger_thread.start()
    # Start camera listeners
    camera_process_threads = []
    for camera_id, socket_name in config.get("camera_sockets", {}).items():
        logger.info(f"Starting ZMQProxyListener for camera {camera_id} on socket {socket_name}...")
        t = CameraListener(
            iou_threshold=0.4,
            state_manager=state_manager,
            anchor_manager=anchor_managers[camera_id],
            bind_addr=f"ipc:///tmp/{socket_name}",
            fps=20,
            mode="video",
            timeout_ms=5000,
        )
        t.start()
        camera_process_threads.append(t)


    # Start coordinator
    logger.info("Starting Coordinator...")
    coordinator = Coordinator(camera_process_threads, state_manager)
    coordinator.start()

    # In real deployment, you'd run indefinitely or under supervision
    try:
        for t in camera_process_threads:
            t.join()
        coordinator.join()
        trigger_thread.join()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        # Stop everything cleanly
        coordinator.stop()
        trigger_thread.stop()
        # join again to ensure clean exit
        for t in camera_process_threads:
            t.join()
        coordinator.join()
        trigger_thread.join()
        logger.info("All services stopped successfully.")


if __name__ == "__main__":
    main()
