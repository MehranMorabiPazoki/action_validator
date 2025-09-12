import threading
import zmq
import msgpack
import numpy as np
import logging
import queue

logger = logging.getLogger("fingerprint_listener")


class FingerprintListener(threading.Thread):
    def __init__(self, addr: str = "tcp://fingerprint:5555", timeout_sec: int = 5):
        super().__init__(daemon=True)  # daemon=True → closes when main thread exits
        self.addr = addr
        self.timeout_sec = timeout_sec
        self._stop_event = threading.Event()

        # store last fingerprint
        self.latest_fingerprint = None
        self.lock = threading.Lock()

        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.connect(self.addr)
        self.output_queue = queue.Queue(maxsize=5)
    def run(self):
        logger.info(f"🧤 FingerprintListener started. Listening on {self.addr}")

        while not self._stop_event.is_set():
            if self.socket.poll(self.timeout_sec * 1000):  # timeout in ms
                try:
                    data = self.socket.recv()
                    payload = msgpack.unpackb(data)

                    image_data = payload.get("image")
                    width = payload.get("width")
                    height = payload.get("height")

                    if image_data and width and height:
                        fingerprint_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width))
                        self.output_queue.put(fingerprint_array)

                        logger.info("✅ Fingerprint received")
                except Exception as e:
                    logger.error(f"❌ Error receiving fingerprint: {e}")
            else:
                logger.debug("⏰ Fingerprint timeout")

        logger.info("🛑 FingerprintReader stopped.")

    def get_data(self):
        if self.output_queue.empty():
            logger.info("Queue is empty")
            return None
        data = self.output_queue.get()
        logger.info(f"Retrieved and removed person frame from the queue")
        return data
    
    def stop(self):
        """Stop thread safely."""
        self._stop_event.set()
        self.socket.close()
        self.context.term()
