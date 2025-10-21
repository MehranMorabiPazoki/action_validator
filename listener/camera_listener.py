import cv2
import os
import tempfile
import shutil
import zmq
import numpy as np
import threading
import logging
import time
import queue


logger = logging.getLogger("CameraListener")

class CameraListener(threading.Thread):
    def __init__(self, iou_threshold, state_manager, anchor_manager, 
                 bind_addr, timeout_ms=5000, fps=20, mode="video"):
        """
        mode = "video" → output video file path
        mode = "frames" → output directory with extracted frames
        """
        super().__init__(daemon=True)
        self.state_manager = state_manager
        self.anchor_manager = anchor_manager
        self.bind_addr = bind_addr
        self.timeout_ms = timeout_ms
        self.iou_threshold = iou_threshold
        self.fps = fps
        self.mode = mode
        self._stopped = threading.Event()

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect(self.bind_addr)
        logger.info(f"Connected SUB socket to {self.bind_addr}")

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        self.output_queue = queue.Queue(maxsize=5) 
        self.camera_id  = bind_addr.split("/")[-1]
        # Recording state
        self.video_writer = None
        self.video_path = None
        self.frames_dir = None
        self.frame_idx = 0
        self.recording = False

    def decode_raw_frame(self, frame_bytes, width, height):
        try:
            frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))
            return frame
        except Exception as e:
            logger.exception(f"Error decoding raw frame: {e}")
            return None

    def get_camera_img(self):
        socks = dict(self.poller.poll(self.timeout_ms))
        if self.socket in socks and socks[self.socket] == zmq.POLLIN:
            try:
                parts = self.socket.recv_multipart(zmq.NOBLOCK)
                if len(parts) != 5:
                    logger.info("Invalid message format received.")
                    return None, None, None
                camera_id_bytes, timestamp_ns_bytes, width_bytes, height_bytes, frame_bytes = parts
                width = int(width_bytes.decode("utf-8"))
                height = int(height_bytes.decode("utf-8"))
                frame = self.decode_raw_frame(frame_bytes, width, height)
                return frame, width, height
            except zmq.Again:
                logger.info("Non-blocking receive failed.")
        return None, None, None

    def _start_recording(self, width, height,base_dir):
        if self.recording:
            return

        ts = int(time.time())
        if self.mode == "video":
            self.video_path = os.path.join(base_dir, f"clip_{self.camera_id}_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (width, height))
        elif self.mode == "frames":
            self.frames_dir = os.path.join(tempfile.gettempdir(), f"clip_{ts}_frames")
            os.makedirs(self.frames_dir, exist_ok=True)
            self.frame_idx = 0

        self.recording = True
        logger.info(f"Started recording ({self.mode})")

    def _stop_recording(self):
        if not self.recording:
            return

        if self.mode == "video":
            self.video_writer.release()
            self.video_writer = None
            self.output_queue.put(self.video_path)
            logger.info(f"Stopped recording. Video saved at {self.video_path}")
        elif self.mode == "frames":
            self.output_queue.put(self.frames_dir)
            logger.info(f"Stopped recording. Frames saved at {self.frames_dir}")

        self.recording = False

    def run(self):
        logger.info("Listener thread started, waiting for camera frames...")
        while not self._stopped.is_set():
            try:
                frame, width, height = self.get_camera_img()
                if frame is None:
                    continue

                if self.state_manager.is_active():
                    if not self.recording:
                        self._start_recording(width=width, height=height,base_dir=os.path.join("/shared/data",self.state_manager.get_id(),"action_validator"))

                    if self.mode == "video":
                        self.video_writer.write(frame)
                    elif self.mode == "frames":
                        self.frame_idx += 1
                        fname = os.path.join(self.frames_dir, f"img_{self.frame_idx:05d}.jpg")
                        cv2.imwrite(fname, frame)

                else:
                    if self.recording:
                        self._stop_recording()

            except Exception as e:
                logger.exception(f"Error in listener loop: {e}")

        # Cleanup on exit
        self._stop_recording()

    def get_data(self):
        """Retrieve recorded video path or frames dir for feeding into MMAction2"""
        if self.output_queue.empty():
            return None
        return self.output_queue.get()

    def stop(self):
        self._stopped.set()
        logger.info("Listener thread signaled for shutdown.")
