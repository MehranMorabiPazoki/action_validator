import zmq
import json
import time
from typing import Any, Dict,List
import logging
import numpy as np

logger = logging.getLogger("BackendSender")

def make_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class BackendSender:
    """
    Publishes action_validator results via ZeroMQ PUB socket.
    """

    def __init__(self, addr: str = "tcp://0.0.0.0:5555", topic: str = "action_validator"):
        """
        :param addr: ZeroMQ bind address for PUB socket
        :param topic: Message topic (subscribers must subscribe to this)
        """
        self.addr = addr
        self.topic = topic

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind(self.addr)

        # small delay to allow subscribers to connect
        time.sleep(0.5)

        logger.info(f"[BackendSender] PUB socket bound to {self.addr} on topic '{self.topic}'")

    def send(self, results: List, sample_id: str):
        """
        Publish an action_validator result message (multipart: [topic, json]).
        """
        message = {
            "service": "action_validator",
            "sample_id": sample_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "action_results": results,
        }

        safe_message = make_json_serializable(message)
        json_message = json.dumps(safe_message)

        # send topic and message as multipart
        self.socket.send_multipart([self.topic.encode("utf-8"), json_message.encode("utf-8")])

        logger.info(f"[BackendSender] Published topic='{self.topic}': {safe_message}")


    def close(self):
        """Close the ZeroMQ socket."""
        self.socket.close()
        logger.info("[BackendSender] PUB socket closed.")
