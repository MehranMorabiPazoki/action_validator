# === FILE: trigger_listener.py ===
import zmq
import threading
import json
import logging

# Setup module logger
logger = logging.getLogger("trigger listener")


def start_trigger_listener(state_manager, anchor_managers,trigger_addr):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(trigger_addr)
    socket.setsockopt_string(zmq.SUBSCRIBE, "detection.command")  # subscribe to your topic only
    logger.info(f"Connected SUB socket to {trigger_addr}")
    while True:
        # Receive a multipart message: [topic, json_payload]
        parts = socket.recv_multipart()
        if len(parts) != 2:
            logger.warning("Invalid message format received.")
            continue

        topic, payload_bytes = parts
        topic = topic.decode("utf-8")
        try:
            msg = json.loads(payload_bytes.decode("utf-8"))
        except Exception as e:
            logger.error(f"JSON decode error: {e}")
            continue

        # Check payload's structure and fields
        if topic == "detection.command" and msg.get("msg_type") == "command":
            command = msg.get("command")
            uuid  = msg.get("msg_id")
            timestamp = msg.get("timestamp")
            logger.info(f"Received command: {command}")
            if command == "ENTRY":
                state_manager.activate(uuid=uuid,timestamp=timestamp)
                logger.info("StateManager activated.")
            elif command == "EXIT" or command == "RESET":
                state_manager.deactivate()
                logger.info("StateManager deactivated..")
        else:
            logger.debug(f"Ignored message with topic '{topic}' or type '{msg.get('msg_type')}'.")
