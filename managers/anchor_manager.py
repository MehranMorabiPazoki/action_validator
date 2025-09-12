
import threading


class AnchorManager:
    def __init__(self, anchors_config):
        """
        anchors_config: list of anchor dicts for a single camera, e.g.
        [
            {"id": "cam1_zone1", "name": "Entrance Left", "box": [x1, y1, x2, y2]},
            ...
        ]
        """
        self.lock = threading.Lock()
        # Initialize anchors with triggered flag
        self.anchors = [
            {"id": a["id"], "box": a["box"], "triggered": False, "name": a.get("name", "")}
            for a in anchors_config
        ]

    def get_all_anchors(self):
        with self.lock:
            return [anchor.copy() for anchor in self.anchors]
    def set_triggered(self, anchor_id):
        with self.lock:
            for anchor in self.anchors:
                if anchor["id"] == anchor_id:
                    anchor["triggered"] = True
                    break
    def is_triggered(self, anchor_id):
        with self.lock:
            for anchor in self.anchors:
                if anchor["id"] == anchor_id:
                    return anchor["triggered"]
            return False

    def reset_all(self):
        with self.lock:
            for anchor in self.anchors:
                    anchor["triggered"] = False


def create_anchor_managers_from_config(config):
    """
    config: full config dictionary loaded from YAML
    Returns dict: {camera_id: AnchorManager_instance}
    """
    managers = {}
    cameras = config.get("cameras", {})
    for cam_id, cam_cfg in cameras.items():
        anchors_config = cam_cfg.get("anchors", [])
        managers[cam_id] = AnchorManager(anchors_config)
    return managers