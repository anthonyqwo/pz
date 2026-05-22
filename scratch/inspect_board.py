import json
from pathlib import Path

board_id = "board_photo_skeleton_w3"
boards_dir = Path(r"d:\pz\data\boards")
config_path = boards_dir / board_id / "board_config.json"

with open(config_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Board ID:", data.get("board_id"))
print("Rectified Size:", data.get("rectified_size"))
slots = data.get("slots", [])
print("Number of slots:", len(slots))
if slots:
    print("Example slot key structures:", list(slots[0].keys()))
    print("First slot:", {k: slots[0][k] for k in ["slot_id", "area", "bbox"]})
