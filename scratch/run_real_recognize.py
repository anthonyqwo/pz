import sys
import time
from pathlib import Path
sys.path.insert(0, r"d:\pz\src")

from puzzle_recognition.config import DetectorConfig, MatcherConfig
from puzzle_recognition.recognizer import recognize

image_path = r"C:\Users\kioki\.gemini\antigravity\brain\28e31ef9-ef68-4527-9a89-ab5861391674\media__1779463900515.jpg"
board_id = "board_photo_skeleton_w3"

det_cfg = DetectorConfig(
    gray_threshold=60,
    min_piece_area=800.0,
    max_piece_area=350000.0,
    max_bbox_width=800,
    max_bbox_height=800,
    solidity_max=1.0,
    expected_max_pieces=10,
)

matcher_cfg = MatcherConfig(
    max_candidates=50,
)

print("Starting recognize()...")
t0 = time.time()
try:
    payload = recognize(
        image_path=image_path,
        board_id=board_id,
        already_rectified=True,
        detector_config=det_cfg,
        matcher_config=matcher_cfg,
        debug=True,
    )
    t1 = time.time()
    print(f"Recognize finished in {t1-t0:.4f}s")
    print(f"Detected: {payload.get('piece_count')} pieces.")
    for p in payload.get("pieces", []):
        print(f"Piece {p['piece_id']}: matched_slot_id={p['matched_slot_id']}, IoU={p['iou']:.4f}")
except Exception as e:
    import traceback
    traceback.print_exc()
