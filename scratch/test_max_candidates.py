import sys
import time
from pathlib import Path
sys.path.insert(0, r"d:\pz\src")

import cv2
from puzzle_recognition.config import DetectorConfig, MatcherConfig
from puzzle_recognition.board_builder import load_board_config
from puzzle_recognition.piece_detector import detect_black_pieces
from puzzle_recognition.shape_matcher import find_candidate_slots, score_piece_slot, match_piece_to_board

# Load data
image_path = r"C:\Users\kioki\.gemini\antigravity\brain\28e31ef9-ef68-4527-9a89-ab5861391674\media__1779463900515.jpg"
board_id = "board_photo_skeleton_w3"
board_config = load_board_config(board_id)
image = cv2.imread(image_path)
det_cfg = DetectorConfig(
    gray_threshold=60,
    min_piece_area=800.0,
    max_piece_area=350000.0,
    max_bbox_width=800,
    max_bbox_height=800,
    solidity_max=1.0,
    expected_max_pieces=10,
)
pieces, _ = detect_black_pieces(image, det_cfg)

print("=== TESTING WITH DIFFERENT MAX_CANDIDATES ===")

for max_c in [5, 8, 10, 50]:
    cfg = MatcherConfig(max_candidates=max_c)
    print(f"\n--- Running recognize with max_candidates={max_c} ---")
    t0 = time.time()
    results = []
    for piece in pieces:
        res = match_piece_to_board(piece, board_config, cfg)
        results.append((piece["piece_id"], res["matched_slot_id"], res["iou"]))
    t1 = time.time()
    print(f"Time taken: {t1-t0:.4f}s")
    for pid, slot_id, iou in results:
        print(f"  {pid}: matched_slot_id={slot_id}, IoU={iou:.4f}")
