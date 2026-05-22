import sys
import time
from pathlib import Path
sys.path.insert(0, r"d:\pz\src")

import cv2
from puzzle_recognition.config import DetectorConfig, MatcherConfig
from puzzle_recognition.board_builder import load_board_config
from puzzle_recognition.piece_detector import detect_black_pieces
from puzzle_recognition.shape_matcher import find_candidate_slots, score_piece_slot

image_path = r"C:\Users\kioki\.gemini\antigravity\brain\28e31ef9-ef68-4527-9a89-ab5861391674\media__1779463900515.jpg"
board_id = "board_photo_skeleton_w3"

print("Loading board...")
board_config = load_board_config(board_id)
print(f"Board loaded. Slots: {len(board_config['slots'])}")

print("Loading image...")
image = cv2.imread(image_path)
if image is None:
    print("Failed to load image!")
    sys.exit(1)

# Default high resolution detector config
det_cfg = DetectorConfig(
    gray_threshold=60,
    min_piece_area=800.0,
    max_piece_area=350000.0,
    max_bbox_width=800,
    max_bbox_height=800,
    solidity_max=1.0,
    expected_max_pieces=10,
)

print("Detecting pieces...")
pieces, masks = detect_black_pieces(image, det_cfg)
print(f"Detected {len(pieces)} pieces.")

matcher_cfg = MatcherConfig()
print(f"Matcher Config: max_candidates={matcher_cfg.max_candidates}, area_tolerance={matcher_cfg.area_tolerance}")

for i, piece in enumerate(pieces):
    print(f"\nPiece {i}: id={piece['piece_id']}, area={piece['area']}, bbox={piece['bbox']}")
    t0 = time.time()
    candidates = find_candidate_slots(piece, board_config["slots"], matcher_cfg)
    t1 = time.time()
    print(f"  Found {len(candidates)} candidates in {t1-t0:.4f}s")
    for c in candidates[:5]:
        print(f"    Candidate: slot_id={c['slot_id']}, area={c['area']}, prefilter_score={c['prefilter_score']:.4f}")
