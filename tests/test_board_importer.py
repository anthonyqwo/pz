import numpy as np
import cv2
import pytest
from puzzle_recognition.auto_board_importer import components_to_slots

def test_components_to_slots_watershed_auto_seeds():
    # Create a 200x200 grid
    # Black background (pieces will be black, walls are white)
    # Draw a outer boundary exactly at the image edges (0, 0) to (199, 199)
    # and pass border_thickness=0 so the border filter does not reject any valid pieces.
    line_mask = np.zeros((200, 200), dtype=np.uint8)
    
    # Draw outer box at image edges
    cv2.rectangle(line_mask, (0, 0), (199, 199), 255, 3)
    
    # Draw horizontal and vertical white lines inside the box
    cv2.line(line_mask, (100, 0), (100, 200), 255, 3)
    cv2.line(line_mask, (0, 100), (200, 100), 255, 3)
    
    # Introduce a small gap in the lines to simulate a broken seam!
    # A gap of size 10 at the intersection (100, 100) or along a line
    line_mask[95:105, 95:105] = 0
    
    # Run watershed-based slots extraction with auto-seeds
    # min_slot_area = 500, max_slot_area = 20000, border_thickness = 0, wall_dilate = 0
    slots = components_to_slots(
        line_mask,
        min_slot_area=500,
        max_slot_area=20000,
        border_thickness=0,
        wall_dilate=0,
    )
    
    # Since watershed separates pieces by seed growth, it should successfully split them into 4 distinct slots!
    assert len(slots) == 4, f"Expected 4 slots, but got {len(slots)}"
    
    # Check that each slot has an area of roughly ~9000-10000 pixels
    for i, slot in enumerate(slots):
        area = np.count_nonzero(slot)
        assert 7000 < area < 11000, f"Slot {i} area is {area}, expected ~9000"

def test_components_to_slots_watershed_grid_seeds():
    # Similar 2x2 grid inside a box at the very edges
    line_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(line_mask, (0, 0), (199, 199), 255, 3)
    cv2.line(line_mask, (100, 0), (100, 200), 255, 3)
    cv2.line(line_mask, (0, 100), (200, 100), 255, 3)
    line_mask[95:105, 95:105] = 0
    
    # Run with explicit grid seeds rows=2, cols=2
    slots = components_to_slots(
        line_mask,
        min_slot_area=500,
        max_slot_area=20000,
        border_thickness=0,
        wall_dilate=0,
        rows=2,
        cols=2
    )
    
    assert len(slots) == 4, f"Expected 4 slots with grid seeds, but got {len(slots)}"

def test_import_board_from_photo_binary(tmp_path):
    from puzzle_recognition.auto_board_importer import import_board_from_photo
    
    # Create a mock binary image (black background, white grid lines)
    # The board corners are the image corners (0,0) to (199,199)
    img_path = tmp_path / "mock_binary_board.png"
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Draw outer box
    cv2.rectangle(img, (0, 0), (199, 199), (255, 255, 255), 3)
    # Draw horizontal and vertical white lines
    cv2.line(img, (100, 0), (100, 200), (255, 255, 255), 3)
    cv2.line(img, (0, 100), (200, 100), (255, 255, 255), 3)
    
    cv2.imwrite(str(img_path), img)
    
    # Run import_board_from_photo in binary mode
    config = import_board_from_photo(
        image_path=img_path,
        board_id="test_binary_board",
        rectified_size=(200, 200),
        min_slot_area=500,
        max_slot_area=20000,
        border_thickness=0,
        wall_dilate=0,
        output_root=tmp_path,
        binary=True,
    )
    
    # It should detect 4 slots!
    assert len(config["slots"]) == 4, f"Expected 4 slots, but got {len(config['slots'])}"

def test_import_board_from_photo_binary_autodetect(tmp_path):
    from puzzle_recognition.auto_board_importer import import_board_from_photo
    
    img_path = tmp_path / "mock_binary_board_auto.png"
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (199, 199), (255, 255, 255), 3)
    cv2.line(img, (100, 0), (100, 200), (255, 255, 255), 3)
    cv2.line(img, (0, 100), (200, 100), (255, 255, 255), 3)
    cv2.imwrite(str(img_path), img)
    
    # Run without specifying binary parameter (should autodetect as True)
    config = import_board_from_photo(
        image_path=img_path,
        board_id="test_binary_board_auto",
        rectified_size=(200, 200),
        min_slot_area=500,
        max_slot_area=20000,
        border_thickness=0,
        wall_dilate=0,
        output_root=tmp_path,
    )
    
    assert len(config["slots"]) == 4, f"Expected 4 slots under autodetect, but got {len(config['slots'])}"
