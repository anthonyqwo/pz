import cv2
import json
import numpy as np

from puzzle_recognition.config import DetectorConfig
from puzzle_recognition.piece_labeler import label_pieces
from puzzle_recognition.piece_detector import detect_black_pieces


def draw_reasonable_piece(image, offset=(0, 0), color=(10, 10, 10)):
    ox, oy = offset
    contour = np.array(
        [
            [[30 + ox, 20 + oy]],
            [[95 + ox, 20 + oy]],
            [[95 + ox, 42 + oy]],
            [[132 + ox, 75 + oy]],
            [[95 + ox, 108 + oy]],
            [[95 + ox, 130 + oy]],
            [[30 + ox, 130 + oy]],
            [[30 + ox, 20 + oy]],
        ],
        dtype=np.int32,
    )
    cv2.drawContours(image, [contour], -1, color, thickness=cv2.FILLED)
    return contour


def test_detect_black_piece_ignores_internal_highlight():
    image = np.full((180, 240, 3), 210, dtype=np.uint8)
    draw_reasonable_piece(image, offset=(25, 20), color=(20, 20, 20))

    cv2.line(image, (80, 62), (162, 118), (245, 245, 245), thickness=5)
    image[12, 16] = (30, 30, 30)
    image[150, 215] = (25, 25, 25)

    pieces, masks = detect_black_pieces(
        image,
        DetectorConfig(
            gray_threshold=85,
            min_piece_area=1000,
            morphology_kernel_size=9,
            close_iterations=2,
        ),
    )

    assert len(pieces) == 1
    assert pieces[0]["area"] > 8000
    assert masks["kept_mask"][90, 125] == 255
    assert masks["kept_mask"][12, 16] == 0


def test_rejects_large_dark_background_component():
    image = np.full((220, 260, 3), 210, dtype=np.uint8)
    cv2.rectangle(image, (10, 175), (250, 215), (30, 30, 30), thickness=cv2.FILLED)
    draw_reasonable_piece(image, offset=(55, 15))

    pieces, masks = detect_black_pieces(image)

    assert len(pieces) == 1
    reasons = [reason for item in masks["debug"]["rejected_components"] for reason in item["reject_reasons"]]
    assert "area_too_large" in reasons or "bbox_too_large" in reasons


def test_rejects_border_dark_component():
    image = np.full((180, 220, 3), 210, dtype=np.uint8)
    cv2.rectangle(image, (0, 40), (70, 105), (15, 15, 15), thickness=cv2.FILLED)
    draw_reasonable_piece(image, offset=(75, 25), color=(15, 15, 15))

    pieces, masks = detect_black_pieces(image)

    assert len(pieces) == 1
    assert any("touches_border" in item["reject_reasons"] for item in masks["debug"]["rejected_components"])


def test_rejects_small_fabric_noise():
    image = np.full((160, 220, 3), 210, dtype=np.uint8)
    cv2.circle(image, (35, 35), 4, (20, 20, 20), thickness=cv2.FILLED)
    draw_reasonable_piece(image, offset=(65, 15), color=(15, 15, 15))

    pieces, masks = detect_black_pieces(image)

    assert len(pieces) == 1
    assert any("area_too_small" in item["reject_reasons"] for item in masks["debug"]["rejected_components"])


def test_keeps_reasonable_dark_piece_component():
    image = np.full((180, 220, 3), 210, dtype=np.uint8)
    draw_reasonable_piece(image, offset=(50, 25))

    pieces, masks = detect_black_pieces(image)

    assert len(pieces) == 1
    assert pieces[0]["bbox"][2] >= 80
    assert pieces[0]["mean_L"] <= 80
    assert masks["debug"]["kept_pieces"] == 1


def test_label_pieces_labels_only_kept_components(tmp_path):
    image = np.full((180, 220, 3), 210, dtype=np.uint8)
    cv2.rectangle(image, (0, 20), (55, 85), (15, 15, 15), thickness=cv2.FILLED)
    draw_reasonable_piece(image, offset=(65, 25))
    image_path = tmp_path / "input.png"
    cv2.imwrite(str(image_path), image)

    result = label_pieces(image_path, output_dir=tmp_path / "out", debug=True)

    assert result["piece_count"] == 1
    assert len(result["pieces"]) == 1
    assert result["pieces"][0]["piece_id"] == "piece_001"
    assert result["detector_debug"]["kept_pieces"] == 1
    with open(result["outputs"]["debug_json_path"], encoding="utf-8") as f:
        debug = json.load(f)
    assert any("touches_border" in item["reject_reasons"] for item in debug["rejected_components"])
