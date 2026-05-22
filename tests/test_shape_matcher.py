import numpy as np

from puzzle_recognition.board_builder import contour_from_mask, contour_features
from puzzle_recognition.config import MatcherConfig
from puzzle_recognition.shape_matcher import (
    classify_status,
    find_candidate_slots,
    mask_iou,
    match_piece_to_slot_iou,
    place_centered,
)


def test_mask_iou_identical():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    assert mask_iou(mask, mask) == 1.0


def test_place_centered():
    mask = np.ones((2, 2), dtype=np.uint8) * 255
    placed = place_centered(mask, (6, 6), [3, 3])
    assert placed.sum() == mask.sum()
    assert placed[2:4, 2:4].sum() == mask.sum()


def test_contour_features_from_irregular_mask():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    mask[9:12, 12:18] = 255
    contour = contour_from_mask(mask)
    features = contour_features(contour)
    assert features["area"] > 80
    assert features["bbox"] == [5, 5, 13, 10]


def piece_from_mask(piece_id: str, mask: np.ndarray) -> dict:
    contour = contour_from_mask(mask)
    return {"piece_id": piece_id, "contour": contour.reshape(-1, 2).astype(int).tolist(), **contour_features(contour)}


def slot_from_mask(slot_id: str, mask: np.ndarray, area_scale: float = 1.0) -> dict:
    contour = contour_from_mask(mask)
    features = contour_features(contour)
    return {
        "slot_id": slot_id,
        "mask_path": f"slots/{slot_id}_mask.png",
        "contour": contour.reshape(-1, 2).astype(int).tolist(),
        "enabled": True,
        **features,
        "area": features["area"] * area_scale,
    }


def rectangle_mask(size: int = 80, x: int = 20, y: int = 20, w: int = 30, h: int = 24) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    return mask


def test_candidate_filtering_keeps_area_diff_within_relaxed_tolerance():
    piece = piece_from_mask("piece_001", rectangle_mask())
    slots = [
        slot_from_mask("slot_001", rectangle_mask(), area_scale=1.30),
        slot_from_mask("slot_002", rectangle_mask(x=5, y=5, w=8, h=8), area_scale=1.0),
    ]

    candidates = find_candidate_slots(piece, slots, MatcherConfig(max_candidates=10, area_tolerance=0.40))

    assert [candidate["slot_id"] for candidate in candidates]
    assert any(candidate["slot_id"] == "slot_001" for candidate in candidates)
    assert all("area_diff_ratio" in candidate for candidate in candidates)


def test_candidate_filtering_fallbacks_when_too_few_candidates():
    piece = piece_from_mask("piece_001", rectangle_mask())
    slots = [slot_from_mask(f"slot_{index:03d}", rectangle_mask(x=5 + index, y=5, w=8, h=8)) for index in range(1, 7)]

    candidates = find_candidate_slots(piece, slots, MatcherConfig(max_candidates=6, area_tolerance=0.01, min_candidates_before_relax=5))

    assert len(candidates) >= 5
    assert any("relaxed_min_candidates" in candidate["debug_flags"] for candidate in candidates)


def test_iou_matching_translation_search_recovers_alignment():
    slot_mask = np.zeros((80, 80), dtype=np.uint8)
    piece_mask = np.zeros((80, 80), dtype=np.uint8)
    slot_mask[20:50, 25:55] = 255
    piece_mask[20:50, 25:55] = 255
    piece_mask[51:66, 23:38] = 255
    slot_mask[45:54, 43:56] = 255
    slot_mask[39:51, 16:25] = 255

    no_translation = match_piece_to_slot_iou(piece_mask, slot_mask, {"translation_search": [0], "angle_step_coarse": 360, "angle_refine_range": 0})
    with_translation = match_piece_to_slot_iou(
        piece_mask,
        slot_mask,
        {"translation_search": list(range(-10, 11)), "angle_step_coarse": 360, "angle_refine_range": 0},
    )

    assert with_translation["iou"] > no_translation["iou"]
    assert with_translation["dx"] != 0 or with_translation["dy"] != 0


def test_mirror_matching_can_match_mirrored_mask():
    slot_mask = np.zeros((80, 80), dtype=np.uint8)
    slot_mask[20:55, 25:40] = 255
    slot_mask[40:55, 40:58] = 255
    piece_mask = np.fliplr(slot_mask)

    without_mirror = match_piece_to_slot_iou(piece_mask, slot_mask, {"allow_mirror": False, "angle_step_coarse": 15})
    with_mirror = match_piece_to_slot_iou(piece_mask, slot_mask, {"allow_mirror": True, "angle_step_coarse": 15})

    assert with_mirror["iou"] > without_mirror["iou"]
    assert with_mirror["mirrored"] is True
    assert with_mirror["mirror_mode"] in {"mirror_x", "mirror_y"}


def test_decision_threshold_is_configurable():
    strict = MatcherConfig(confident_iou=0.90, ambiguous_iou=0.80, min_margin=0.05)
    relaxed = MatcherConfig(confident_iou=0.70, ambiguous_iou=0.55, min_margin=0.05)

    assert classify_status(0.72, 0.60, strict) == "rejected"
    assert classify_status(0.72, 0.60, relaxed) == "confident"


def test_match_result_debug_contains_top_matches_and_decision(tmp_path, monkeypatch):
    from puzzle_recognition import shape_matcher
    from puzzle_recognition.shape_matcher import match_piece_to_board

    slot_mask = rectangle_mask()
    monkeypatch.setattr(shape_matcher, "load_slot_mask", lambda board_id, slot: slot_mask)

    piece = piece_from_mask("piece_001", slot_mask)
    slot = slot_from_mask("slot_001", slot_mask)
    board_config = {"board_id": "board_test", "rectified_size": [80, 80], "slots": [slot]}

    result = match_piece_to_board(piece, board_config, MatcherConfig(max_candidates=5))

    assert result["debug"]["matching"]["top_matches"]
    assert result["debug"]["decision"]["status"] in {"confident", "ambiguous", "rejected"}
