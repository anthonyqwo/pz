from __future__ import annotations

from typing import Any

import numpy as np

from .shape_matcher import contour_array


def draw_piece_contours(image: np.ndarray, pieces: list[dict[str, Any]]) -> np.ndarray:
    import cv2

    output = image.copy()
    for piece in pieces:
        contour = contour_array(piece["contour"])
        cv2.drawContours(output, [contour], -1, (0, 0, 255), 3)
        x, y, _, _ = piece["bbox"]
        cv2.putText(output, piece["piece_id"], (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return output


def draw_numbered_pieces(image: np.ndarray, pieces: list[dict[str, Any]]) -> np.ndarray:
    import cv2

    output = image.copy()
    for piece in pieces:
        contour = contour_array(piece["contour"])
        cx, cy = piece["center"]
        label = piece["piece_id"].replace("piece_", "")
        center = (int(round(cx)), int(round(cy)))

        cv2.drawContours(output, [contour], -1, (0, 255, 0), 4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.4
        thickness = 4
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 12
        x0 = center[0] - text_w // 2 - pad
        y0 = center[1] - text_h // 2 - pad
        x1 = center[0] + text_w // 2 + pad
        y1 = center[1] + text_h // 2 + baseline + pad

        cv2.rectangle(output, (x0, y0), (x1, y1), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(output, (x0, y0), (x1, y1), (0, 0, 0), 3)
        text_x = center[0] - text_w // 2
        text_y = center[1] + text_h // 2
        cv2.putText(output, label, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    return output


def draw_match_results(image: np.ndarray, results: list[dict[str, Any]], board_config: dict[str, Any]) -> np.ndarray:
    import cv2

    output = image.copy()
    slots_by_id = {slot["slot_id"]: slot for slot in board_config["slots"]}
    for result in results:
        slot_id = result.get("matched_slot_id")
        if not slot_id or slot_id not in slots_by_id:
            continue
        slot = slots_by_id[slot_id]
        contour = contour_array(slot["contour"])
        color = (0, 180, 0) if result["status"] == "confident" else (0, 180, 255)
        cv2.drawContours(output, [contour], -1, color, 3)
        x, y, _, _ = slot["bbox"]
        label = f"{slot_id} {result['confidence']:.2f}"
        cv2.putText(output, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return output


def draw_candidate_overlay(image: np.ndarray, results: list[dict[str, Any]], board_config: dict[str, Any], top_n: int = 10) -> np.ndarray:
    import cv2

    output = image.copy()
    slots_by_id = {slot["slot_id"]: slot for slot in board_config["slots"]}
    for result in results:
        debug = result.get("debug", {})
        candidates = debug.get("candidate_filter", {}).get("top_candidates_before_iou", [])[:top_n]
        for index, candidate in enumerate(candidates, start=1):
            slot = slots_by_id.get(candidate.get("slot_id"))
            if slot is None:
                continue
            contour = contour_array(slot["contour"])
            color = (255, 160, 0) if index > 1 else (0, 255, 255)
            cv2.drawContours(output, [contour], -1, color, 2)
            x, y, _, _ = slot["bbox"]
            cv2.putText(output, str(index), (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return output


def draw_top_match_overlay(image: np.ndarray, results: list[dict[str, Any]], board_config: dict[str, Any], top_n: int = 1) -> np.ndarray:
    import cv2

    output = image.copy()
    slots_by_id = {slot["slot_id"]: slot for slot in board_config["slots"]}
    palette = [(0, 220, 0), (0, 210, 255), (255, 120, 0), (220, 0, 255), (255, 0, 0)]
    for result in results:
        matches = result.get("debug", {}).get("matching", {}).get("top_matches", [])[:top_n]
        for index, match in enumerate(matches):
            slot = slots_by_id.get(match.get("slot_id"))
            if slot is None:
                continue
            contour = contour_array(slot["contour"])
            color = palette[index % len(palette)]
            cv2.drawContours(output, [contour], -1, color, 3 if index == 0 else 2)
            x, y, _, _ = slot["bbox"]
            label = f"{match.get('slot_id')} {float(match.get('iou') or 0):.2f}"
            cv2.putText(output, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output
