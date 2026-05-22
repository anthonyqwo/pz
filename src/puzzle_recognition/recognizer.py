from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .board_builder import load_board_config
from .calibration import rectify_board
from .config import DetectorConfig, MatcherConfig, OUTPUTS_DIR
from .io_utils import write_json
from .piece_detector import detect_black_pieces
from .shape_matcher import match_piece_to_board
from .visualization import draw_candidate_overlay, draw_match_results, draw_piece_contours, draw_top_match_overlay


def load_image(path: str | Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def prepare_image(
    image: np.ndarray,
    board_config: dict[str, Any],
    already_rectified: bool = False,
    corners: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, list[list[float]] | None]:
    if already_rectified:
        return image, None
    if corners is None:
        raise ValueError("corners are required unless already_rectified is true")
    rectified, matrix = rectify_board(image, corners, tuple(board_config["rectified_size"]))
    return rectified, matrix.tolist()


def recognize(
    image_path: str | Path,
    board_id: str,
    already_rectified: bool = False,
    corners: list[tuple[float, float]] | None = None,
    detector_config: DetectorConfig | None = None,
    matcher_config: MatcherConfig | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    import cv2

    board_config = load_board_config(board_id)
    image = load_image(image_path)
    rectified, transform_matrix = prepare_image(image, board_config, already_rectified, corners)

    pieces, masks = detect_black_pieces(rectified, detector_config)
    results = [match_piece_to_board(piece, board_config, matcher_config) for piece in pieces]

    payload: dict[str, Any] = {
        "board_id": board_id,
        "status": "ok",
        "image": str(image_path),
        "piece_count": len(pieces),
        "pieces": results,
    }
    if transform_matrix is not None:
        payload["transform_matrix"] = transform_matrix

    if debug:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        debug_dir = OUTPUTS_DIR / "debug" / run_id
        debug_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(debug_dir / "rectified.png"), rectified)
        cv2.imwrite(str(debug_dir / "black_mask.png"), masks["black_mask"])
        cv2.imwrite(str(debug_dir / "cleaned_mask.png"), masks["cleaned_mask"])
        cv2.imwrite(str(debug_dir / "piece_contour_overlay.png"), draw_piece_contours(rectified, pieces))
        cv2.imwrite(str(debug_dir / "top_candidates_overlay.png"), draw_candidate_overlay(rectified, results, board_config, top_n=10))
        cv2.imwrite(str(debug_dir / "best_match_overlay.png"), draw_top_match_overlay(rectified, results, board_config, top_n=1))
        cv2.imwrite(str(debug_dir / "top5_match_overlay.png"), draw_top_match_overlay(rectified, results, board_config, top_n=5))
        cv2.imwrite(str(debug_dir / "match_overlay.png"), draw_match_results(rectified, results, board_config))
        write_json(debug_dir / "result.json", payload)
        payload["debug"] = {
            "dir": str(debug_dir),
            "rectified_image_path": str(debug_dir / "rectified.png"),
            "black_mask_path": str(debug_dir / "black_mask.png"),
            "cleaned_mask_path": str(debug_dir / "cleaned_mask.png"),
            "piece_contour_overlay_path": str(debug_dir / "piece_contour_overlay.png"),
            "top_candidates_overlay_path": str(debug_dir / "top_candidates_overlay.png"),
            "best_match_overlay_path": str(debug_dir / "best_match_overlay.png"),
            "top5_match_overlay_path": str(debug_dir / "top5_match_overlay.png"),
            "match_overlay_path": str(debug_dir / "match_overlay.png"),
            "result_path": str(debug_dir / "result.json"),
        }

    return payload
