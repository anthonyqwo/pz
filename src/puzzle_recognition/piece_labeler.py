from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DetectorConfig, OUTPUTS_DIR
from .io_utils import write_json
from .piece_detector import detect_black_pieces
from .recognizer import load_image
from .visualization import draw_numbered_pieces, draw_piece_contours


def sort_pieces_spatial(pieces: list[dict[str, Any]], row_bucket: int = 120) -> list[dict[str, Any]]:
    def key(piece: dict[str, Any]) -> tuple[int, float]:
        cx, cy = piece["center"]
        return (int(cy // row_bucket), float(cx))

    sorted_pieces = sorted(pieces, key=key)
    for index, piece in enumerate(sorted_pieces, start=1):
        piece["piece_id"] = f"piece_{index:03d}"
    return sorted_pieces


def label_pieces(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    detector_config: DetectorConfig | None = None,
    row_bucket: int = 120,
    save_crops: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    image = load_image(image_path)
    pieces, masks = detect_black_pieces(image, detector_config)
    pieces = sort_pieces_spatial(pieces, row_bucket=row_bucket)

    if output_dir is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        out_dir = OUTPUTS_DIR / "labeled_pieces" / run_id
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    numbered = draw_numbered_pieces(image, pieces)
    contours = draw_piece_contours(image, pieces)

    numbered_path = out_dir / "original_numbered.png"
    contours_path = out_dir / "pieces_contours.png"
    black_mask_path = out_dir / "black_mask.png"
    cleaned_mask_path = out_dir / "cleaned_mask.png"
    raw_dark_mask_path = out_dir / "raw_dark_mask.png"
    result_path = out_dir / "pieces.json"
    debug_json_path = out_dir / "piece_detector_debug.json"

    cv2.imwrite(str(numbered_path), numbered)
    cv2.imwrite(str(contours_path), contours)
    cv2.imwrite(str(black_mask_path), masks["black_mask"])
    cv2.imwrite(str(raw_dark_mask_path), masks["raw_dark_mask"])
    cv2.imwrite(str(cleaned_mask_path), masks["cleaned_mask"])

    debug_outputs: dict[str, str] = {}
    if debug:
        all_components_path = out_dir / "all_components_overlay.png"
        rejected_path = out_dir / "rejected_components_overlay.png"
        kept_path = out_dir / "kept_pieces_overlay.png"
        cv2.imwrite(str(all_components_path), masks["all_components_overlay"])
        cv2.imwrite(str(rejected_path), masks["rejected_components_overlay"])
        cv2.imwrite(str(kept_path), masks["kept_pieces_overlay"])
        write_json(debug_json_path, masks["debug"])
        debug_outputs = {
            "debug_json_path": str(debug_json_path),
            "all_components_overlay_path": str(all_components_path),
            "rejected_components_overlay_path": str(rejected_path),
            "kept_pieces_overlay_path": str(kept_path),
        }
    debug_summary = {
        "raw_components": masks["debug"]["raw_components"],
        "kept_pieces": masks["debug"]["kept_pieces"],
        "rejected_count": len(masks["debug"]["rejected_components"]),
        "diagnosis": masks["debug"]["diagnosis"],
    }

    crop_items: list[dict[str, Any]] = []
    if save_crops:
        crops_dir = out_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for piece in pieces:
            x, y, w, h = piece["bbox"]
            pad = 20
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(image.shape[1], x + w + pad)
            y1 = min(image.shape[0], y + h + pad)
            crop = image[y0:y1, x0:x1]

            mask_full = np.zeros(image.shape[:2], dtype=np.uint8)
            contour = np.asarray(piece["contour"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(mask_full, [contour], -1, 255, thickness=cv2.FILLED)
            crop_mask = mask_full[y0:y1, x0:x1]

            crop_path = crops_dir / f"{piece['piece_id']}.png"
            crop_mask_path = crops_dir / f"{piece['piece_id']}_mask.png"
            cv2.imwrite(str(crop_path), crop)
            cv2.imwrite(str(crop_mask_path), crop_mask)
            crop_items.append(
                {
                    "piece_id": piece["piece_id"],
                    "crop_path": str(crop_path),
                    "crop_mask_path": str(crop_mask_path),
                }
            )

    payload = {
        "image": str(image_path),
        "piece_count": len(pieces),
        "pieces": pieces,
        "outputs": {
            "output_dir": str(out_dir),
            "original_numbered_path": str(numbered_path),
            "pieces_contours_path": str(contours_path),
            "black_mask_path": str(black_mask_path),
            "raw_dark_mask_path": str(raw_dark_mask_path),
            "cleaned_mask_path": str(cleaned_mask_path),
            "pieces_json_path": str(result_path),
            "crops": crop_items,
            **debug_outputs,
        },
        "detector_debug": debug_summary if debug else None,
    }
    write_json(result_path, payload)
    return payload
