from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import BOARDS_DIR
from .io_utils import read_json, write_json


def mask_from_polygon(polygon: list[list[float]] | list[tuple[float, float]], image_size: tuple[int, int]) -> np.ndarray:
    import cv2

    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.asarray(polygon, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("polygon must contain at least three [x, y] points")
    cv2.fillPoly(mask, [pts], 255)
    return mask


def contour_from_mask(mask: np.ndarray) -> np.ndarray:
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("slot mask does not contain a contour")
    return max(contours, key=cv2.contourArea)


def contour_features(contour: np.ndarray) -> dict[str, Any]:
    import cv2

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"]:
        center = [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]
    else:
        center = [float(x + w / 2), float(y + h / 2)]
    return {
        "area": area,
        "perimeter": perimeter,
        "bbox": [int(x), int(y), int(w), int(h)],
        "center": center,
        "aspect_ratio": float(w / h) if h else 0.0,
    }


def create_slot_from_polygon(slot_id: str, polygon: list[list[float]], image_size: tuple[int, int], output_dir: Path) -> dict[str, Any]:
    import cv2

    mask = mask_from_polygon(polygon, image_size)
    contour = contour_from_mask(mask)
    features = contour_features(contour)

    slots_dir = output_dir / "slots"
    slots_dir.mkdir(parents=True, exist_ok=True)
    mask_path = slots_dir / f"{slot_id}_mask.png"
    cv2.imwrite(str(mask_path), mask)

    return {
        "slot_id": slot_id,
        "mask_path": str(mask_path.relative_to(output_dir)).replace("\\", "/"),
        "polygon": polygon,
        "contour": contour.reshape(-1, 2).astype(int).tolist(),
        "rotation_mode": "any",
        "enabled": True,
        **features,
    }


def create_slot_from_mask(slot_id: str, source_mask_path: str | Path, image_size: tuple[int, int], output_dir: Path) -> dict[str, Any]:
    import cv2

    source = Path(source_mask_path)
    mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read slot mask: {source}")

    width, height = image_size
    if mask.shape[:2] != (height, width):
        raise ValueError(
            f"slot mask {source} has size {mask.shape[1]}x{mask.shape[0]}, "
            f"expected {width}x{height}"
        )

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contour = contour_from_mask(binary)
    features = contour_features(contour)

    slots_dir = output_dir / "slots"
    slots_dir.mkdir(parents=True, exist_ok=True)
    mask_path = slots_dir / f"{slot_id}_mask.png"
    if source.resolve() != mask_path.resolve():
        cv2.imwrite(str(mask_path), binary)
    else:
        cv2.imwrite(str(mask_path), binary)

    return {
        "slot_id": slot_id,
        "mask_path": str(mask_path.relative_to(output_dir)).replace("\\", "/"),
        "source_mask_path": str(source),
        "contour": contour.reshape(-1, 2).astype(int).tolist(),
        "rotation_mode": "any",
        "enabled": True,
        **features,
    }


def build_board_from_slots_json(board_id: str, slots_json_path: str | Path, rectified_size: tuple[int, int], output_root: Path = BOARDS_DIR) -> dict[str, Any]:
    payload = read_json(slots_json_path)
    board_dir = output_root / board_id
    board_dir.mkdir(parents=True, exist_ok=True)

    slots = []
    for item in payload.get("slots", []):
        if "polygon" in item:
            slots.append(create_slot_from_polygon(item["slot_id"], item["polygon"], rectified_size, board_dir))
        elif "mask_path" in item:
            mask_path = Path(item["mask_path"])
            if not mask_path.is_absolute():
                mask_path = Path(slots_json_path).parent / mask_path
            slots.append(create_slot_from_mask(item["slot_id"], mask_path, rectified_size, board_dir))
        else:
            raise ValueError(f"slot {item.get('slot_id', '<missing id>')} must contain polygon or mask_path")

    board_config = {
        "board_id": board_id,
        "version": 1,
        "rectified_size": [rectified_size[0], rectified_size[1]],
        "marker_type": payload.get("marker_type", "manual"),
        "slots": slots,
    }
    write_json(board_dir / "board_config.json", board_config)
    return board_config


def load_board_config(board_id: str, board_root: Path = BOARDS_DIR) -> dict[str, Any]:
    return read_json(board_root / board_id / "board_config.json")
