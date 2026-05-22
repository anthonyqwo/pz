from __future__ import annotations

import numpy as np


def parse_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(" ", "")
    if "x" not in normalized:
        raise ValueError("Size must use WIDTHxHEIGHT format, for example 2000x2000")
    width, height = normalized.split("x", 1)
    return int(width), int(height)


def parse_corners(value: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in value.split(";"):
        x, y = item.split(",", 1)
        points.append((float(x), float(y)))
    if len(points) != 4:
        raise ValueError("Exactly four corners are required")
    return points


def order_corners(corners: list[tuple[float, float]] | np.ndarray) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("corners must contain four (x, y) points")

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(4)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def rectify_board(image: np.ndarray, corners: list[tuple[float, float]], output_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    width, height = output_size
    src = order_corners(corners)
    dst = np.asarray(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(image, matrix, (width, height))
    return rectified, matrix

