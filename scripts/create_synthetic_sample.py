from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from puzzle_recognition.io_utils import read_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sample_dir = root / "data" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    board = read_json(root / "data" / "boards" / "board_001" / "slots_input.json")
    polygon = np.asarray(board["slots"][0]["polygon"], dtype=np.int32)

    image = np.full((400, 700, 3), 255, dtype=np.uint8)
    cv2.fillPoly(image, [polygon], (0, 0, 0))
    cv2.imwrite(str(sample_dir / "input_001.png"), image)

    print(sample_dir / "input_001.png")


if __name__ == "__main__":
    main()

