from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from puzzle_recognition.board_builder import build_board_from_slots_json
from puzzle_recognition.calibration import parse_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Build board config and slot masks from polygon JSON.")
    parser.add_argument("--board-id", required=True)
    parser.add_argument("--rectified-size", required=True, help="WIDTHxHEIGHT, for example 2000x2000")
    parser.add_argument("--slots-json", required=True)
    args = parser.parse_args()

    config = build_board_from_slots_json(
        board_id=args.board_id,
        slots_json_path=Path(args.slots_json),
        rectified_size=parse_size(args.rectified_size),
    )
    print(json.dumps({"board_id": config["board_id"], "slot_count": len(config["slots"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
