from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puzzle_recognition.auto_board_importer import import_board_from_photo
from puzzle_recognition.calibration import parse_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-import a black board with white jigsaw lines.")
    parser.add_argument("--image", required=True, help="Board photo path.")
    parser.add_argument("--board-id", required=True)
    parser.add_argument("--rectified-size", required=True, help="WIDTHxHEIGHT, for example 3000x2000")
    parser.add_argument("--dark-threshold", type=int, default=235, help="Value threshold for finding the low-saturation board area.")
    parser.add_argument("--white-threshold", type=int, default=150, help="Global threshold for white jigsaw lines.")
    parser.add_argument("--line-mode", choices=["simple_binary", "tophat_hsv", "adaptive", "hybrid"], default="adaptive")
    parser.add_argument("--tophat-kernel-size", type=int, default=17)
    parser.add_argument("--hsv-s-max", type=int, default=120)
    parser.add_argument("--hsv-v-min", type=int, default=100)
    parser.add_argument("--component-min-area", type=int, default=5)
    parser.add_argument("--component-max-area", type=int, default=0)
    parser.add_argument("--component-min-ratio", type=float, default=1.0)
    parser.add_argument("--adaptive-block-size", type=int, default=31)
    parser.add_argument("--adaptive-c", type=int, default=-8)
    parser.add_argument("--gap-limit", type=int, default=8, help="Maximum skeleton endpoint gap to connect.")
    parser.add_argument("--skeleton-close-iterations", type=int, default=1, help="Small MORPH_CLOSE iterations on skeleton.")
    parser.add_argument("--wall-dilate", type=int, default=2, help="Tiny dilation after skeleton repair, only for flood/connected region extraction.")
    parser.add_argument("--no-skeleton", action="store_true", help="Use clean binary line mask directly as walls.")
    parser.add_argument("--min-slot-area", type=int, default=1000)
    parser.add_argument("--max-slot-area", type=int)
    parser.add_argument("--border-thickness", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rows", type=int, help="Puzzle rows for grid seed-based watershed.")
    parser.add_argument("--cols", type=int, help="Puzzle columns for grid seed-based watershed.")
    parser.add_argument("--binary", action="store_true", default=None, help="Force binary mode (auto-detected by default).")
    parser.add_argument("--no-binary", dest="binary", action="store_false", help="Force disable binary mode.")
    args = parser.parse_args()

    config = import_board_from_photo(
        image_path=args.image,
        board_id=args.board_id,
        rectified_size=parse_size(args.rectified_size),
        dark_threshold=args.dark_threshold,
        white_threshold=args.white_threshold,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
        line_mode=args.line_mode,
        tophat_kernel_size=args.tophat_kernel_size,
        hsv_s_max=args.hsv_s_max,
        hsv_v_min=args.hsv_v_min,
        component_min_area=args.component_min_area,
        component_max_area=args.component_max_area,
        component_min_ratio=args.component_min_ratio,
        gap_limit=args.gap_limit,
        skeleton_close_iterations=args.skeleton_close_iterations,
        wall_dilate=args.wall_dilate,
        use_skeleton=not args.no_skeleton,
        min_slot_area=args.min_slot_area,
        max_slot_area=args.max_slot_area,
        border_thickness=args.border_thickness,
        debug=args.debug,
        rows=args.rows,
        cols=args.cols,
        binary=args.binary,
    )
    print(
        json.dumps(
            {
                "board_id": config["board_id"],
                "slot_count": len(config["slots"]),
                "board_config": f"data/boards/{config['board_id']}/board_config.json",
                "debug": config.get("debug"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
