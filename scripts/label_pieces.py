from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from puzzle_recognition.config import DetectorConfig
from puzzle_recognition.piece_labeler import label_pieces


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect black puzzle pieces and draw piece numbers on the original image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--gray-threshold", type=int, default=85)
    parser.add_argument("--min-piece-area", type=float, default=800.0)
    parser.add_argument("--max-piece-area", type=float, default=20000.0)
    parser.add_argument("--morphology-kernel-size", type=int, default=9)
    parser.add_argument("--close-iterations", type=int, default=2)
    parser.add_argument("--contour-epsilon-ratio", type=float, default=0.002)
    parser.add_argument("--min-bbox-width", type=int, default=20)
    parser.add_argument("--min-bbox-height", type=int, default=20)
    parser.add_argument("--max-bbox-width", type=int, default=220)
    parser.add_argument("--max-bbox-height", type=int, default=220)
    parser.add_argument("--aspect-ratio-min", type=float, default=0.35)
    parser.add_argument("--aspect-ratio-max", type=float, default=2.8)
    parser.add_argument("--extent-min", type=float, default=0.20)
    parser.add_argument("--extent-max", type=float, default=0.90)
    parser.add_argument("--solidity-min", type=float, default=0.35)
    parser.add_argument("--solidity-max", type=float, default=0.98)
    parser.add_argument("--max-piece-mean-L", type=float, default=80.0)
    parser.add_argument("--border-margin", type=int, default=8)
    parser.add_argument("--allow-border-components", action="store_true")
    parser.add_argument("--expected-max-pieces", type=int, default=100)
    parser.add_argument("--row-bucket", type=int, default=120, help="Y bucket size for top-to-bottom, left-to-right numbering.")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    detector = DetectorConfig(
        gray_threshold=args.gray_threshold,
        min_piece_area=args.min_piece_area,
        max_piece_area=args.max_piece_area,
        morphology_kernel_size=args.morphology_kernel_size,
        min_bbox_width=args.min_bbox_width,
        min_bbox_height=args.min_bbox_height,
        max_bbox_width=args.max_bbox_width,
        max_bbox_height=args.max_bbox_height,
        aspect_ratio_min=args.aspect_ratio_min,
        aspect_ratio_max=args.aspect_ratio_max,
        extent_min=args.extent_min,
        extent_max=args.extent_max,
        solidity_min=args.solidity_min,
        solidity_max=args.solidity_max,
        max_piece_mean_L=args.max_piece_mean_L,
        border_margin=args.border_margin,
        reject_border_components=not args.allow_border_components,
        expected_max_pieces=args.expected_max_pieces,
        close_iterations=args.close_iterations,
        contour_epsilon_ratio=args.contour_epsilon_ratio,
    )
    result = label_pieces(
        image_path=args.image,
        output_dir=args.output_dir,
        detector_config=detector,
        row_bucket=args.row_bucket,
        save_crops=args.save_crops,
        debug=args.debug,
    )
    print(
        json.dumps(
            {
                "image": result["image"],
                "piece_count": result["piece_count"],
                "outputs": result["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
