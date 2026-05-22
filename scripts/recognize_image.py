from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from puzzle_recognition.calibration import parse_corners
from puzzle_recognition.config import DetectorConfig, MatcherConfig
from puzzle_recognition.recognizer import recognize


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize black puzzle pieces from an image.")
    parser.add_argument("--board-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--already-rectified", action="store_true")
    parser.add_argument("--corners", help='Four corners as "x1,y1;x2,y2;x3,y3;x4,y4"')
    parser.add_argument("--debug", action="store_true")
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
    parser.add_argument("--allow-mirror", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--area-tolerance", type=float, default=0.40)
    parser.add_argument("--confident-iou", type=float, default=0.70)
    parser.add_argument("--ambiguous-iou", type=float, default=0.55)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--angle-step-coarse", type=int, default=5)
    parser.add_argument("--angle-refine-range", type=int, default=5)
    parser.add_argument("--angle-step-fine", type=int, default=1)
    parser.add_argument("--translation-search", default="-10,-5,0,5,10")
    args = parser.parse_args()

    corners = parse_corners(args.corners) if args.corners else None
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
    translation_search = tuple(int(value.strip()) for value in args.translation_search.split(",") if value.strip())
    matcher = MatcherConfig(
        allow_mirror=args.allow_mirror,
        max_candidates=args.max_candidates,
        area_tolerance=args.area_tolerance,
        confident_iou=args.confident_iou,
        ambiguous_iou=args.ambiguous_iou,
        min_margin=args.min_margin,
        angle_step_coarse=args.angle_step_coarse,
        angle_refine_range=args.angle_refine_range,
        angle_step_fine=args.angle_step_fine,
        translation_search=translation_search,
    )
    result = recognize(
        image_path=args.image,
        board_id=args.board_id,
        already_rectified=args.already_rectified,
        corners=corners,
        detector_config=detector,
        matcher_config=matcher,
        debug=args.debug,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
