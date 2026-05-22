from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
BOARDS_DIR = DATA_DIR / "boards"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class DetectorConfig:
    gray_threshold: int = 85
    min_piece_area: float = 800.0
    max_piece_area: float = 20_000.0
    morphology_kernel_size: int = 9
    min_bbox_width: int = 20
    min_bbox_height: int = 20
    max_bbox_width: int = 220
    max_bbox_height: int = 220
    aspect_ratio_min: float = 0.35
    aspect_ratio_max: float = 2.8
    extent_min: float = 0.20
    extent_max: float = 0.90
    solidity_min: float = 0.35
    solidity_max: float = 0.98
    max_piece_mean_L: float = 80.0
    border_margin: int = 8
    reject_border_components: bool = True
    expected_max_pieces: int = 100
    debug_label_min_area: float = 800.0
    close_iterations: int = 2
    contour_epsilon_ratio: float = 0.002
    pre_blur_ksize: int = 5
    contour_smooth_sigma: float = 1.2


@dataclass(frozen=True)
class MatcherConfig:
    max_candidates: int = 8
    min_candidates_before_relax: int = 5
    area_tolerance: float = 0.40
    perimeter_tolerance: float = 0.35
    aspect_ratio_tolerance: float = 0.60
    angle_step_coarse: int = 15
    angle_refine_range: int = 15
    angle_step_fine: int = 1
    translation_search: tuple[int, ...] = (-10, -5, 0, 5, 10)
    confident_iou: float = 0.70
    ambiguous_iou: float = 0.55
    min_margin: float = 0.05
    allow_mirror: bool = False
    confidence_iou_weight: float = 0.70
    confidence_area_weight: float = 0.15
    confidence_shape_weight: float = 0.10
    confidence_margin_weight: float = 0.05
    soft_mask_ksize: int = 5
    use_soft_iou: bool = True

    @property
    def coarse_angle_step(self) -> int:
        return self.angle_step_coarse

    @property
    def fine_angle_window(self) -> int:
        return self.angle_refine_range

    @property
    def fine_angle_step(self) -> int:
        return self.angle_step_fine
