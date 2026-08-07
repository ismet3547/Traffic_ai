"""Reference-frame compatibility for configured road geometry."""

from __future__ import annotations

import math

from app.config import GeometryIntegrityConfig, LanesConfig
from app.models import FrameGeometry


def resolve_frame_geometry(
    width: int,
    height: int,
    lanes: LanesConfig,
    policy: GeometryIntegrityConfig,
) -> FrameGeometry:
    """Describe the only supported mapping from reference to runtime pixels."""

    reference_width = lanes.reference_width
    reference_height = lanes.reference_height
    if reference_width is None or reference_height is None:
        return FrameGeometry(
            width=width,
            height=height,
            aspect_ratio=width / height,
            reference_width=None,
            reference_height=None,
            reference_aspect_ratio=None,
            scale_x=None,
            scale_y=None,
            compatible=False,
            mapping_mode="unverified_reference_size",
            scaling_mode=lanes.scaling_mode,
            reason_codes=("LANE_REFERENCE_SIZE_UNAVAILABLE",),
        )

    scale_x = width / reference_width
    scale_y = height / reference_height
    exact = width == reference_width and height == reference_height
    uniform = math.isclose(scale_x, scale_y, rel_tol=1e-6, abs_tol=1e-9)
    uniform_allowed = (
        lanes.scaling_mode == "uniform" and policy.allow_uniform_frame_scaling
    )
    compatible = exact or (uniform and uniform_allowed)
    if exact:
        mapping_mode = "exact"
        reasons: tuple[str, ...] = ()
    elif compatible:
        mapping_mode = "uniform_scale"
        reasons = ("FRAME_UNIFORMLY_SCALED",)
    elif not uniform:
        mapping_mode = "aspect_ratio_mismatch"
        reasons = ("FRAME_ASPECT_RATIO_MISMATCH",)
    else:
        mapping_mode = "scaling_not_allowed"
        reasons = ("FRAME_SCALING_NOT_ALLOWED",)
    return FrameGeometry(
        width=width,
        height=height,
        aspect_ratio=width / height,
        reference_width=reference_width,
        reference_height=reference_height,
        reference_aspect_ratio=reference_width / reference_height,
        scale_x=scale_x,
        scale_y=scale_y,
        compatible=compatible,
        mapping_mode=mapping_mode,
        scaling_mode=lanes.scaling_mode,
        reason_codes=reasons,
    )
