from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import AppConfig, load_config


def test_default_config_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    config = load_config(path)

    assert config.lanes.leftmost_lane_id == "left"
    assert config.rules.left_lane.occupancy_threshold_seconds == 8.0
    assert config.rules.left_lane.overtaking_clearance_mode == "contextual"
    assert config.traffic_context.history_seconds == 12.0
    assert config.right_lane_opportunity.front_gap_normalized == 0.08
    assert config.calibration.mode == "normalized"
    assert config.speed_estimation.enabled
    assert config.candidate_lifecycle.finalize_after_seconds is None
    assert config.candidate_lifecycle.evidence_settle_seconds == 2.0
    assert config.physical_measurements.require_independent_validation


def test_calibrated_example_config_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "calibrated_example.yaml"
    config = load_config(path)

    assert config.calibration.mode == "homography"
    assert len(config.calibration.image_points) == 4
    assert config.calibration.world_units == "meters"


def test_rule_lane_must_be_marked_leftmost() -> None:
    with pytest.raises(ValidationError, match="left-lane rule must reference"):
        AppConfig.model_validate(
            {
                "lanes": {
                    "lanes": [
                        {
                            "id": "a",
                            "label": "A",
                            "leftmost": True,
                            "polygon": [[0, 0], [1, 0], [1, 1]],
                        },
                        {
                            "id": "b",
                            "label": "B",
                            "polygon": [[0, 0], [1, 0], [1, 1]],
                        },
                    ]
                },
                "rules": {"left_lane": {"left_lane_id": "b"}},
            }
        )


def test_phase_one_style_config_receives_phase_two_defaults() -> None:
    config = AppConfig.model_validate(
        {
            "lanes": {
                "lanes": [
                    {
                        "id": "left",
                        "label": "Left",
                        "leftmost": True,
                        "polygon": [[0, 0], [1, 0], [1, 1]],
                    }
                ]
            },
            "rules": {
                "left_lane": {
                    "left_lane_id": "left",
                    "overtaking_clearance_mode": "none",
                }
            },
        }
    )

    assert config.traffic_context.history_seconds == 12.0
    assert config.lane_change.minimum_frames == 3
    assert config.rules.left_lane.overtaking_clearance_mode == "none"


def test_lane_order_validation_explains_context_migration() -> None:
    with pytest.raises(ValidationError, match="ordered left-to-right"):
        AppConfig.model_validate(
            {
                "lanes": {
                    "lanes": [
                        {
                            "id": "right",
                            "label": "Right",
                            "polygon": [[0, 0], [0.5, 0], [0.5, 1]],
                        },
                        {
                            "id": "left",
                            "label": "Left",
                            "leftmost": True,
                            "polygon": [[0.5, 0], [1, 0], [1, 1]],
                        },
                    ]
                },
                "rules": {"left_lane": {"left_lane_id": "left"}},
            }
        )
