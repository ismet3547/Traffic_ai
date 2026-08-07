from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import AppConfig, load_config


def test_default_config_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    config = load_config(path)

    assert config.lanes.leftmost_lane_id == "left"
    assert config.rules.left_lane.occupancy_threshold_seconds == 8.0
    assert config.rules.left_lane.overtaking_clearance_mode == "none"


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
