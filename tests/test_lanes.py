from app.config import LanesConfig
from app.lanes import LaneAssigner
from app.models import BoundingBox, TrackedVehicle
from tests.helpers import trusted_geometry


def _vehicle(track_id: int, bottom_center_x: float, bottom_y: float) -> TrackedVehicle:
    return TrackedVehicle(
        track_id=track_id,
        bbox=BoundingBox(
            bottom_center_x - 5, bottom_y - 10, bottom_center_x + 5, bottom_y
        ),
        confidence=0.9,
        class_id=2,
        class_name="car",
    )


def test_assigns_by_bottom_center_in_normalized_polygon() -> None:
    config = LanesConfig.model_validate(
        {
            "coordinate_space": "normalized",
            "lanes": [
                {
                    "id": "left",
                    "label": "Left",
                    "leftmost": True,
                    "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]],
                },
                {
                    "id": "right",
                    "label": "Right",
                    "polygon": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]],
                },
            ],
        }
    )
    assigner = LaneAssigner(config)
    observations = assigner.assign(
        [_vehicle(1, 25, 80), _vehicle(2, 75, 80), _vehicle(3, 120, 80)],
        frame_width=100,
        frame_height=100,
        geometry_integrity=trusted_geometry(),
    )

    assert [item.lane_id for item in observations] == ["left", "right", None]


def test_polygon_edge_counts_as_inside() -> None:
    config = LanesConfig.model_validate(
        {
            "coordinate_space": "pixels",
            "lanes": [
                {
                    "id": "left",
                    "label": "Left",
                    "leftmost": True,
                    "polygon": [[0, 0], [50, 0], [50, 100], [0, 100]],
                }
            ],
        }
    )
    observation = LaneAssigner(config).assign(
        [_vehicle(1, 50, 50)],
        frame_width=100,
        frame_height=100,
        geometry_integrity=trusted_geometry(),
    )[0]

    assert observation.lane_id == "left"
