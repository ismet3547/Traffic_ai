"""Ultralytics YOLO detector adapter."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.config import DetectorConfig
from app.models import BoundingBox, Detection

LOGGER = logging.getLogger(__name__)


class UltralyticsDetector:
    def __init__(self, config: DetectorConfig) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        LOGGER.info("Loading YOLO model: %s", config.model_path)
        self._model = YOLO(config.model_path)
        self._config = config

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            source=frame,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            imgsz=self._config.image_size,
            classes=self._config.vehicle_class_ids,
            device=self._config.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        detections: list[Detection] = []
        for coords, confidence, class_id in zip(xyxy, confidences, class_ids):
            detections.append(
                Detection(
                    bbox=BoundingBox(*(float(value) for value in coords)),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=_class_name(result.names, int(class_id)),
                )
            )
        return detections


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    try:
        return str(names[class_id])
    except (IndexError, TypeError):
        return str(class_id)
