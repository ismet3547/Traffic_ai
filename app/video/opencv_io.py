"""OpenCV implementations for prerecorded files and annotated output."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from app.models import FramePacket, VideoInfo


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "OpenCV is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return cv2


class OpenCVVideoSource:
    def __init__(self, path: str | Path) -> None:
        cv2 = _cv2()
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"input video not found: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open input video: {self.path}")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            self._capture.release()
            raise RuntimeError("input video reports an invalid frame rate")
        self._info = VideoInfo(
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    @property
    def info(self) -> VideoInfo:
        return self._info

    def __iter__(self) -> Iterator[FramePacket]:
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield FramePacket(
                index=index,
                timestamp_seconds=index / self._info.fps,
                image=frame,
            )
            index += 1

    def close(self) -> None:
        self._capture.release()


class OpenCVVideoSink:
    def __init__(
        self,
        path: str | Path,
        info: VideoInfo,
        codec: str = "mp4v",
    ) -> None:
        cv2 = _cv2()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(
            str(self.path), fourcc, info.fps, (info.width, info.height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"could not create output video: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()
