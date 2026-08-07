"""Replaceable video source and sink contracts."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np

from app.models import FramePacket, VideoInfo


class FrameSource(Protocol):
    @property
    def info(self) -> VideoInfo: ...

    def __iter__(self) -> Iterator[FramePacket]: ...

    def close(self) -> None: ...


class VideoSink(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...
