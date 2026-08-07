"""Video input/output abstractions."""

from .opencv_io import OpenCVVideoSink, OpenCVVideoSource
from .protocols import FrameSource, VideoSink

__all__ = ["FrameSource", "OpenCVVideoSink", "OpenCVVideoSource", "VideoSink"]
