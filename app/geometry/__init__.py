"""Geometry integrity and reference-frame safety."""

from .frame import resolve_frame_geometry
from .integrity import GeometryIntegrityPolicy

__all__ = ["GeometryIntegrityPolicy", "resolve_frame_geometry"]
