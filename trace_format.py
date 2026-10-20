"""NaviTrace-compatible 2D trace coordinates (pixel space)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

COORDINATE_SPACE = "pixel"
ORIGIN = "top-left"

NAVITRACE_EMBODIMENTS = (
    "Human",
    "Legged Robot",
    "Wheeled Robot",
    "Bicycle",
)

# ground_truth[embodiment]: None | List[path] where path is List[[x, y], ...]
GroundTruthPath = List[List[float]]
GroundTruthPaths = List[GroundTruthPath]
GroundTruth = dict[str, Optional[GroundTruthPaths]]

PointArray = Union[Sequence[Sequence[float]], np.ndarray]


def empty_ground_truth() -> GroundTruth:
    return {name: None for name in NAVITRACE_EMBODIMENTS}


def _as_xy_array(points: PointArray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"trace must be shape (N, 2), got {arr.shape}")
    return arr


def _looks_normalized(arr: np.ndarray) -> bool:
    if arr.size == 0:
        return False
    return bool(np.all(arr >= 0.0) and np.all(arr <= 1.0) and np.max(arr) <= 1.0)


def project_trace_to_pixels(
    points: PointArray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Return trace points in pixel space; reject normalized [0, 1] inputs."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    arr = _as_xy_array(points)
    if _looks_normalized(arr):
        raise ValueError(
            "trace looks normalized to [0, 1]; store pixel coordinates "
            f"with origin {ORIGIN} instead"
        )
    return arr


def build_ground_truth_entry(
    alternative_paths: Iterable[PointArray],
) -> GroundTruthPaths:
    """One embodiment's value: one or more equally valid pixel paths."""
    out: GroundTruthPaths = []
    for path in alternative_paths:
        arr = _as_xy_array(path)
        out.append(arr.tolist())
    return out


def set_embodiment_paths(
    ground_truth: GroundTruth,
    embodiment: str,
    alternative_paths: Optional[Iterable[PointArray]],
) -> None:
    if embodiment not in NAVITRACE_EMBODIMENTS:
        raise KeyError(f"unknown embodiment: {embodiment}")
    if alternative_paths is None:
        ground_truth[embodiment] = None
        return
    ground_truth[embodiment] = build_ground_truth_entry(alternative_paths)
