"""Print the NaviTrace validation-split schema and one example.

Validation ``ground_truth`` traces use absolute pixel (x, y) with a top-left
origin, matching :mod:`trace_format` (not normalized [0, 1]).

  python tools/navitrace_peek.py
"""

import collections

import numpy as np
from datasets import load_dataset
from PIL import Image


def describe(value, depth=0):
    indent = "  " * depth
    if isinstance(value, Image.Image):
        return f"{indent}PIL.Image mode={value.mode} size={value.size}"
    if isinstance(value, np.ndarray):
        return f"{indent}ndarray dtype={value.dtype} shape={value.shape}"
    if isinstance(value, dict):
        lines = [f"{indent}dict len={len(value)}"]
        for key, item in value.items():
            lines.append(f"{indent}  {key}: {type(item).__name__}")
            lines.append(describe(item, depth + 2))
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        kind = type(value).__name__
        if not value:
            return f"{indent}{kind} len=0"
        head = value[0]
        if isinstance(head, (int, float)):
            arr = np.asarray(value)
            return f"{indent}{kind} len={len(value)} numeric shape={arr.shape} dtype={arr.dtype}"
        if isinstance(head, (list, tuple)) and head and isinstance(head[0], (int, float)):
            arr = np.asarray(value)
            return f"{indent}{kind} len={len(value)} numeric shape={arr.shape} dtype={arr.dtype}"
        lines = [f"{indent}{kind} len={len(value)} item0={type(head).__name__}"]
        lines.append(describe(head, depth + 1))
        return "\n".join(lines)
    text = repr(value)
    if len(text) > 400:
        text = text[:400] + "..."
    return f"{indent}{type(value).__name__} {text}"


def trace_array(example):
    for key in ("trace", "trajectory", "path", "points", "gt_trace"):
        if key in example:
            return key, np.asarray(example[key], dtype=np.float64)

    ground_truth = example.get("ground_truth")
    if not isinstance(ground_truth, dict):
        return None, None

    preferred = example.get("embodiments") or []
    keys = list(preferred) + [
        k for k in ("Human", "Legged Robot", "Wheeled Robot", "Bicycle") if k not in preferred
    ]
    for embodiment in keys:
        paths = ground_truth.get(embodiment)
        if paths:
            label = f"ground_truth[{embodiment!r}][0]"
            return label, np.asarray(paths[0], dtype=np.float64)
    return None, None


def iter_trace_arrays(example):
    ground_truth = example.get("ground_truth")
    if not isinstance(ground_truth, dict):
        return
    for embodiment, paths in ground_truth.items():
        if not paths:
            continue
        for path_index, path in enumerate(paths):
            arr = np.asarray(path, dtype=np.float64)
            if arr.size:
                yield f"ground_truth[{embodiment!r}][{path_index}]", arr


def main():
    dataset = load_dataset("leggedrobotics/navitrace", split="validation")
    print("num_rows", len(dataset))
    print("coordinate_space pixel (matches NaviTrace ground_truth)")
    print("columns", dataset.column_names)
    print("features")
    print(dataset.features)

    example = dataset[0]
    print("example_index 0")
    for key, value in example.items():
        print(f"field {key}")
        print(describe(value, 1))

    trace_key, trace = trace_array(example)
    if trace is not None:
        print("trace_key", trace_key)
        print("trace_point_count", int(trace.shape[0]) if trace.ndim >= 1 else 0)
        print("trace_shape", trace.shape)
        print("trace_min", trace.min(axis=0).tolist() if trace.size else None)
        print("trace_max", trace.max(axis=0).tolist() if trace.size else None)
        print("trace_full")
        print(trace.tolist())

    image = example.get("image")
    if isinstance(image, Image.Image):
        print("image_size", image.size)
        print("image_mode", image.mode)

    embodiments = collections.Counter()
    image_sizes = collections.Counter()
    point_counts = collections.Counter()
    coord_min = None
    coord_max = None
    for row in dataset:
        if "embodiments" in row and row["embodiments"]:
            for embodiment in row["embodiments"]:
                embodiments[embodiment] += 1
        elif "embodiment" in row:
            embodiments[row["embodiment"]] += 1
        img = row.get("image")
        if isinstance(img, Image.Image):
            image_sizes[img.size] += 1
        for _key, arr in iter_trace_arrays(row):
            point_counts[int(arr.shape[0])] += 1
            row_min = arr.min(axis=0)
            row_max = arr.max(axis=0)
            coord_min = row_min if coord_min is None else np.minimum(coord_min, row_min)
            coord_max = row_max if coord_max is None else np.maximum(coord_max, row_max)

    print("embodiment_values", dict(embodiments))
    print("embodiment_note", "counts are list entries per row (rows can list multiple embodiments)")
    print("image_sizes", dict(image_sizes))
    print("trace_point_counts", dict(point_counts))
    if coord_min is not None:
        print("split_coord_min", coord_min.tolist())
        print("split_coord_max", coord_max.tolist())


if __name__ == "__main__":
    main()
