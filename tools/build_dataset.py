"""Finish rendered views into dataset files and review previews. Plain Python (no Blender).

- work/masks_rgb/<view>.png -> masks/<view>.png (one class id per pixel, mode L)
- classes.json (id -> name, mask color)
- previews/<view>_<k>.png: the image with each Human path and the task drawn on it
- checks every record: image size, trace inside the image, at least 2 points

  python tools/build_dataset.py --dataset dataset
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "blender"))

from worldlib.classes import CLASSES  # noqa: E402

EXPECTED_SIZE = (1365, 1024)
PATH_COLORS = [(255, 40, 40), (40, 160, 255), (255, 200, 0)]


def rgb_mask_to_ids(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    palette = np.array([CLASSES[i][1] for i in sorted(CLASSES)], dtype=np.int32)
    ids = np.array(sorted(CLASSES), dtype=np.uint8)
    flat = rgb.reshape(-1, 3)
    # Nearest palette color; exact for the flat Workbench render, robust to +/-1 rounding.
    dist = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    nearest = dist.argmin(axis=1)
    worst = int(np.sqrt(dist[np.arange(len(flat)), nearest].max()))
    return ids[nearest].reshape(rgb.shape[:2]), worst


def draw_preview(image_path, record, out_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, path in enumerate(record["ground_truth"]["Human"] or []):
        color = PATH_COLORS[index % len(PATH_COLORS)]
        points = [tuple(p) for p in path]
        draw.line(points, fill=color, width=6, joint="curve")
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
        sx, sy = points[0]
        draw.ellipse((sx - 11, sy - 11, sx + 11, sy + 11), outline=(255, 255, 255), width=3)
    banner = f"{record['task']}   [{', '.join(record['category'])}]"
    draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
    draw.text((10, 9), banner, fill=(255, 255, 255))
    image.save(out_path)


def check_record(record, image_size):
    problems = []
    if image_size != EXPECTED_SIZE:
        problems.append(f"image size {image_size}")
    paths = record["ground_truth"].get("Human") or []
    if not paths:
        problems.append("no Human path")
    for path in paths:
        if len(path) < 2:
            problems.append("path with < 2 points")
        for x, y in path:
            if not (0 <= x < image_size[0] and 0 <= y < image_size[1]):
                problems.append(f"point outside image ({x}, {y})")
                break
    return problems


def main():
    parser = argparse.ArgumentParser(description="Finish masks, previews, and checks for rendered samples.")
    parser.add_argument("--dataset", default=os.path.join(REPO_ROOT, "dataset"))
    args = parser.parse_args()
    root = os.path.abspath(args.dataset)
    for sub in ("masks", "previews"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    with open(os.path.join(root, "classes.json"), "w", encoding="utf-8") as handle:
        json.dump({str(i): {"name": name, "mask_rgb": list(rgb)} for i, (name, rgb) in sorted(CLASSES.items())}, handle, indent=1)

    with open(os.path.join(root, "samples.jsonl"), encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    views = sorted({r["view"] for r in records})
    for view in views:
        src = os.path.join(root, "work", "masks_rgb", f"{view}.png")
        dst = os.path.join(root, "masks", f"{view}.png")
        if not os.path.isfile(src):
            print(f"missing_mask {view}")
            continue
        if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            continue
        ids, worst = rgb_mask_to_ids(src)
        Image.fromarray(ids, mode="L").save(dst)
        if worst > 3:
            print(f"mask_color_drift {view} max_distance={worst}")

    failed = 0
    per_view = {}
    for record in records:
        k = per_view.get(record["view"], 0)
        per_view[record["view"]] = k + 1
        image_path = os.path.join(root, record["image"])
        if not os.path.isfile(image_path):
            print(f"missing_image {record['image']}")
            failed += 1
            continue
        with Image.open(image_path) as image:
            size = image.size
        problems = check_record(record, size)
        if problems:
            failed += 1
            print(f"check_failed {record['sample_id']} {record['view']}: {'; '.join(problems)}")
        draw_preview(image_path, record, os.path.join(root, "previews", f"{record['view']}_{k}.png"))

    print(f"build_dataset records={len(records)} views={len(views)} failed_checks={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
