"""Finish rendered views into dataset files and review previews. Plain Python (no Blender).

- work/masks_rgb/<view>.png -> masks/<view>.png (one class id per pixel, mode L)
- classes.json (id -> name, mask color)
- previews/<view>_<k>.png: the image with each Human path and the task drawn on it
- checks every record: image size, trace inside the image, at least 2 points
- drops views whose image is black (a crashed render) so the batch runner renders them again
- --photo_fx: phone-photo look on each new image (exposure, white balance, contrast, saturation,
  vignette, slight blur, sensor noise, JPEG compression), recorded in the sample's "synthetic" field

  python tools/build_dataset.py --dataset dataset
"""

import argparse
import hashlib
import io
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "blender"))

from worldlib.classes import CLASSES  # noqa: E402

EXPECTED_SIZE = (1365, 1024)
PATH_COLORS = [(255, 40, 40), (40, 160, 255), (255, 200, 0)]


def rgb_mask_to_ids(path):
    """Class id per pixel. Exact color lookup first (the flat Workbench render is exact);
    only pixels off by rounding get a nearest-color search, which keeps memory small."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    flat = rgb.reshape(-1, 3)
    codes = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    palette = np.array([CLASSES[i][1] for i in sorted(CLASSES)], dtype=np.int32)
    ids = np.array(sorted(CLASSES), dtype=np.uint8)
    pal_codes = (palette[:, 0] << 16) | (palette[:, 1] << 8) | palette[:, 2]
    order = np.argsort(pal_codes)
    pos = np.clip(np.searchsorted(pal_codes[order], codes), 0, len(order) - 1)
    exact = pal_codes[order][pos] == codes
    out = ids[order][pos]
    worst = 0
    if not exact.all():
        off = flat[~exact]
        dist = ((off[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        nearest = dist.argmin(axis=1)
        out[~exact] = ids[nearest]
        worst = int(np.sqrt(dist[np.arange(len(off)), nearest].max()))
    return out.reshape(rgb.shape[:2]), worst


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


BLACK_MEAN = 3.0   # mean grey level below which an image counts as black


def photo_fx(path, view):
    """Phone-photo look, deterministic per view. Geometry is untouched, so traces and masks stay valid."""
    rng = random.Random(int(hashlib.sha1(view.encode()).hexdigest()[:8], 16))
    p = {
        "gain": round(rng.uniform(0.86, 1.14), 3),
        "wb_r": round(rng.uniform(0.95, 1.05), 3),
        "wb_b": round(rng.uniform(0.95, 1.05), 3),
        "contrast": round(rng.uniform(0.9, 1.15), 3),
        "saturation": round(rng.uniform(0.85, 1.2), 3),
        "vignette": round(rng.uniform(0.0, 0.28), 3),
        "blur": round(rng.choice((0.0, 0.0, rng.uniform(0.3, 0.9))), 2),
        "noise": round(rng.uniform(0.5, 4.0), 2),
        "jpeg": rng.randint(72, 93),
    }
    image = Image.open(path).convert("RGB")
    a = np.asarray(image, dtype=np.float32)
    a *= p["gain"]
    a[..., 0] *= p["wb_r"]
    a[..., 2] *= p["wb_b"]
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    a *= (1.0 - p["vignette"] * np.clip(r2 / 2.0, 0.0, 1.0))[..., None]
    image = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    image = ImageEnhance.Contrast(image).enhance(p["contrast"])
    image = ImageEnhance.Color(image).enhance(p["saturation"])
    if p["blur"] > 0:
        image = image.filter(ImageFilter.GaussianBlur(p["blur"]))
    a = np.asarray(image, dtype=np.float32)
    noise_rng = np.random.default_rng(rng.randrange(1 << 30))
    a += noise_rng.normal(0.0, p["noise"], a.shape[:2])[..., None]          # luma noise
    a += noise_rng.normal(0.0, p["noise"] * 0.5, a.shape)                   # chroma noise
    image = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=p["jpeg"])
    buffer.seek(0)
    Image.open(buffer).convert("RGB").save(path)
    return p


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
    parser.add_argument("--pending_only", action="store_true",
                        help="Only records whose image is still on disk (earlier batches were uploaded and deleted).")
    parser.add_argument("--photo_fx", action="store_true", help="Apply the phone-photo look to new images (World 3).")
    parser.add_argument("--max_previews", type=int, default=-1,
                        help="Draw at most this many previews (-1 = all, 0 = none); checks always run on every record.")
    args = parser.parse_args()
    root = os.path.abspath(args.dataset)
    for sub in ("masks", "previews"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    with open(os.path.join(root, "classes.json"), "w", encoding="utf-8") as handle:
        json.dump({str(i): {"name": name, "mask_rgb": list(rgb)} for i, (name, rgb) in sorted(CLASSES.items())}, handle, indent=1)

    with open(os.path.join(root, "samples.jsonl"), encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    all_records = records
    if args.pending_only:
        records = [r for r in records if os.path.isfile(os.path.join(root, r["image"]))]

    # Black images (a render that crashed): drop the view so the next batch renders it again.
    black = set()
    for view in sorted({r["view"] for r in records}):
        image_path = os.path.join(root, "images", f"{view}.png")
        if os.path.isfile(image_path):
            with Image.open(image_path) as image:
                if float(np.asarray(image.convert("L"), dtype=np.float32).mean()) < BLACK_MEAN:
                    black.add(view)
    if black:
        for view in black:
            for path in (os.path.join(root, "images", f"{view}.png"), os.path.join(root, "work", "masks_rgb", f"{view}.png"),
                         os.path.join(root, "masks", f"{view}.png")):
                if os.path.isfile(path):
                    os.remove(path)
            print(f"dropped_black {view}")
        all_records = [r for r in all_records if r["view"] not in black]
        records = [r for r in records if r["view"] not in black]

    if args.photo_fx:
        done = {}
        for record in records:
            syn = record.setdefault("synthetic", {})
            if syn.get("photo_fx") is not None:
                continue
            view = record["view"]
            if view not in done:
                done[view] = photo_fx(os.path.join(root, record["image"]), view)
            syn["photo_fx"] = done[view]
        print(f"photo_fx views={len(done)}")
    if black or args.photo_fx:
        tmp = os.path.join(root, "samples.jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            for record in all_records:
                handle.write(json.dumps(record) + "\n")
        os.replace(tmp, os.path.join(root, "samples.jsonl"))
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
    drawn = 0
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
        if args.max_previews < 0 or drawn < args.max_previews:
            draw_preview(image_path, record, os.path.join(root, "previews", f"{record['view']}_{k}.png"))
            drawn += 1

    print(f"build_dataset records={len(records)} views={len(views)} failed_checks={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
