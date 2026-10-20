"""Flag a rendered frame that is blank or the wrong size.

Plain Python (no Blender). A near-uniform image has a tiny pixel standard
deviation, which is what a failed or all-sky render looks like.

  python tools/qa_check.py blender/test_renders/scene_0000.png
"""

import argparse
import sys

import numpy as np
from PIL import Image

EXPECTED_SIZE = (1365, 1024)
# Near-uniform frames (empty sky, a missed render) sit well below this.
BLANK_STD_THRESHOLD = 5.0


def check_image(path):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    pixels = np.asarray(image, dtype=np.float64)
    std = float(pixels.std())
    resolution_ok = (width, height) == EXPECTED_SIZE
    blank = std < BLANK_STD_THRESHOLD
    return {
        "path": path,
        "width": width,
        "height": height,
        "std": std,
        "resolution_ok": resolution_ok,
        "blank_or_degenerate": blank,
    }


def main():
    parser = argparse.ArgumentParser(description="Check a render for blank frames and resolution.")
    parser.add_argument("images", nargs="+", help="PNG paths to check.")
    args = parser.parse_args()
    failed = False
    for path in args.images:
        result = check_image(path)
        print(
            f"{result['path']} "
            f"std={result['std']:.3f} "
            f"blank_or_degenerate={result['blank_or_degenerate']} "
            f"size={result['width']}x{result['height']} "
            f"resolution_ok={result['resolution_ok']}"
        )
        if result["blank_or_degenerate"] or not result["resolution_ok"]:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
