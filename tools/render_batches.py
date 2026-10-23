"""Render all pending views in batches: render -> masks/checks -> upload to the Hub -> delete locally.

Resumable: views that already have records are skipped (render_views --skip_done) and
uploaded samples are tracked in dataset/uploaded.json, so rerunning continues where it stopped.

  python tools/render_batches.py --repo ahd03/vln-synthetic-streets --batch_size 150
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(cmd)}\n")
        handle.flush()
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    return result.returncode


def count_records():
    path = os.path.join(REPO_ROOT, "dataset", "samples.jsonl")
    if not os.path.isfile(path):
        return 0, set()
    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return len(records), {r["view"] for r in records}


def main():
    parser = argparse.ArgumentParser(description="Render pending views in upload-and-delete batches.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--batch_size", type=int, default=150)
    parser.add_argument("--blender", default=os.environ.get("BLENDER_EXE", DEFAULT_BLENDER))
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--log", default=os.path.join(REPO_ROOT, "dataset", "render_batches.log"))
    args = parser.parse_args()
    world = os.path.join(REPO_ROOT, "blender", "world", "world.blend")

    for index in range(args.max_batches):
        before, views_before = count_records()
        start = time.perf_counter()
        code = run([args.blender, "--background", world, "--python", "blender/render_views.py", "--",
                    "--out", "dataset", "--skip_done", "--limit", str(args.batch_size)], args.log)
        after, views_after = count_records()
        new_views = len(views_after - views_before)
        print(f"batch {index}: render exit={code} new_views={new_views} samples={before}->{after} "
              f"({time.perf_counter() - start:.0f}s)", flush=True)
        if new_views == 0:
            break
        if run([sys.executable, "tools/build_dataset.py", "--dataset", "dataset", "--pending_only", "--max_previews", "0"], args.log):
            print("build_dataset failed; see log", flush=True)
            return 1
        tag = f"b{int(time.time())}"
        if run([sys.executable, "tools/upload_batch.py", "new", "--repo", args.repo, "--batch", tag, "--delete"],
               args.log):
            print("upload failed; local files kept, rerun to retry", flush=True)
            return 1
        print(f"batch {index}: uploaded as {tag}", flush=True)
    run([sys.executable, "tools/upload_batch.py", "card", "--repo", args.repo], args.log)
    total, views = count_records()
    print(f"render_batches_done samples={total} views={len(views)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
