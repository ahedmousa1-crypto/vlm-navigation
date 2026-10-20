"""Generate many street scenes via Blender, keeping only QA-passing renders.

Plain Python (no bpy). Invokes blender/gen_scene.py per seed, then checks
placement_pass in the QA JSON and runs tools/qa_check.py on the PNG.

  python tools/batch_run.py --target_count 50 --out_dir blender/batch_street --save_blend true
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_SCENE = os.path.join(REPO_ROOT, "blender", "gen_scene.py")
QA_CHECK = os.path.join(REPO_ROOT, "tools", "qa_check.py")
PYTHON = sys.executable


def find_blender():
    env = os.environ.get("BLENDER_EXE", "").strip()
    if env and os.path.isfile(env):
        return env
    return "blender"


def scene_basename(seed):
    return f"scene_{seed:04d}"


def scene_paths(out_dir, seed):
    base = os.path.join(out_dir, scene_basename(seed))
    return {
        "png": base + ".png",
        "qa": base + "_qa.json",
        "blend": base + ".blend",
    }


def run_gen_scene(blender_exe, out_dir, seed, save_blend):
    cmd = [
        blender_exe,
        "--background",
        "--python",
        GEN_SCENE,
        "--",
        "--seed",
        str(seed),
        "--out",
        os.path.abspath(out_dir),
    ]
    if save_blend:
        cmd.extend(["--save_blend", "true"])
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result


def read_placement_pass(qa_path):
    with open(qa_path, encoding="utf-8") as handle:
        data = json.load(handle)
    return bool(data.get("placement_pass", False)), data


def run_qa_check(png_path):
    result = subprocess.run(
        [PYTHON, QA_CHECK, png_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    reasons = []
    if result.returncode != 0:
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if line:
                reasons.append(f"qa_check: {line}")
        if not reasons:
            reasons.append(f"qa_check failed (exit {result.returncode})")
            if result.stderr:
                reasons.append(result.stderr.strip())
    return result.returncode == 0, reasons


def ensure_rejected_dir(out_dir):
    path = os.path.join(out_dir, "rejected")
    os.makedirs(path, exist_ok=True)
    return path


def move_to_rejected(paths, rejected_dir):
    for key in ("png", "qa", "blend"):
        src = paths[key]
        if os.path.isfile(src):
            shutil.move(src, os.path.join(rejected_dir, os.path.basename(src)))


def append_batch_log(log_path, seed, status, reasons):
    write_header = not os.path.isfile(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["seed", "status", "reasons"])
        writer.writerow([seed, status, "; ".join(reasons)])


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def print_progress(kept, attempted, target, start_time):
    elapsed = time.perf_counter() - start_time
    avg = elapsed / attempted if attempted else 0.0
    remaining_kept = target - kept
    eta = avg * remaining_kept if kept > 0 else 0.0
    print(
        f"progress kept={kept}/{target} attempted={attempted} "
        f"elapsed={format_duration(elapsed)} eta_remaining~{format_duration(eta)} "
        f"avg_per_scene={avg:.1f}s"
    )


def parse_save_blend(value):
    return str(value).lower() in ("true", "1", "yes", "on")


def main():
    parser = argparse.ArgumentParser(description="Batch-generate QA-filtered street scenes.")
    parser.add_argument("--target_count", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--save_blend",
        type=parse_save_blend,
        default=False,
        help="Pass --save_blend true to gen_scene.py when set.",
    )
    parser.add_argument(
        "--blender",
        default="",
        help="Blender executable (default: BLENDER_EXE env or 'blender' on PATH).",
    )
    args = parser.parse_args()

    if args.target_count < 1:
        print("target_count must be >= 1", file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rejected_dir = ensure_rejected_dir(out_dir)
    log_path = os.path.join(out_dir, "batch_log.csv")

    blender_exe = args.blender.strip() or find_blender()
    max_attempts = args.target_count * 3

    kept = 0
    attempted = 0
    rejected = 0
    seed = 0
    start_time = time.perf_counter()

    while kept < args.target_count and attempted < max_attempts:
        paths = scene_paths(out_dir, seed)
        gen = run_gen_scene(blender_exe, out_dir, seed, args.save_blend)
        attempted += 1

        if gen.returncode != 0:
            reasons = [f"gen_scene exit {gen.returncode}"]
            tail = (gen.stderr or gen.stdout or "").strip().splitlines()
            if tail:
                reasons.append(tail[-1][:500])
            move_to_rejected(paths, rejected_dir)
            append_batch_log(log_path, seed, "rejected", reasons)
            rejected += 1
            seed += 1
            if attempted % 10 == 0:
                print_progress(kept, attempted, args.target_count, start_time)
            continue

        failures = []
        if not os.path.isfile(paths["png"]):
            failures.append("missing PNG")
        if not os.path.isfile(paths["qa"]):
            failures.append("missing QA JSON")
        else:
            try:
                placement_ok, qa_data = read_placement_pass(paths["qa"])
                if not placement_ok:
                    floating = qa_data.get("floating_or_sunken", [])
                    visible = qa_data.get("visible_objects", 0)
                    failures.append(
                        f"placement_pass=false visible={visible} floating={len(floating)}"
                    )
            except (OSError, json.JSONDecodeError) as exc:
                failures.append(f"invalid QA JSON: {exc}")

        if not failures and os.path.isfile(paths["png"]):
            qa_ok, qa_reasons = run_qa_check(paths["png"])
            if not qa_ok:
                failures.extend(qa_reasons)

        if failures:
            move_to_rejected(paths, rejected_dir)
            append_batch_log(log_path, seed, "rejected", failures)
            rejected += 1
        else:
            append_batch_log(log_path, seed, "kept", [])
            kept += 1

        if attempted % 10 == 0:
            print_progress(kept, attempted, args.target_count, start_time)

        seed += 1

    total_elapsed = time.perf_counter() - start_time
    avg = total_elapsed / attempted if attempted else 0.0
    stopped_reason = (
        "target reached"
        if kept >= args.target_count
        else f"safety cap ({max_attempts} attempts)"
    )
    print(
        f"batch_done kept={kept} attempted={attempted} rejected={rejected} "
        f"stopped={stopped_reason} total_time={format_duration(total_elapsed)} "
        f"avg_per_scene={avg:.2f}s log={log_path}"
    )
    return 0 if kept >= args.target_count else 2


if __name__ == "__main__":
    sys.exit(main())
