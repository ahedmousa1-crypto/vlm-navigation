"""Render World 3 on Modal GPUs, then finish and upload it there too (the PC can be off).

Steps (run from the repo root; work lives on the Modal volume "vln-render"):

  modal run tools/modal_render.py --step test                 # 5 views on one GPU; images copied home to compare
  modal run --detach tools/modal_render.py --step full        # render every pending view on many GPUs, then
                                                              # build_dataset --photo_fx + upload in batches
  modal run tools/modal_render.py --step status               # progress: views rendered, samples uploaded

"Pending" = every camera in the world minus the views already in dataset_campus_0/samples.jsonl
(the laptop's renders). The orchestrator runs on Modal, so --detach keeps it going after the
laptop disconnects. Rerunning --step full resumes: finished views are skipped.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
BLENDER_URL = "https://download.blender.org/release/Blender5.2/blender-5.2.2-linux-x64.tar.xz"
WORLD = "/repo/blender/world_campus/world.blend"
REPO = "ahd03/vln-synthetic-streets"
GPU = "L4"
CHUNK = 25            # views per GPU task
UPLOAD_VIEWS = 300    # views per uploaded shard
SKIP_DIRS = {"world", "world_oldtown", "__pycache__"}

EGL_VENDOR = '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}'

app = modal.App("vln-render")
volume = modal.Volume.from_name("vln-render", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget", "xz-utils", "libx11-6", "libxi6", "libxxf86vm1", "libxfixes3", "libxrender1", "libxext6",
        "libxkbcommon0", "libsm6", "libice6", "libgl1", "libegl1", "libgles2", "libglvnd0", "libvulkan1",
        "libwayland-client0", "libwayland-cursor0", "libwayland-egl1", "libxinerama1", "libxrandr2", "libxcursor1",
    )
    .run_commands(
        f"wget -q {BLENDER_URL} -O /tmp/blender.tar.xz",
        "mkdir -p /opt/blender && tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 && rm /tmp/blender.tar.xz",
        "mkdir -p /usr/share/glvnd/egl_vendor.d",
        f"echo '{EGL_VENDOR}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    .pip_install("numpy<2.3", "pillow", "pyarrow", "pyyaml", "datasets==4.1.1", "huggingface_hub>=0.35")
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "blender", "/repo/blender",
                   ignore=lambda p: bool(SKIP_DIRS & set(Path(p).parts)) or str(p).endswith(".blend1"))
    .add_local_dir(ROOT / "tools", "/repo/tools", ignore=["__pycache__"])
    .add_local_file(ROOT / "trace_format.py", "/repo/trace_format.py")
)
hf_secret = modal.Secret.from_name("huggingface-secret")


def _records(folder):
    path = os.path.join(folder, "samples.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _blender(args, log_path, script="/repo/blender/render_views.py"):
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ blender {' '.join(args)}\n")
        log.flush()
        return subprocess.run(["/opt/blender/blender", "--background", WORLD, "--python", script, "--"] + args,
                              cwd="/repo", stdout=log, stderr=subprocess.STDOUT).returncode


@app.function(image=image, gpu=GPU, volumes={"/vol": volume}, timeout=2 * 3600, retries=2, max_containers=20)
def render(chunk_id: str, views: list, out_root: str = "/vol/w3/chunks", skip_done: bool = True):
    """Render views into <out_root>/<chunk_id> (images, work/masks_rgb, samples.jsonl, render_log.csv).
    Renders into local disk, then copies to the volume; a Blender crash just reruns the views left."""
    final = os.path.join(out_root, chunk_id)
    work = f"/tmp/{chunk_id}"
    if os.path.isdir(final):
        shutil.copytree(final, work, dirs_exist_ok=True)
    os.makedirs(work, exist_ok=True)
    log_path = os.path.join(work, "blender.log")
    start = time.time()
    for attempt in range(3):
        done = {r["view"] for r in _records(work)}
        logged = set()
        log_csv = os.path.join(work, "render_log.csv")
        if os.path.isfile(log_csv):
            with open(log_csv, encoding="utf-8") as handle:
                logged = {line.split(",")[0] for line in handle}
        left = [v for v in views if v not in done and v not in logged] if skip_done else views
        if not left:
            break
        code = _blender(["--out", work, "--max_tasks", "3", "--views"] + left, log_path)
        skip_done = True
        if code == 0:
            break
    os.makedirs(out_root, exist_ok=True)
    shutil.copytree(work, final, dirs_exist_ok=True)
    volume.commit()
    rendered = {r["view"] for r in _records(work)}
    return {"chunk": chunk_id, "views": len(views), "rendered": len(rendered & set(views)),
            "seconds": round(time.time() - start)}


@app.function(image=image, gpu=GPU, volumes={"/vol": volume}, timeout=1800, secrets=[hf_secret])
def probe():
    """What the test needs to know: which GPU renderer Blender gets, and the secret's key names (not values)."""
    script = "/tmp/probe.py"
    with open(script, "w") as handle:
        handle.write("import gpu\nprint('GPU_RENDERER', gpu.platform.renderer_get(), '|', gpu.platform.vendor_get(), "
                     "'|', gpu.platform.backend_type_get())\n")
    out = subprocess.run(["/opt/blender/blender", "--background", "--python", script], capture_output=True, text=True)
    lines = [l for l in (out.stdout + out.stderr).splitlines() if "GPU_RENDERER" in l or "rror" in l]
    keys = sorted(k for k in os.environ if "HF" in k.upper() or "HUGGING" in k.upper() or "TOKEN" in k.upper())
    return {"blender": lines[-10:], "secret_keys": keys}


@app.function(image=image, volumes={"/vol": volume}, timeout=1800)
def list_cameras():
    script = "/tmp/cams.py"
    with open(script, "w") as handle:
        handle.write("import bpy\nc = bpy.data.collections.get('Views')\n"
                     "print('CAMS', ' '.join(sorted(o.name for o in c.objects if o.type == 'CAMERA')))\n")
    out = subprocess.run(["/opt/blender/blender", "--background", WORLD, "--python", script], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("CAMS "):
            return line.split()[1:]
    raise RuntimeError(out.stdout[-2000:] + out.stderr[-2000:])


def _hf_env():
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN", "TOKEN"):
        if os.environ.get(key):
            os.environ["HF_TOKEN"] = os.environ[key]
            return True
    return False


@app.function(image=image, volumes={"/vol": volume}, secrets=[hf_secret], timeout=12 * 3600, cpu=4, memory=16384)
def finish():
    """Move rendered chunks into one dataset folder UPLOAD_VIEWS views at a time, then build_dataset --photo_fx
    and upload_batch --delete (the same steps the laptop runner uses). Uploaded chunks are marked and skipped."""
    import sys

    if not _hf_env():
        raise RuntimeError("no Hugging Face token in the secret")
    volume.reload()
    ds = "/vol/w3/dataset"
    os.makedirs(os.path.join(ds, "images"), exist_ok=True)
    os.makedirs(os.path.join(ds, "work", "masks_rgb"), exist_ok=True)
    chunks_root = "/vol/w3/chunks"
    chunks = sorted(c for c in os.listdir(chunks_root) if not os.path.isfile(os.path.join(chunks_root, c, ".merged")))
    uploaded = 0
    staged = []
    log = os.path.join(ds, "finish.log")

    def run(cmd):
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(f"\n$ {' '.join(cmd)}\n")
            handle.flush()
            return subprocess.run(cmd, cwd="/repo", stdout=handle, stderr=subprocess.STDOUT).returncode

    def flush_batch():
        nonlocal uploaded, staged
        if not staged:
            return
        if run([sys.executable, "tools/build_dataset.py", "--dataset", ds, "--pending_only", "--photo_fx",
                "--max_previews", "0"]):
            raise RuntimeError("build_dataset failed; see finish.log")
        tag = f"b{int(time.time())}m"
        for attempt in range(5):
            if run([sys.executable, "tools/upload_batch.py", "new", "--repo", REPO, "--batch", tag, "--delete",
                    "--dataset", ds, "--prefix", "w3"]) == 0:
                break
            if attempt == 4:
                raise RuntimeError("upload failed 5 times; rerun --step finish")
            time.sleep(60 * 2 ** attempt)
        for chunk in staged:
            open(os.path.join(chunks_root, chunk, ".merged"), "w").close()
        uploaded += len(staged)
        print(f"finish: uploaded batch {tag} ({len(staged)} chunks)", flush=True)
        staged = []
        volume.commit()

    existing = {r["view"] for r in _records(ds)}
    staged_views = 0
    for chunk in chunks:
        src = os.path.join(chunks_root, chunk)
        recs = [r for r in _records(src) if r["view"] not in existing
                and os.path.isfile(os.path.join(src, "images", r["view"] + ".png"))]
        views = {r["view"] for r in recs}
        for view in views:
            shutil.move(os.path.join(src, "images", view + ".png"), os.path.join(ds, "images", view + ".png"))
            shutil.move(os.path.join(src, "work", "masks_rgb", view + ".png"),
                        os.path.join(ds, "work", "masks_rgb", view + ".png"))
        with open(os.path.join(ds, "samples.jsonl"), "a", encoding="utf-8") as handle:
            for r in recs:
                handle.write(json.dumps(r) + "\n")
        existing |= views
        staged.append(chunk)
        staged_views += len(views)
        if staged_views >= UPLOAD_VIEWS:
            flush_batch()
            staged_views = 0
    flush_batch()
    records = _records(ds)
    return {"chunks_uploaded": uploaded, "dataset_records": len(records), "dataset_views": len({r["view"] for r in records})}


@app.function(image=image, volumes={"/vol": volume}, timeout=24 * 3600)
def orchestrate(pending: list):
    chunks = [(f"c{i // CHUNK:03d}", pending[i:i + CHUNK]) for i in range(0, len(pending), CHUNK)]
    print(f"orchestrate: {len(pending)} views in {len(chunks)} chunks", flush=True)
    total = 0
    for result in render.starmap(chunks, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"orchestrate: chunk failed: {result}", flush=True)
            continue
        total += result["rendered"]
        print(f"orchestrate: {result}  total_rendered={total}", flush=True)
    print(f"orchestrate: rendering done, {total} views. Finishing + uploading.", flush=True)
    summary = finish.remote()
    print(f"orchestrate: {summary}", flush=True)
    return summary


@app.function(image=image, volumes={"/vol": volume}, timeout=600)
def status():
    root = "/vol/w3/chunks"
    rendered = merged = 0
    if os.path.isdir(root):
        for chunk in os.listdir(root):
            n = len({r["view"] for r in _records(os.path.join(root, chunk))})
            rendered += n
            merged += n if os.path.isfile(os.path.join(root, chunk, ".merged")) else 0
    ds = _records("/vol/w3/dataset")
    return {"views_rendered": rendered, "views_uploaded": merged, "samples_on_modal": len(ds)}


@app.local_entrypoint()
def main(step: str = "test"):
    if step == "test":
        print(probe.remote())
        views = ["cw_view_0610", "cw_view_0640", "cw_view_0700", "cw_view_2790", "cw_view_2795"]
        print(render.remote("test", views, out_root="/vol/w3/test", skip_done=False))
        dest = ROOT / "dataset_campus_0" / "modal_test"
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", "--force", "vln-render", "w3/test/test", str(dest)], check=True)
        print(f"test images in {dest}")
    elif step == "full":
        local = ROOT / "dataset_campus_0"
        done = {r["view"] for r in _records(str(local))}
        log_csv = local / "render_log.csv"
        if log_csv.is_file():   # views the laptop found no task for
            done |= {line.split(",")[0] for line in log_csv.read_text(encoding="utf-8").splitlines()
                     if ",no_task," in line}
        cams = list_cameras.remote()
        pending = [c for c in cams if c not in done]
        print(f"cameras={len(cams)} done_on_laptop={len(done)} pending={len(pending)}")
        call = orchestrate.spawn(pending)
        print(f"orchestrator running on Modal: {call.object_id}")
    elif step == "finish":
        print(finish.remote())
    elif step == "status":
        print(status.remote())
    else:
        raise SystemExit(f"unknown step {step}")
