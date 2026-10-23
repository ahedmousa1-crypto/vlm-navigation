"""Upload rendered samples to the Hub in batches, so the local disk never holds the whole dataset.

Modes:
  new      records whose image and mask exist locally and are not uploaded yet become
           data/{train,test}-p2-<batch>.parquet; with --delete the local images, masks
           and previews of those views are removed afterwards.
  retrace  rewrite the Phase 1 shards on the Hub with the ground truth now in
           samples.jsonl (denser traces from blender/retrace.py); images are kept as they are.
  card     write the dataset card (license, splits, description) from samples.jsonl.

  python tools/upload_batch.py new --repo ahd03/vln-synthetic-streets --batch b01 --delete
  python tools/upload_batch.py retrace --repo ahd03/vln-synthetic-streets
  python tools/upload_batch.py card --repo ahd03/vln-synthetic-streets

Uploaded sample ids are tracked in dataset/uploaded.json.
"""

import argparse
import json
import os
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_hf import EMBODIMENTS, FEATURES  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO_ROOT, "dataset")
LAYOUT = os.path.join(REPO_ROOT, "blender", "world", "layout.json")


def load_records():
    with open(os.path.join(DATASET, "samples.jsonl"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_registry():
    path = os.path.join(DATASET, "uploaded.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_registry(registry):
    path = os.path.join(DATASET, "uploaded.json")
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=1)
    os.replace(path + ".tmp", path)


def test_blocks():
    with open(LAYOUT, encoding="utf-8") as handle:
        return set(json.load(handle).get("test_blocks", []))


def split_of(record, blocks):
    return "test" if record.get("synthetic", {}).get("block") in blocks else "train"


def row_for(record, image_bytes=None, mask_bytes=None):
    if image_bytes is None:
        with open(os.path.join(DATASET, record["image"]), "rb") as handle:
            image_bytes = handle.read()
        with open(os.path.join(DATASET, record["segmentation_mask"]), "rb") as handle:
            mask_bytes = handle.read()
    return {
        "sample_id": record["sample_id"],
        "task": record["task"],
        "embodiments": record["embodiments"],
        "image": {"bytes": image_bytes, "path": os.path.basename(record["image"])},
        "segmentation_mask": {"bytes": mask_bytes, "path": os.path.basename(record["segmentation_mask"])},
        "ground_truth": {name: record["ground_truth"].get(name) for name in EMBODIMENTS},
        "category": record["category"],
        "context": record["context"],
        "metadata": {**record["metadata"], "task_type": record["metadata"].get("task_type", "Goal")},
        "synthetic": json.dumps(record.get("synthetic", {})),
    }


def write_shard(records, path, chunk=20):
    """Parquet shard written ``chunk`` samples at a time, so only a few images are in memory at once.

    Each chunk goes through datasets for encoding, so the file carries the same
    schema and Hugging Face feature metadata as push_to_hub output.
    """
    writer = None
    try:
        for start in range(0, len(records), chunk):
            table = Dataset.from_list([row_for(r) for r in records[start:start + chunk]], features=FEATURES).data.table
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
            del table
    finally:
        if writer is not None:
            writer.close()


def upload_new(args):
    api = HfApi()
    registry = load_registry()
    done = {sid for ids in registry.values() for sid in ids}
    blocks = test_blocks()
    pending = [r for r in load_records() if r["sample_id"] not in done
               and os.path.isfile(os.path.join(DATASET, r["image"]))
               and os.path.isfile(os.path.join(DATASET, r["segmentation_mask"]))]
    if not pending:
        print("upload_new nothing to upload")
        return 0
    views = set()
    for split in ("train", "test"):
        rows = [r for r in pending if split_of(r, blocks) == split]
        if not rows:
            continue
        shard = f"data/{split}-p2-{args.batch}.parquet"
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "shard.parquet")
            write_shard(rows, local)
            api.upload_file(path_or_fileobj=local, path_in_repo=shard, repo_id=args.repo, repo_type="dataset",
                            commit_message=f"Add {shard} ({len(rows)} samples)")
        registry[shard] = [r["sample_id"] for r in rows]
        save_registry(registry)
        views.update(r["view"] for r in rows)
        print(f"uploaded {shard} samples={len(rows)}")
    if args.delete:
        removed = 0
        for view in views:
            for rel in (f"images/{view}.png", f"masks/{view}.png", f"work/masks_rgb/{view}.png"):
                path = os.path.join(DATASET, rel)
                if os.path.isfile(path):
                    os.remove(path)
                    removed += 1
        previews = os.path.join(DATASET, "previews")
        if os.path.isdir(previews):
            for name in os.listdir(previews):
                if name.rsplit("_", 1)[0] in views:
                    os.remove(os.path.join(previews, name))
                    removed += 1
        print(f"deleted_local files={removed} views={len(views)}")
    return 0


def rewrite_columns(src, out, truth):
    """Copy a parquet shard row group by row group, replacing only ground_truth and synthetic.

    Image bytes are passed through untouched, so memory stays at about one row group.
    Returns (rows, rows_changed).
    """
    reader = pq.ParquetFile(src)
    schema = reader.schema_arrow
    gt_type = schema.field("ground_truth").type
    rows = changed = 0
    with pq.ParquetWriter(out, schema) as writer:
        for index in range(reader.num_row_groups):
            table = reader.read_row_group(index)
            ids = table.column("sample_id").to_pylist()
            old_gt = table.column("ground_truth").to_pylist()
            old_syn = table.column("synthetic").to_pylist()
            new_gt, new_syn = [], []
            for sid, gt, syn in zip(ids, old_gt, old_syn):
                record = truth.get(sid)
                if record is None:
                    new_gt.append(gt)
                    new_syn.append(syn)
                    continue
                fresh = {e: record["ground_truth"].get(e) for e in EMBODIMENTS}
                if fresh != gt:
                    changed += 1
                new_gt.append(fresh)
                new_syn.append(json.dumps(record.get("synthetic", {})))
            table = table.set_column(table.schema.get_field_index("ground_truth"), "ground_truth",
                                     pa.array(new_gt, type=gt_type))
            table = table.set_column(table.schema.get_field_index("synthetic"), "synthetic",
                                     pa.array(new_syn, type=pa.string()))
            writer.write_table(table)
            rows += len(ids)
            del table
    return rows, changed


def retrace(args):
    """Swap in the current ground truth for Phase 1 shards (the ones push_to_hub wrote)."""
    api = HfApi()
    truth = {r["sample_id"]: r for r in load_records()}
    files = [f for f in api.list_repo_files(args.repo, repo_type="dataset")
             if f.startswith("data/") and f.endswith(".parquet") and "-p2-" not in f]
    for name in files:
        with tempfile.TemporaryDirectory() as tmp:
            src = hf_hub_download(args.repo, name, repo_type="dataset", local_dir=tmp)
            out = os.path.join(tmp, "out.parquet")
            rows, changed = rewrite_columns(src, out, truth)
            if changed == 0:
                print(f"retraced {name} rows={rows} already up to date, skipped")
                continue
            api.upload_file(path_or_fileobj=out, path_in_repo=name, repo_id=args.repo, repo_type="dataset",
                            commit_message=f"Denser traces for {name}")
        print(f"retraced {name} rows={rows} updated={changed}")
    registry = load_registry()
    blocks = test_blocks()
    v1 = [r["sample_id"] for r in load_records() if r.get("synthetic", {}).get("retraced")]
    registry.setdefault("phase1", v1)
    save_registry(registry)
    print(f"registry phase1={len(registry['phase1'])} (test blocks {sorted(blocks)})")
    return 0


CARD = """
# VLN Synthetic Streets

Synthetic first-person street images with ground-truth navigation traces for
the **Human** embodiment, in the format of
[NaviTrace](https://huggingface.co/datasets/leggedrobotics/navitrace).
Rendered in Blender from a procedural town; code:
https://github.com/ahedmousa1-crypto/vlm-navigation

## Contents

| split | samples |
|---|---|
| train | {train} |
| test | {test} |

- One town (world seed 0), 20 blocks: streets, plazas, narrow streets, a construction
  site, a terrace, neighborhoods with houses, parks with fountains, ponds, bridges and
  playgrounds, and a skate park with a basketball court.
- Task types: Goal ({goal}) and Directions ({directions}), e.g. "Go to the front door of
  the yellow house" or "Go straight for about 20 meters, then cross the bridge, then turn
  left, then continue to the tree".
- Lighting: day, overcast, dusk, night and moonlight.
- The test split holds every camera standing in blocks {blocks}, so near-identical views
  never appear in both splits. A separate world will be generated later.

## Fields

Same columns as NaviTrace: `sample_id`, `task`, `embodiments`, `image`,
`segmentation_mask`, `ground_truth`, `category`, `context`, `metadata`.
Differences: `segmentation_mask` is a grayscale PNG of our own class ids, not
NaviTrace's class list; only `ground_truth["Human"]` is filled. Traces are pixel
`[x, y]` with a top-left origin, start at the bottom center of the image and have
6-15 points; tasks whose goal is reached by stairs or a ramp list both paths. The
extra `synthetic` column (JSON) holds the camera pose, world-space paths and the target
object.

## License and credits

Released under CC BY 4.0. Built with Blender. Cat and dog meshes: Quaternius,
Animal Pack Vol.2 (CC0). Optional sky/texture look: Poly Haven (CC0).
Format and category taxonomy follow NaviTrace.
"""


def write_card(args):
    records = load_records()
    registry = load_registry()
    uploaded = {sid for ids in registry.values() for sid in ids}
    live = [r for r in records if r["sample_id"] in uploaded]
    blocks = test_blocks()
    counts = {"train": 0, "test": 0}
    for r in live:
        counts[split_of(r, blocks)] += 1
    goal = sum(1 for r in live if r["metadata"].get("task_type", "Goal") == "Goal")
    meta = {
        "license": "cc-by-4.0",
        "pretty_name": "VLN Synthetic Streets",
        "tags": ["navigation", "vision-language-navigation", "synthetic", "blender", "navitrace"],
        "configs": [{"config_name": "default", "data_files": [
            {"split": "train", "path": "data/train-*"}, {"split": "test", "path": "data/test-*"}]}],
    }
    body = CARD.format(train=counts["train"], test=counts["test"], goal=goal, directions=len(live) - goal,
                       blocks=", ".join(sorted(blocks)))
    content = "---\n" + yaml.safe_dump(meta, sort_keys=False) + "---\n" + body
    HfApi().upload_file(path_or_fileobj=content.encode("utf-8"), path_in_repo="README.md", repo_id=args.repo,
                        repo_type="dataset", commit_message="Update dataset card")
    print(f"card train={counts['train']} test={counts['test']} goal={goal} directions={len(live) - goal}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Batch uploads to the Hugging Face Hub.")
    parser.add_argument("mode", choices=("new", "retrace", "card"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--batch", default="b00", help="Shard name suffix for mode 'new'.")
    parser.add_argument("--delete", action="store_true", help="Delete local files of uploaded views (mode 'new').")
    args = parser.parse_args()
    return {"new": upload_new, "retrace": retrace, "card": write_card}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
