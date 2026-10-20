"""Package dataset/ as a Hugging Face dataset in the NaviTrace column layout, optionally upload it.

Columns match leggedrobotics/navitrace: sample_id, task, embodiments, image,
segmentation_mask, ground_truth, category, context, metadata. One difference:
segmentation_mask is stored as a grayscale PNG (class ids, see classes.json)
instead of a nested int list, which is ~100x smaller. An extra ``synthetic``
column keeps camera pose, world paths, and the target object.

  python tools/export_hf.py --dataset dataset                         # build + check locally only
  python tools/export_hf.py --dataset dataset --repo USER/NAME --push  # upload (needs `hf auth login`)
"""

import argparse
import json
import os
import sys

from datasets import Dataset, Features, Image, List, Value

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBODIMENTS = ("Bicycle", "Human", "Legged Robot", "Wheeled Robot")

PATHS = List(List(List(Value("float64"))))
FEATURES = Features({
    "sample_id": Value("string"),
    "task": Value("string"),
    "embodiments": List(Value("string")),
    "image": Image(),
    "segmentation_mask": Image(),
    "ground_truth": {name: PATHS for name in EMBODIMENTS},
    "category": List(Value("string")),
    "context": Value("string"),
    "metadata": {key: Value("string") for key in (
        "city", "country", "lighting_conditions", "natural_structured", "task_type", "urban_rural", "weather_conditions")},
    "synthetic": Value("string"),  # JSON: world seed, block, look, camera pose, world-space paths
})


def rows(root):
    with open(os.path.join(root, "samples.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            yield {
                "sample_id": record["sample_id"],
                "task": record["task"],
                "embodiments": record["embodiments"],
                "image": os.path.join(root, record["image"]),
                "segmentation_mask": os.path.join(root, record["segmentation_mask"]),
                "ground_truth": {name: record["ground_truth"].get(name) for name in EMBODIMENTS},
                "category": record["category"],
                "context": record["context"],
                "metadata": record["metadata"],
                "synthetic": json.dumps(record.get("synthetic", {})),
            }


def main():
    parser = argparse.ArgumentParser(description="Export dataset/ to Hugging Face format.")
    parser.add_argument("--dataset", default=os.path.join(REPO_ROOT, "dataset"))
    parser.add_argument("--repo", default="", help="Hub dataset repo, e.g. your-name/synthetic-navitrace.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    parser.add_argument("--push", action="store_true", help="Actually upload. Without it nothing leaves this machine.")
    args = parser.parse_args()
    root = os.path.abspath(args.dataset)

    missing = [p for r in rows(root) for p in (r["image"], r["segmentation_mask"]) if not os.path.isfile(p)]
    if missing:
        print(f"missing files ({len(missing)}), run tools/build_dataset.py first; e.g. {missing[0]}")
        return 1

    dataset = Dataset.from_list(list(rows(root)), features=FEATURES)
    print(dataset)
    print("example task:", dataset[0]["task"], "| Human paths:", len(dataset[0]["ground_truth"]["Human"]))
    local = os.path.join(root, "work", "hf_export")
    dataset.save_to_disk(local)
    print("saved_local", local)

    if args.push:
        if not args.repo:
            print("--push needs --repo USER/NAME")
            return 1
        dataset.push_to_hub(args.repo, split=args.split, private=args.private)
        print(f"pushed https://huggingface.co/datasets/{args.repo} split={args.split} private={args.private}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
