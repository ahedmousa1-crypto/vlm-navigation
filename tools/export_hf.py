"""Package dataset/ as a Hugging Face dataset in the NaviTrace column layout, optionally upload it.

Columns match leggedrobotics/navitrace: sample_id, task, embodiments, image,
segmentation_mask, ground_truth, category, context, metadata. One difference:
segmentation_mask is stored as a grayscale PNG (class ids, see classes.json)
instead of a nested int list, which is ~100x smaller. An extra ``synthetic``
column keeps camera pose, world paths, and the target object.

  python tools/export_hf.py --dataset dataset                         # build + check locally only
  python tools/export_hf.py --dataset dataset --repo USER/NAME --push  # upload (needs `hf auth login`)

Cameras in the test blocks (default b12, b22) form the test split; all others are train.
"""

import argparse
import json
import os
import sys

from datasets import Dataset, DatasetDict, Features, Image, List, Value
from huggingface_hub import DatasetCard

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBODIMENTS = ("Bicycle", "Human", "Legged Robot", "Wheeled Robot")
# World seed 0: one street block and one plaza block are held out for testing.
# A separate world will be generated later (see the dataset card).
TEST_BLOCKS = ("b12", "b22")

CARD_TEXT = """
# VLN Synthetic Streets

Synthetic first-person street images with ground-truth navigation traces for
the **Human** embodiment, in the format of
[NaviTrace](https://huggingface.co/datasets/leggedrobotics/navitrace).
Rendered in Blender from a procedural town; code:
https://github.com/ahedmousa1-crypto/vlm-navigation

## Splits

| split | samples |
|---|---|
| train | {train} |
| test | {test} |

All samples come from one world (seed 0). The test split holds every camera
standing in blocks {blocks}, so near-identical views never appear in both
splits. A separate world will be generated later.

## Fields

Same columns as NaviTrace: `sample_id`, `task`, `embodiments`, `image`,
`segmentation_mask`, `ground_truth`, `category`, `context`, `metadata`.
Differences: `segmentation_mask` is a grayscale PNG of our own class ids
(road, sidewalk, crosswalk, stairs, bench, car, cat, ...), not NaviTrace's
class list; only `ground_truth["Human"]` is filled. Traces are pixel `[x, y]`
with a top-left origin, starting at the bottom center of the image. The extra
`synthetic` column (JSON) holds the camera pose, world-space paths, and target
object.

## License and credits

Released under CC BY 4.0. Built with Blender. Cat and dog meshes: Quaternius,
Animal Pack Vol.2 (CC0). Optional sky/texture look: Poly Haven (CC0).
Format and category taxonomy follow NaviTrace.
"""

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
    parser.add_argument("--test_blocks", default=",".join(TEST_BLOCKS),
                        help="Blocks whose cameras form the test split (whole blocks, so near-duplicate views never straddle splits).")
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    parser.add_argument("--push", action="store_true", help="Actually upload. Without it nothing leaves this machine.")
    args = parser.parse_args()
    root = os.path.abspath(args.dataset)
    test_blocks = {b.strip() for b in args.test_blocks.split(",") if b.strip()}

    all_rows = list(rows(root))
    missing = [p for r in all_rows for p in (r["image"], r["segmentation_mask"]) if not os.path.isfile(p)]
    if missing:
        print(f"missing files ({len(missing)}), run tools/build_dataset.py first; e.g. {missing[0]}")
        return 1

    splits = {"train": [], "test": []}
    for row in all_rows:
        block = json.loads(row["synthetic"]).get("block")
        splits["test" if block in test_blocks else "train"].append(row)
    dataset = DatasetDict({name: Dataset.from_list(items, features=FEATURES) for name, items in splits.items()})
    print(dataset)
    print(f"test_blocks={sorted(test_blocks)} train={len(splits['train'])} test={len(splits['test'])}")
    if not args.push:
        # Local check only; skipped when pushing so a big release isn't stored twice on disk.
        local = os.path.join(root, "work", "hf_export")
        dataset.save_to_disk(local)
        print("saved_local", local)
        return 0

    if args.push:
        if not args.repo:
            print("--push needs --repo USER/NAME")
            return 1
        dataset.push_to_hub(args.repo, private=args.private)
        card = DatasetCard.load(args.repo, repo_type="dataset")
        card.data.license = "cc-by-4.0"
        card.data.pretty_name = "VLN Synthetic Streets"
        card.data.tags = ["navigation", "vision-language-navigation", "synthetic", "blender", "navitrace"]
        card.text = CARD_TEXT.format(train=len(splits["train"]), test=len(splits["test"]),
                                     blocks=", ".join(sorted(test_blocks)))
        card.push_to_hub(args.repo, repo_type="dataset")
        print(f"pushed https://huggingface.co/datasets/{args.repo} private={args.private}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
