"""Fine-tune and evaluate Qwen3-VL-8B on Human navigation traces, on Modal GPUs.

Steps (run from the repo root; results live on the Modal volume "vln-finetune"):

  modal run finetune/modal_app.py --step prepare                     # CPU: download model + data into the volume
  modal run finetune/modal_app.py --step check                       # CPU: verify prompt/labels (cents)
  modal run finetune/modal_app.py --step predict --run baseline      # GPU: untrained model on NaviTrace val (Human)
  modal run finetune/modal_app.py --step score --run baseline        # CPU: official NaviTrace score
  modal run finetune/modal_app.py --step train --run lora-v1 --epochs 2
  modal run finetune/modal_app.py --step predict --run lora-v1 --adapter lora-v1
  modal run finetune/modal_app.py --step score --run lora-v1

  modal volume get vln-finetune scores/ ./finetune/results/          # copy scores home

Training data: ahd03/vln-synthetic-streets (train split). Evaluation: leggedrobotics/navitrace
(validation split, Human embodiment) with the official prompt and scoring (navitrace_eval/).
"""

import json
import os
import sys
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
TRAIN_DATA = "ahd03/vln-synthetic-streets"
EVAL_DATA = "leggedrobotics/navitrace"
MAX_SIDE = 896          # images are resized so the longer side is 896 px (coordinates are normalized)
GPU = "A100-80GB"

app = modal.App("vln-finetune")
volume = modal.Volume.from_name("vln-finetune", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0", "torchvision==0.23.0", "transformers==4.57.1", "accelerate==1.10.1", "peft==0.17.1",
        "datasets==4.1.1", "huggingface_hub>=0.35", "pillow", "numpy<2.3", "scipy", "scikit-image==0.25.2", "pandas",
    )
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(Path(__file__).parent, "/root/finetune", ignore=["results", "__pycache__"])
)


def _setup():
    sys.path.insert(0, "/root/finetune")


def _resize(img):
    img = img.convert("RGB")
    scale = MAX_SIDE / max(img.size)
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    return img


def _load_model(adapter=""):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, f"/vol/adapters/{adapter}")
    return processor, model


def _assistant_start(tokenizer):
    return tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)


def _mask_prompt(input_ids, marker, pad_id):
    """Labels = input ids, with everything up to and including the assistant marker (and padding) ignored."""
    labels = input_ids.clone()
    for row in range(input_ids.shape[0]):
        ids = input_ids[row].tolist()
        start = None
        for k in range(len(ids) - len(marker), -1, -1):
            if ids[k:k + len(marker)] == marker:
                start = k + len(marker)
                break
        labels[row, : start if start is not None else len(ids)] = -100
    labels[input_ids == pad_id] = -100
    return labels


@app.function(image=image, volumes={"/vol": volume}, cpu=8, memory=65536, timeout=3 * 3600)
def prepare():
    """Download model weights and both datasets into the volume on a cheap CPU machine,
    so GPU time is not spent downloading."""
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID)
    train = load_dataset(TRAIN_DATA, split="train")
    val = load_dataset(EVAL_DATA, split="validation")
    volume.commit()
    return {"model": MODEL_ID, "train_samples": len(train),
            "navitrace_val": len(val), "navitrace_val_human": sum("Human" in e for e in val["embodiments"])}


@app.function(image=image, volumes={"/vol": volume}, cpu=4, memory=16384, timeout=1800)
def check():
    """Format one training example and show the text, token counts, and which tokens are trained."""
    _setup()
    from datasets import load_dataset
    from transformers import AutoProcessor
    from prompts import messages, trace_to_answer

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    ds = load_dataset(TRAIN_DATA, split="train", streaming=True)
    sample = next(iter(ds))
    img = sample["image"]
    answer = trace_to_answer(sample["ground_truth"]["Human"][0], *img.size)
    text = processor.apply_chat_template(messages(_resize(img), sample["task"], answer=answer), tokenize=False)
    prompt = processor.apply_chat_template(messages(_resize(img), sample["task"]), tokenize=False,
                                           add_generation_prompt=True)
    batch = processor(text=[text], images=[_resize(img)], return_tensors="pt")
    labels = _mask_prompt(batch["input_ids"], _assistant_start(processor.tokenizer), processor.tokenizer.pad_token_id)
    trained = processor.tokenizer.decode(batch["input_ids"][0][labels[0] != -100])
    volume.commit()
    return {
        "task": sample["task"],
        "answer": answer,
        "prompt_tail": prompt[-300:],
        "full_tail": text[-400:],
        "tokens": int(batch["input_ids"].shape[1]),
        "trained_text": trained,
        "image_size_after_resize": _resize(img).size,
    }


@app.function(image=image, volumes={"/vol": volume}, gpu=GPU, timeout=3 * 3600)
def predict(run, adapter="", limit=0, batch_size=8):
    """NaviTrace validation, Human embodiment: predictions in pixels, saved to /vol/preds/<run>.jsonl."""
    _setup()
    import torch
    from datasets import load_dataset
    from navitrace_eval.scoring import parse_trace
    from prompts import messages

    processor, model = _load_model(adapter)
    processor.tokenizer.padding_side = "left"
    model.eval()
    ds = load_dataset(EVAL_DATA, split="validation").remove_columns(["segmentation_mask"])
    indices = [i for i, emb in enumerate(ds["embodiments"]) if "Human" in emb]
    if limit:
        indices = indices[:limit]
    out_path = Path(f"/vol/preds/{run}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for start in range(0, len(indices), batch_size):
            rows = [ds[i] for i in indices[start:start + batch_size]]
            images = [_resize(r["image"]) for r in rows]
            texts = [processor.apply_chat_template(messages(img, r["task"]), tokenize=False, add_generation_prompt=True)
                     for img, r in zip(images, rows)]
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            replies = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, reply in zip(rows, replies):
                trace = parse_trace(reply) or []
                width, height = r["image"].size
                handle.write(json.dumps({
                    "sample_id": r["sample_id"], "embodiment": "Human", "category": r["category"],
                    "raw_response": reply, "prediction": [[int(x * width), int(y * height)] for x, y in trace],
                }) + "\n")
            print(f"predicted {min(start + batch_size, len(indices))}/{len(indices)}", flush=True)
    volume.commit()
    return {"run": run, "samples": len(indices), "path": str(out_path)}


def _score_rows(args):
    rows, = args
    import numpy as np
    from datasets import load_dataset
    from navitrace_eval.scoring import score as navitrace_score

    ds = load_dataset(EVAL_DATA, split="validation")
    index = {sid: i for i, sid in enumerate(ds["sample_id"])}
    out = []
    for row in rows:
        if not row["prediction"]:
            out.append((row["sample_id"], float("nan")))
            continue
        sample = ds[index[row["sample_id"]]]
        mask = np.array(sample["segmentation_mask"])
        out.append((row["sample_id"], float(navitrace_score(row["prediction"], sample["ground_truth"]["Human"], mask, "Human"))))
    return out


@app.function(image=image, volumes={"/vol": volume}, cpu=8, memory=65536, timeout=3 * 3600)
def score(run, workers=8):
    """Official NaviTrace score for /vol/preds/<run>.jsonl; summary saved to /vol/scores/<run>.json."""
    _setup()
    import math
    import multiprocessing

    from datasets import load_dataset

    load_dataset(EVAL_DATA, split="validation")  # make sure the cache is warm before forking workers
    rows = [json.loads(line) for line in open(f"/vol/preds/{run}.jsonl", encoding="utf-8")]
    chunks = [rows[k::workers] for k in range(workers)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        results = [item for part in pool.map(_score_rows, [(c,) for c in chunks]) for item in part]
    scores = dict(results)
    valid = [s for s in scores.values() if not math.isnan(s)]
    by_cat = {}
    for row in rows:
        s = scores[row["sample_id"]]
        for cat in row["category"]:
            by_cat.setdefault(cat, []).append(0.0 if math.isnan(s) else s)
    summary = {
        "run": run,
        "samples": len(rows),
        "valid_predictions": len(valid),
        "mean_score_valid": sum(valid) / max(1, len(valid)),
        "mean_score_invalid_as_0": sum(valid) / max(1, len(rows)),
        "by_category_invalid_as_0": {k: sum(v) / len(v) for k, v in sorted(by_cat.items())},
        "per_sample": scores,
    }
    Path("/vol/scores").mkdir(parents=True, exist_ok=True)
    with open(f"/vol/scores/{run}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    volume.commit()
    return {k: v for k, v in summary.items() if k != "per_sample"}


@app.function(image=image, volumes={"/vol": volume}, gpu=GPU, timeout=6 * 3600)
def train(run, epochs=2, lr=1e-4, rank=16, batch_size=4, grad_accum=4, limit=0):
    """LoRA on the language model (vision encoder frozen); adapter saved to /vol/adapters/<run>."""
    _setup()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    from prompts import messages, trace_to_answer

    processor, model = _load_model()
    processor.tokenizer.padding_side = "right"
    data = load_dataset(TRAIN_DATA, split="train")
    if limit:
        data = data.select(range(limit))
    marker = _assistant_start(processor.tokenizer)
    pad_id = processor.tokenizer.pad_token_id

    class Examples(torch.utils.data.Dataset):
        def __len__(self):
            return len(data)

        def __getitem__(self, i):
            row = data[i]
            img = row["image"]
            # Two-path (stairs/ramp) tasks: train on the first, cheapest path.
            return _resize(img), row["task"], trace_to_answer(row["ground_truth"]["Human"][0], *img.size)

    def collate(batch):
        images = [b[0] for b in batch]
        texts = [processor.apply_chat_template(messages(img, task, answer=answer), tokenize=False)
                 for img, task, answer in batch]
        enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
        enc["labels"] = _mask_prompt(enc["input_ids"], marker, pad_id)
        return enc

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=f"/vol/runs/{run}", per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
        logging_steps=5, save_strategy="no", report_to=[], remove_unused_columns=False, dataloader_num_workers=4,
    )
    trainer = Trainer(model=model, args=args, train_dataset=Examples(), data_collator=collate)
    trainer.train()
    model.save_pretrained(f"/vol/adapters/{run}")
    with open(f"/vol/adapters/{run}/train_log.json", "w", encoding="utf-8") as handle:
        json.dump({"args": {"epochs": epochs, "lr": lr, "rank": rank, "batch": batch_size * grad_accum,
                            "samples": len(data), "max_side": MAX_SIDE, "model": MODEL_ID},
                   "log": trainer.state.log_history}, handle, indent=1)
    volume.commit()
    return {"run": run, "samples": len(data), "final_loss": trainer.state.log_history[-1]}


@app.local_entrypoint()
def main(step: str = "check", run: str = "baseline", adapter: str = "", epochs: int = 2, limit: int = 0):
    if step == "prepare":
        print(json.dumps(prepare.remote(), indent=1))
    elif step == "check":
        print(json.dumps(check.remote(), indent=1))
    elif step == "predict":
        print(predict.remote(run, adapter=adapter, limit=limit))
    elif step == "score":
        print(json.dumps(score.remote(run), indent=1))
    elif step == "train":
        print(train.remote(run, epochs=epochs, limit=limit))
    else:
        raise SystemExit(f"unknown step {step}")
