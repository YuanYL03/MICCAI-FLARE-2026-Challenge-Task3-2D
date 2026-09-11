#!/usr/bin/env python3
"""Self-contained FLARE Task 3 inference for MedGemma 1.5 MoE-LoRA.

The raw-data discovery and submission writer intentionally operate directly on
the evaluator's mounted directory. Model prompting and generation reuse the
same ``mle.engine.infer`` implementation used for local validation inference.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from rich.console import Console

from mle.engine.infer import generate_predictions, load_model_and_processor
from mle.engine.preprocess import ALL_TASKS, load_json, row_from_record
from mle.vars import ExpConfig


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover_rows(data_root: Path, split_name: str, max_samples: int | None):
    rows: list[dict[str, Any]] = []
    originals: dict[str, dict[str, Any]] = {}
    ignored_names = {"predictions.json", "inference_details.json"}
    json_paths = sorted(
        path for path in data_root.rglob("*.json")
        if path.name not in ignored_names and ".cache" not in path.parts
    )
    for json_path in json_paths:
        records = load_json(json_path)
        for index, record in enumerate(records):
            row = row_from_record(
                record=record,
                json_path=json_path,
                input_root=data_root,
                fallback_split=split_name,
                selected_tasks=ALL_TASKS,
                assistant_content_style="string",
                allow_missing_images=False,
                include_unanswered=True,
                index=index,
                task_classifier=None,
            )
            if row is None:
                continue
            # The mounted directory determines the inference split; source JSON
            # metadata such as Split=val must not change official testing UIDs.
            old_uid = row["uid"]
            uid_parts = old_uid.split(":", 1)
            row["split"] = split_name
            row["uid"] = f"{split_name}:{uid_parts[1]}"
            rows.append(row)
            originals[row["uid"]] = deepcopy(record)
            if max_samples is not None and len(rows) >= max_samples:
                return rows, originals
    return rows, originals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--model_base", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--output_filename", default="predictions.json")
    parser.add_argument("--split_name", default="testing")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=896)
    parser.add_argument("--resize_mode", choices=("square", "longest", "none"), default="square")
    parser.add_argument("--max_images", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_jsonl = args.output_dir / f"{args.split_name}_predictions.jsonl"
    details_path = args.output_dir / "inference_details.json"
    submission_path = args.output_dir / args.output_filename
    if not args.resume and pred_jsonl.exists():
        pred_jsonl.unlink()

    rows, originals = discover_rows(args.data_root, args.split_name, args.max_samples)
    done_records = load_jsonl(pred_jsonl) if args.resume else []
    predictions = {str(record["uid"]): str(record.get("prediction", "")) for record in done_records}
    pending = [row for row in rows if row["uid"] not in predictions]
    print(f"Loaded {len(rows)} rows from {args.data_root}; pending={len(pending)}")
    print("Task counts:", dict(Counter(row["task_type"] for row in rows)))
    if not rows:
        raise RuntimeError(f"No FLARE JSON samples found under {args.data_root}")

    config = ExpConfig("medgemma15-docker-infer", "/workspace", "FLARE-MLLM-2D")
    infer_kwargs = {
        "model_name_or_path": args.model_base,
        "adapter_path": args.adapter_path,
        "local_files_only": True,
        "device": args.device,
        "load_in_4bit": args.load_4bit,
        "image_size": args.image_size,
        "resize_mode": args.resize_mode,
        "max_images_per_sample": args.max_images,
        "batch_size": 1,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "prompt_style": "default",
        "system_prompt": "You are an expert medical imaging assistant.",
        "log_every": 1,
    }
    model_bundle = load_model_and_processor(config, infer_kwargs)
    failures = []
    console = Console()
    for index, row in enumerate(pending, 1):
        try:
            record = generate_predictions(
                config, [row], console, infer_kwargs,
                model_bundle=model_bundle, split=args.split_name,
            )[0]
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            record = {"uid": row["uid"], "prediction": "", "task_type": row["task_type"], "case_id": row["case_id"]}
            failures.append({"uid": row["uid"], "error": f"CUDA OOM: {exc}"})
        except Exception as exc:
            record = {"uid": row["uid"], "prediction": "", "task_type": row["task_type"], "case_id": row["case_id"]}
            failures.append({"uid": row["uid"], "error": repr(exc)})
        append_jsonl(pred_jsonl, [record])
        predictions[row["uid"]] = str(record.get("prediction", ""))
        if index % 25 == 0 or index == len(pending):
            print(f"Completed {index}/{len(pending)} pending rows")

    submission = []
    for row in rows:
        sample = deepcopy(originals[row["uid"]])
        sample["Answer"] = predictions.get(row["uid"], "")
        submission.append(sample)
    write_json(submission_path, submission)
    write_json(details_path, {
        "data_root": str(args.data_root),
        "model_base": args.model_base,
        "adapter_path": args.adapter_path,
        "split": args.split_name,
        "num_rows": len(rows),
        "num_predictions": len(predictions),
        "task_counts": dict(Counter(row["task_type"] for row in rows)),
        "failures": failures,
        "submission": str(submission_path),
    })
    print(f"Done. submission={submission_path}")
    if failures:
        print(f"WARNING: {len(failures)} failures; see {details_path}")


if __name__ == "__main__":
    main()
