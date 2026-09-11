import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rich.console import Console

from mle.vars import ExpConfig

try:
    from datasets import load_from_disk
except Exception:  # pragma: no cover
    load_from_disk = None


TASK_METRIC = {
    "disease_diagnosis_classification": "balanced_accuracy",
    "multi_label_classification": "f1_score",
    "detection": "f1_iou_0.5",
    "cell_counting": "mean_absolute_error",
    "regression": "mean_absolute_error",
    "report_generation": "green_score",
}

ALL_EVAL_SPLITS = ("testing", "validation_public", "validation_hidden")
MODEL_ID = "google/medgemma-1.5-4b-it"
GROUND_TRUTH_KEYS = ("raw_answer", "answer", "target", "reference", "label", "ground_truth")
PREDICTION_KEYS = ("prediction", "pred", "answer", "output", "response", "report", "text")
EMPTY_LABELS = {"", "none", "no finding", "no findings", "n/a", "na", "null", "[]"}
BOX_FIELDS = ("bbox", "bboxes", "box", "boxes", "bounding_box", "bounding_boxes", "detections")

TASK_ALIASES = {
    "classification": "disease_diagnosis_classification",
    "cls": "disease_diagnosis_classification",
    "disease diagnosis classification": "disease_diagnosis_classification",
    "multi-label classification": "multi_label_classification",
    "multi label classification": "multi_label_classification",
    "multi_label": "multi_label_classification",
    "multilabel": "multi_label_classification",
    "instance detection": "detection",
    "instance_detection": "detection",
    "count": "cell_counting",
    "counting": "cell_counting",
    "cell counting": "cell_counting",
    "report": "report_generation",
    "report generation": "report_generation",
}

TASK_INSTRUCTIONS = {
    "disease_diagnosis_classification": (
        "You are given a medical image. Answer the classification question using only the provided image. "
        "If options are provided, return only the correct option letter or class label."
    ),
    "multi_label_classification": (
        "You are given a medical image. Identify all applicable findings or labels. "
        "Return labels separated by semicolons. Return an empty string if none apply."
    ),
    "report_generation": (
        "You are given a medical image. Generate a concise radiology-style report with relevant findings and impression."
    ),
    "detection": "You are given a medical image. Return detections exactly in the coordinate format requested by the question.",
    "cell_counting": "You are given a medical image. Return only the integer count requested.",
    "regression": "You are given a medical image. Return only the requested numeric measurement.",
}


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str | None = None


def evaluate(
    config: ExpConfig,
    tasks: Sequence[str],
    use_wandb: bool,
    smoke_test: bool,
    *,
    console: Console = Console(),
    **kwargs,
) -> None:
    """
    This is a template entrypoint for evaluation. You MUST NOT change its signature, but you may add functions and
    classes to this file.

    All your logs MUST be sent to the provided console. Your implementation MUST support WandB logging and it MUST ONLY
    be enabled if :param use_wandb is `True`.

    :param config: experiment configuration
    :param tasks: the tasks to evaluate on
    :param use_wandb: whether to use wandb for logging
    :param smoke_test: whether to run in smoke test mode
    :param console: the console for logging
    :param kwargs: custom arguments
    """
    kwargs = dict(kwargs)
    if smoke_test:
        console.print("Smoke test mode: limiting evaluation rows and skipping heavy report scorers by default.")
        kwargs.setdefault("max_samples", 4)
        kwargs.setdefault("green_batch_size", 1)
        kwargs.setdefault("green_max_length", 1024)
        kwargs.setdefault("skip_green_score", True)
        kwargs.setdefault("skip_crimson_score", True)

    selected_tasks = normalize_task_list(tasks or kwargs.get("tasks") or TASK_METRIC)
    splits = resolve_eval_splits(kwargs)
    output_dir = Path(kwargs.get("eval_output_dir") or kwargs.get("output_dir") or Path(config.output_dir) / f"{config.experiment_name}-eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = start_evaluate_wandb_run(config, output_dir, selected_tasks, splits, kwargs) if use_wandb else None
    completed = False
    split_results = {}
    try:
        if wandb_run is not None:
            wandb_run.log({"eval/status": "started", "eval/num_splits": len(splits)})
        for split in splits:
            if wandb_run is not None:
                wandb_run.log({f"eval/{split}/status": "started"})
            split_results[split] = evaluate_one_split(config, selected_tasks, split, output_dir, console, kwargs)
            if wandb_run is not None:
                wandb_run.log({
                    f"eval/{split}/status": "completed",
                    f"eval/{split}/num_predictions": split_results[split]["num_predictions"],
                })

        if len(splits) == 1:
            results = split_results[splits[0]]
            scores_payload = results["metrics"]
        else:
            results = aggregate_split_results(split_results)
            results["splits"] = splits
            results["model_variant"] = "predictions_file"
            scores_payload = {
                "mean": results["mean_metrics"],
                "per_split": {split: payload["metrics"] for split, payload in split_results.items()},
            }

        scores_path = Path(kwargs.get("scores_json") or output_dir / "scores.json")
        details_path = Path(kwargs.get("details_json") or output_dir / "details.json")
        write_json(scores_path, scores_payload)
        write_json(details_path, results)
        print_summary(results, console)
        console.print(f"Saved metric summary to {scores_path}")
        console.print(f"Saved detailed evaluation to {details_path}")

        if wandb_run is not None:
            if len(splits) == 1:
                wandb_run.log({f"eval/{key}": value for key, value in results["metrics"].items() if isinstance(value, (int, float))})
            else:
                wandb_run.log({f"eval_mean/{key}": value for key, value in results["mean_metrics"].items() if isinstance(value, (int, float))})
            wandb_run.save(str(scores_path))
            wandb_run.save(str(details_path))
            wandb_run.log({"eval/status": "completed"})
        completed = True
    finally:
        if wandb_run is not None:
            if not completed:
                wandb_run.log({"eval/status": "failed"})
            wandb_run.finish(exit_code=0 if completed else 1)


def start_evaluate_wandb_run(config: ExpConfig, output_dir: Path, selected_tasks: Sequence[str], splits: Sequence[str], kwargs: Mapping[str, Any]):
    import wandb

    wandb_dir = Path(kwargs.get("wandb_dir") or output_dir / "wandb")
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_MODE", str(kwargs.get("wandb_mode", "online")))
    os.environ.setdefault("WANDB_PROJECT", str(kwargs.get("wandb_project") or os.environ.get("WANDB_PROJECT", "medgemma15-flare-mllm-2d")))
    if kwargs.get("wandb_entity"):
        os.environ.setdefault("WANDB_ENTITY", str(kwargs["wandb_entity"]))
    if kwargs.get("wandb_tags"):
        os.environ.setdefault("WANDB_TAGS", str(kwargs["wandb_tags"]))
    os.environ.pop("WANDB_DISABLED", None)

    run = wandb.init(
        project=str(kwargs.get("wandb_project") or os.environ["WANDB_PROJECT"]),
        name=str(kwargs.get("wandb_run_name", f"{config.experiment_name}-evaluate")),
        dir=str(wandb_dir),
        config={"splits": list(splits), "tasks": list(selected_tasks)},
        settings=wandb.Settings(init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "300"))),
    )
    if getattr(run, "url", None):
        Console().print(f"WandB run: {run.url}")
    return run


def evaluate_one_split(
    config: ExpConfig,
    selected_tasks: Sequence[str],
    split: str,
    output_dir: Path,
    console: Console,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    console.print(f"Loading {split} rows from {config.preprocessed_dataset_dir}")
    rows = load_converted_split(Path(config.preprocessed_dataset_dir), split, optional_int(kwargs.get("max_samples")))
    rows = filter_rows(rows, selected_tasks)
    if not rows:
        raise RuntimeError(f"No answerable rows found for split={split!r} and tasks={selected_tasks}")
    console.print(f"Evaluating {len(rows)} row(s) across {', '.join(selected_tasks)}")

    predictions_out = input_predictions_path_for_split(config, kwargs, split)
    if not predictions_out.exists():
        raise FileNotFoundError(
            f"Predictions for split={split!r} were not found at {predictions_out}. "
            "Run `mle infer ...` first or pass `predictions` in custom args."
        )
    prediction_records = normalize_prediction_records(load_records(predictions_out))
    console.print(f"Loaded predictions from {predictions_out}")

    results = evaluate_rows(
        rows,
        make_prediction_index(prediction_records),
        allow_missing_predictions=bool(kwargs.get("allow_missing_predictions", False)),
        iou_threshold=float(kwargs.get("iou_threshold", 0.5)),
        green_model_name=str(kwargs.get("green_model_name", "StanfordAIMI/GREEN-radllama2-7b")),
        green_output_dir=green_output_dir_for_split(kwargs.get("green_output_dir"), output_dir, split),
        green_batch_size=int(kwargs.get("green_batch_size", 8)),
        green_max_length=int(kwargs.get("green_max_length", 2048)),
        skip_green_score=parse_bool(kwargs.get("skip_green_score", False)),
        crimson_api=str(kwargs.get("crimson_api", "hf")),
        crimson_model_name=optional_str(kwargs.get("crimson_model_name")),
        crimson_batch_size=int(kwargs.get("crimson_batch_size", 1)),
        crimson_include_guidelines=not parse_bool(kwargs.get("crimson_no_guidelines", False)),
        skip_crimson_score=parse_bool(kwargs.get("skip_crimson_score", False)),
    )
    results["predictions"] = str(predictions_out)
    results["split"] = split
    results["model_variant"] = "predictions_file"
    results["num_predictions"] = len(prediction_records)
    results["num_rows"] = len(rows)
    return results


def resolve_eval_splits(kwargs: Mapping[str, Any]) -> list[str]:
    raw_splits = kwargs.get("splits")
    if raw_splits is None:
        raw_split = kwargs.get("split", "validation")
        if isinstance(raw_split, str) and raw_split.strip().lower() in {"all", "all_three", "all-three", "all_splits", "all-splits"}:
            return list(ALL_EVAL_SPLITS)
        raw_splits = [raw_split]
    if isinstance(raw_splits, str):
        if raw_splits.strip().lower() in {"all", "all_three", "all-three", "all_splits", "all-splits"}:
            return list(ALL_EVAL_SPLITS)
        raw_splits = re.split(r"[,; ]+", raw_splits.strip())
    splits = [normalize_split(split) for split in raw_splits if str(split).strip()]
    if not splits:
        raise ValueError("At least one evaluation split is required.")
    return sorted(set(splits), key=lambda split: list(ALL_EVAL_SPLITS).index(split) if split in ALL_EVAL_SPLITS else len(ALL_EVAL_SPLITS))


def prediction_path_for_split(predictions: Any, split: str) -> Path:
    if isinstance(predictions, Mapping):
        if split not in predictions:
            raise ValueError(f"Missing predictions path for split {split!r}")
        return Path(str(predictions[split]))
    text = str(predictions)
    if "{split}" in text:
        return Path(text.format(split=split))
    path = Path(text)
    if not path.suffix:
        return path / f"{split}_predictions.jsonl"
    return path


def input_predictions_path_for_split(config: ExpConfig, kwargs: Mapping[str, Any], split: str) -> Path:
    predictions = kwargs.get("predictions") or kwargs.get("predictions_in")
    if predictions:
        path = prediction_path_for_split(predictions, split)
        return first_existing_prediction_path(path, split)
    infer_output_dir = Path(
        kwargs.get("infer_output_dir")
        or kwargs.get("predictions_output_dir")
        or Path(config.output_dir) / f"{config.experiment_name}-infer"
    )
    return first_existing_prediction_path(infer_output_dir / f"{split}_predictions.jsonl", split)


def first_existing_prediction_path(path: Path, split: str) -> Path:
    if path.exists():
        return path
    legacy_candidates = [
        path.with_name(f"{{split_predictions_{split}.jsonl}}"),
        path.with_name(f"{{split}}_predictions_{split}.jsonl"),
        path.with_name(f"{{split_predictions.jsonl}}"),
    ]
    for candidate in legacy_candidates:
        if candidate.exists():
            return candidate
    return path


def predictions_out_path_for_split(predictions_out: Any, output_dir: Path, split: str) -> Path:
    if predictions_out is None:
        return output_dir / f"{split}_predictions.jsonl"
    if isinstance(predictions_out, Mapping):
        if split not in predictions_out:
            raise ValueError(f"Missing predictions_out path for split {split!r}")
        return Path(str(predictions_out[split]))
    text = str(predictions_out)
    if "{split}" in text:
        return Path(text.format(split=split))
    path = Path(text)
    if path.suffix:
        return path if split == "validation" else path.with_name(f"{path.stem}_{split}{path.suffix}")
    return path / f"{split}_predictions.jsonl"


def green_output_dir_for_split(green_output_dir: Any, output_dir: Path, split: str) -> Path:
    if green_output_dir is None:
        return output_dir / "green" / split
    text = str(green_output_dir)
    if "{split}" in text:
        return Path(text.format(split=split))
    return Path(text) / split


def normalize_split(split: str) -> str:
    return {
        "validation": "validation_public",
        "validation-public": "validation_public",
        "validation_public": "validation_public",
        "public": "validation_public",
        "val": "validation_public",
        "validation-hidden": "validation_hidden",
        "validation_hidden": "validation_hidden",
        "hidden": "validation_hidden",
        "test": "testing",
    }.get(split.strip().lower(), split.strip().lower())


def normalize_task_name(value: Any) -> str:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return TASK_ALIASES.get(text) or TASK_ALIASES.get(text.replace(" ", "_")) or text.replace(" ", "_")


def normalize_task_list(values: Sequence[Any]) -> list[str]:
    tasks = []
    for value in values:
        task = normalize_task_name(value)
        if task not in TASK_METRIC:
            raise ValueError(f"Unsupported task {value!r}. Supported tasks: {', '.join(TASK_METRIC)}")
        tasks.append(task)
    return sorted(set(tasks), key=list(TASK_METRIC).index)


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    out = int(value)
    return out if out > 0 else None


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_records(path: Path) -> list[Any]:
    if path.suffix == ".jsonl" or ".jsonl" in path.name:
        return read_jsonl_records(path)
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            if "Extra data" not in str(exc):
                raise
            return read_jsonl_records(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported prediction file top-level type: {type(data).__name__}")


def read_jsonl_records(path: Path) -> list[Any]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise json.JSONDecodeError(f"{exc.msg} at {path}:{line_number}", exc.doc, exc.pos) from exc
    return records


def normalize_prediction_records(records: list[Any]) -> list[dict[str, Any]]:
    if len(records) == 1 and isinstance(records[0], dict) and "uid" not in records[0]:
        return [{"uid": uid, "prediction": value} for uid, value in records[0].items()]
    out = []
    for record in records:
        if not isinstance(record, dict) or "uid" not in record:
            raise ValueError(f"Prediction rows must be JSON objects with uid: {record}")
        out.append(record)
    return out


def find_split_source(data_dir: Path, split: str) -> tuple[Path, str]:
    split = normalize_split(split)
    split_aliases = {
        "validation_public": ("validation_public", "validation"),
        "validation_hidden": ("validation_hidden",),
        "testing": ("testing", "test"),
        "train": ("train",),
    }
    candidate_splits = split_aliases.get(split, (split,))
    for base in (data_dir / "hf_dataset", data_dir):
        for candidate_split in candidate_splits:
            split_dir = base / candidate_split
            if split_dir.exists():
                return split_dir, "hf"
    for candidate_split in candidate_splits:
        split_jsonl = data_dir / f"{candidate_split}.jsonl"
        if split_jsonl.exists():
            return split_jsonl, "jsonl"
    raise FileNotFoundError(f"Could not find split {split!r} under {data_dir}")


def load_converted_split(data_dir: Path, split: str, max_samples: int | None) -> list[dict[str, Any]]:
    source, kind = find_split_source(data_dir, split)
    if kind == "hf":
        if load_from_disk is None:
            raise RuntimeError("Install datasets to load HF dataset splits.")
        dataset = load_from_disk(str(source))
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return [dict(row) for row in dataset]
    rows = read_jsonl(source)
    return rows[:max_samples] if max_samples else rows


def filter_rows(rows: Sequence[dict[str, Any]], tasks: Sequence[str]) -> list[dict[str, Any]]:
    selected = set(tasks)
    out = []
    for row in rows:
        task = normalize_task_name(row.get("task_type", ""))
        if task not in selected:
            continue
        answer = row.get("raw_answer", row.get("answer"))
        if answer is None or as_text(answer).strip() == "":
            continue
        row = dict(row)
        row["task_type"] = task
        out.append(row)
    return out


def maybe_json_load(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return collapse_ws(value).lower()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
    return collapse_ws(str(value)).lower()


def first_present(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def prediction_value(record: Mapping[str, Any] | None) -> Any:
    return "" if record is None else first_present(record, PREDICTION_KEYS)


def ground_truth_value(record: Mapping[str, Any]) -> Any:
    return first_present(record, GROUND_TRUTH_KEYS)


def parse_numeric(value: Any) -> float:
    value = maybe_json_load(value)
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not numeric: {value}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("value", "count", "number", "prediction"):
            if key in value:
                return parse_numeric(value[key])
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one numeric value, got {value}")
        return parse_numeric(value[0])
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not match:
        raise ValueError(f"Could not parse numeric value from {value!r}")
    return float(match.group(0))


def parse_multilabel(value: Any) -> set[str]:
    value = maybe_json_load(value)
    if value is None:
        items: list[Any] = []
    elif isinstance(value, dict):
        if "labels" in value:
            return parse_multilabel(value["labels"])
        items = [key for key, flag in value.items() if bool(flag)]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        text = str(value).strip()
        if normalize_text(text) in EMPTY_LABELS:
            return set()
        if ";" in text or "\n" in text or "|" in text:
            items = [part for part in re.split(r"[;\n|]+", text) if part.strip()]
        elif "," in text:
            items = [part for part in text.split(",") if part.strip()]
        else:
            items = [text]
    return {normalize_text(item) for item in items if normalize_text(item) not in EMPTY_LABELS}


def parse_bbox_from_numbers(numbers: Sequence[Any], label: str | None = None, *, xyxy: bool = True) -> Box:
    if len(numbers) != 4:
        raise ValueError(f"Expected four bbox numbers, got {numbers}")
    a, b, c, d = [float(value) for value in numbers]
    if xyxy:
        return Box(a, b, c, d, normalize_text(label) or None)
    return Box(a, b, a + c, b + d, normalize_text(label) or None)


def parse_boxes(value: Any) -> list[Box]:
    value = maybe_json_load(value)
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or normalize_text(text) in EMPTY_LABELS:
            return []
        parsed = maybe_json_load(text)
        if parsed is not value:
            return parse_boxes(parsed)
        groups = re.findall(r"\[\s*([-+0-9eE.,\s]+)\]", text)
        boxes = []
        for group in groups:
            numbers = [float(item.strip()) for item in group.split(",") if item.strip()]
            boxes.append(parse_bbox_from_numbers(numbers))
        if boxes:
            return boxes
        raise ValueError(f"Could not parse detection boxes from string: {value!r}")
    if isinstance(value, Mapping):
        label = first_present(value, ("label", "class", "category", "name"))
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            return [Box(float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"]), normalize_text(label) or None)]
        if all(key in value for key in ("x", "y", "width", "height")):
            return [Box(float(value["x"]), float(value["y"]), float(value["x"]) + float(value["width"]), float(value["y"]) + float(value["height"]), normalize_text(label) or None)]
        for key in BOX_FIELDS:
            if key in value:
                boxes = parse_boxes(value[key])
                if label is not None:
                    return [Box(box.x1, box.y1, box.x2, box.y2, box.label or normalize_text(label)) for box in boxes]
                return boxes
        boxes = []
        for item_label, item_value in value.items():
            boxes.extend(Box(box.x1, box.y1, box.x2, box.y2, box.label or normalize_text(item_label)) for box in parse_boxes(item_value))
        return boxes
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            return [parse_bbox_from_numbers(value)]
        boxes = []
        for item in value:
            boxes.extend(parse_boxes(item))
        return boxes
    raise ValueError(f"Unsupported detection value: {value!r}")


def iou(box_a: Box, box_b: Box) -> float:
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, box_a.x2 - box_a.x1) * max(0.0, box_a.y2 - box_a.y1)
    area_b = max(0.0, box_b.x2 - box_b.x1) * max(0.0, box_b.y2 - box_b.y1)
    denom = area_a + area_b - inter_area
    return 0.0 if denom <= 0 else inter_area / denom


def greedy_match_counts(gold_boxes: Sequence[Box], pred_boxes: Sequence[Box], threshold: float) -> tuple[int, int, int]:
    candidates = []
    for gold_index, gold_box in enumerate(gold_boxes):
        for pred_index, pred_box in enumerate(pred_boxes):
            if gold_box.label and pred_box.label and gold_box.label != pred_box.label:
                continue
            overlap = iou(gold_box, pred_box)
            if overlap >= threshold:
                candidates.append((overlap, gold_index, pred_index))
    candidates.sort(reverse=True)
    matched_gold = set()
    matched_pred = set()
    tp = 0
    for _overlap, gold_index, pred_index in candidates:
        if gold_index in matched_gold or pred_index in matched_pred:
            continue
        matched_gold.add(gold_index)
        matched_pred.add(pred_index)
        tp += 1
    return tp, len(pred_boxes) - tp, len(gold_boxes) - tp


def balanced_accuracy_score(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return float("nan")
    totals: dict[str, int] = {}
    hits: dict[str, int] = {}
    for gold, pred in zip(y_true, y_pred):
        totals[gold] = totals.get(gold, 0) + 1
        hits[gold] = hits.get(gold, 0) + int(gold == pred)
    return sum(hits[label] / totals[label] for label in sorted(totals)) / len(totals)


def accuracy_score(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return float("nan")
    return sum(int(gold == pred) for gold, pred in zip(y_true, y_pred)) / len(y_true)


def weighted_f1_score(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return float("nan")
    labels = sorted(set(y_true) | set(y_pred))
    total_support = len(y_true)
    weighted_sum = 0.0
    for label in labels:
        tp = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(y_true, y_pred) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred != label)
        support = sum(1 for gold in y_true if gold == label)
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        weighted_sum += support * f1
    return weighted_sum / total_support


def example_f1_score(y_true: Sequence[set[str]], y_pred: Sequence[set[str]]) -> float:
    return multilabel_metrics(y_true, y_pred)["f1_score"]


def multilabel_metrics(y_true: Sequence[set[str]], y_pred: Sequence[set[str]]) -> dict[str, float]:
    if not y_true:
        return {"f1_score": float("nan"), "precision": float("nan"), "recall": float("nan")}
    f1_scores = []
    precisions = []
    recalls = []
    for gold, pred in zip(y_true, y_pred):
        if not gold and not pred:
            f1_scores.append(1.0)
            precisions.append(1.0)
            recalls.append(1.0)
            continue
        tp = len(gold & pred)
        fp = len(pred - gold)
        fn = len(gold - pred)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    return {
        "f1_score": sum(f1_scores) / len(f1_scores),
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
    }


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not y_true:
        return float("nan")
    return sum(abs(gold - pred) for gold, pred in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not y_true:
        return float("nan")
    residual_norm = 0.0
    for gold, pred in zip(y_true, y_pred):
        residual_norm = math.hypot(residual_norm, gold - pred)
    return residual_norm / math.sqrt(len(y_true))


def prf_from_counts(total_tp: int, total_fp: int, total_fn: int) -> dict[str, float]:
    precision = 0.0 if total_tp + total_fp == 0 else total_tp / (total_tp + total_fp)
    recall = 0.0 if total_tp + total_fn == 0 else total_tp / (total_tp + total_fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"f1_score": f1, "precision": precision, "recall": recall}


def detection_counts(gold_values: Sequence[Any], pred_values: Sequence[Any], threshold: float) -> dict[str, int]:
    total_tp = total_fp = total_fn = 0
    for gold_raw, pred_raw in zip(gold_values, pred_values):
        gold_boxes = parse_boxes(gold_raw)
        try:
            pred_boxes = parse_boxes(pred_raw)
        except ValueError:
            pred_boxes = []
        tp, fp, fn = greedy_match_counts(gold_boxes, pred_boxes, threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return {"tp": total_tp, "fp": total_fp, "fn": total_fn}


def detection_metrics(gold_values: Sequence[Any], pred_values: Sequence[Any], threshold: float) -> dict[str, Any]:
    counts = detection_counts(gold_values, pred_values, threshold)
    return {**prf_from_counts(counts["tp"], counts["fp"], counts["fn"]), **counts}


def per_label_detection_metrics(gold_values: Sequence[Any], pred_values: Sequence[Any], threshold: float) -> dict[str, Any]:
    label_counts: dict[str, dict[str, int]] = {}
    for gold_raw, pred_raw in zip(gold_values, pred_values):
        gold_boxes = parse_boxes(gold_raw)
        try:
            pred_boxes = parse_boxes(pred_raw)
        except ValueError:
            pred_boxes = []
        labels = sorted({box.label or "__unlabeled__" for box in gold_boxes} | {box.label or "__unlabeled__" for box in pred_boxes})
        for label in labels:
            label_gold = [box for box in gold_boxes if (box.label or "__unlabeled__") == label]
            label_pred = [box for box in pred_boxes if (box.label or "__unlabeled__") == label]
            tp, fp, fn = greedy_match_counts(label_gold, label_pred, threshold)
            counts = label_counts.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
            counts["tp"] += tp
            counts["fp"] += fp
            counts["fn"] += fn
    return {
        label: {**prf_from_counts(counts["tp"], counts["fp"], counts["fn"]), **counts}
        for label, counts in sorted(label_counts.items())
    }


def detection_f1_score(gold_values: Sequence[Any], pred_values: Sequence[Any], threshold: float) -> tuple[float, dict[str, int]]:
    metrics = detection_metrics(gold_values, pred_values, threshold)
    return metrics["f1_score"], {"tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"]}


def detection_threshold_metrics(
    gold_values: Sequence[Any],
    pred_values: Sequence[Any],
    thresholds: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> dict[str, Any]:
    detailed_metrics = {}
    f1_values = []
    precision_values = []
    recall_values = []
    for threshold in thresholds:
        metrics = detection_metrics(gold_values, pred_values, threshold)
        detailed_metrics[f"IoU_{threshold:.1f}"] = metrics
        f1_values.append(metrics["f1_score"])
        precision_values.append(metrics["precision"])
        recall_values.append(metrics["recall"])

    metrics_at_03 = detailed_metrics["IoU_0.3"]
    metrics_at_05 = detailed_metrics["IoU_0.5"]
    return {
        "f1_score": metrics_at_05["f1_score"],
        "precision": metrics_at_05["precision"],
        "recall": metrics_at_05["recall"],
        "f1_score_at_03": metrics_at_03["f1_score"],
        "f1_score_at_05": metrics_at_05["f1_score"],
        "average_f1": sum(f1_values) / len(f1_values),
        "precision_at_03": metrics_at_03["precision"],
        "recall_at_03": metrics_at_03["recall"],
        "detailed_metrics": detailed_metrics,
        "coco_style_metrics": {
            "average_f1": sum(f1_values) / len(f1_values),
            "average_precision": sum(precision_values) / len(precision_values),
            "average_recall": sum(recall_values) / len(recall_values),
        },
        "per_chromosome_metrics": per_label_detection_metrics(gold_values, pred_values, 0.3),
    }


def green_score(refs: Sequence[str], hyps: Sequence[str], model_name: str, output_dir: Path, batch_size: int, max_length: int) -> float:
    try:
        from green_score import GREEN
    except ImportError as exc:
        raise ImportError("GREEN scoring requires the ATATC/GREEN package, imported as `green_score`.") from exc
    scorer = GREEN(model_name, output_dir=str(output_dir))
    if batch_size > 0 and hasattr(scorer, "batch_size"):
        scorer.batch_size = batch_size
    if max_length > 0 and hasattr(scorer, "max_length"):
        scorer.max_length = max_length
    tokenizer = getattr(scorer, "tokenizer", None)
    if tokenizer is not None and not hasattr(tokenizer, "batch_encode_plus") and callable(tokenizer):
        setattr(tokenizer, "batch_encode_plus", lambda texts, *args, **kw: tokenizer(texts, *args, **kw))
    mean, _std, _per_case, _summary, _result_df = scorer(list(refs), list(hyps))
    return float(mean)


def crimson_patient_context(row: Mapping[str, Any]) -> dict[str, str] | None:
    context_keys = {
        "Age": ("Age", "age", "patient_age"),
        "Indication": ("Indication", "indication", "clinical_indication", "reason_for_exam"),
    }
    context = {}
    for output_key, source_keys in context_keys.items():
        value = first_present(row, source_keys)
        if value is not None and as_text(value).strip():
            context[output_key] = as_text(value).strip()
    return context or None


def crimson_score(
    refs: Sequence[str],
    hyps: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    api: str,
    model_name: str | None,
    batch_size: int,
    include_guidelines: bool,
) -> dict[str, Any]:
    try:
        from CRIMSON import CRIMSONScore
    except ImportError as exc:
        raise ImportError("CRIMSON scoring requires the `crimson-score` package, imported as `CRIMSON`.") from exc

    scorer_kwargs = {"api": api}
    if model_name:
        scorer_kwargs["model_name"] = model_name
    scorer = CRIMSONScore(**scorer_kwargs)
    contexts = [crimson_patient_context(row) for row in rows]
    results = []

    if api in {"hf", "huggingface", "vllm"} and hasattr(scorer, "evaluate_batch"):
        results = scorer.evaluate_batch(
            list(refs),
            list(hyps),
            patient_contexts=contexts,
            include_guidelines=include_guidelines,
            batch_size=max(1, batch_size),
        )
    else:
        for ref, hyp, context in zip(refs, hyps, contexts):
            results.append(
                scorer.evaluate(
                    reference_findings=ref,
                    predicted_findings=hyp,
                    patient_context=context,
                    include_guidelines=include_guidelines,
                )
            )

    valid_results = [result for result in results if isinstance(result, Mapping) and result.get("crimson_score") is not None]
    if not valid_results:
        return {"crimson_score": float("nan"), "crimson_valid_samples": 0, "crimson_failed_samples": len(results)}

    scores = [float(result["crimson_score"]) for result in valid_results]
    error_count_keys = (
        "false_findings",
        "missing_findings",
        "attribute_errors",
        "location_errors",
        "severity_errors",
        "descriptor_errors",
        "measurement_errors",
        "certainty_errors",
        "unspecific_errors",
        "overinterpretation_errors",
        "temporal_errors",
    )
    aggregate: dict[str, Any] = {
        "crimson_score": sum(scores) / len(scores),
        "crimson_valid_samples": len(valid_results),
        "crimson_failed_samples": len(results) - len(valid_results),
    }
    for key in error_count_keys:
        values = [float(result.get("error_counts", {}).get(key, 0.0)) for result in valid_results]
        aggregate[f"crimson_{key}"] = sum(values) / len(values)
    for key in ("false_findings", "missing_findings", "attribute_errors"):
        values = [float(result.get("weighted_error_counts", {}).get(key, 0.0)) for result in valid_results]
        aggregate[f"crimson_weighted_{key}"] = sum(values) / len(values)
    for key in ("N_G", "E_penalty", "correct", "errors_more_than_correct", "S"):
        values = [float(result.get("metrics", {}).get(key, 0.0)) for result in valid_results]
        aggregate[f"crimson_{key}"] = sum(values) / len(values)
    return aggregate


def make_prediction_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index = {}
    for record in records:
        uid = str(record["uid"])
        if uid in index:
            continue
        index[uid] = record
    return index


def evaluate_rows(
    ground_truth_rows: Sequence[dict[str, Any]],
    prediction_index: Mapping[str, dict[str, Any]],
    *,
    allow_missing_predictions: bool,
    iou_threshold: float,
    green_model_name: str,
    green_output_dir: Path,
    green_batch_size: int,
    green_max_length: int,
    skip_green_score: bool = False,
    crimson_api: str = "hf",
    crimson_model_name: str | None = None,
    crimson_batch_size: int = 1,
    crimson_include_guidelines: bool = True,
    skip_crimson_score: bool = False,
) -> dict[str, Any]:
    task_rows: dict[str, list[tuple[dict[str, Any], dict[str, Any] | None]]] = {}
    missing = []
    for row in ground_truth_rows:
        uid = str(row.get("uid", ""))
        if not uid:
            raise ValueError(f"Ground-truth row is missing uid: {row}")
        task_type = normalize_task_name(row.get("task_type"))
        if task_type not in TASK_METRIC:
            raise ValueError(f"Unsupported task_type {task_type!r} for uid {uid}")
        pred_row = prediction_index.get(uid)
        if pred_row is None and not allow_missing_predictions:
            missing.append(uid)
        task_rows.setdefault(task_type, []).append((row, pred_row))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing predictions for {len(missing)} rows: {preview}")

    flat_metrics: dict[str, float] = {}
    by_task: dict[str, dict[str, Any]] = {}
    for task_type, pairs in sorted(task_rows.items()):
        gold_values = [ground_truth_value(row) for row, _ in pairs]
        pred_values = [prediction_value(pred) for _, pred in pairs]
        metric_name = TASK_METRIC[task_type]

        if task_type == "disease_diagnosis_classification":
            normalized_gold = [normalize_text(v) for v in gold_values]
            normalized_pred = [normalize_text(v) for v in pred_values]
            task_metrics = {
                "balanced_accuracy": balanced_accuracy_score(normalized_gold, normalized_pred),
                "accuracy": accuracy_score(normalized_gold, normalized_pred),
                "f1_score": weighted_f1_score(normalized_gold, normalized_pred),
                "valid_samples": len(pairs),
            }
            metric_value = task_metrics["balanced_accuracy"]
            flat_metrics["balanced_accuracy"] = metric_value
            flat_metrics["classification_accuracy"] = task_metrics["accuracy"]
            flat_metrics["classification_f1_score"] = task_metrics["f1_score"]
        elif task_type == "multi_label_classification":
            task_metrics = multilabel_metrics([parse_multilabel(v) for v in gold_values], [parse_multilabel(v) for v in pred_values])
            task_metrics["valid_samples"] = len(pairs)
            metric_value = task_metrics["f1_score"]
            flat_metrics["f1_score"] = metric_value
            flat_metrics["multi_label_f1_score"] = task_metrics["f1_score"]
            flat_metrics["multi_label_precision"] = task_metrics["precision"]
            flat_metrics["multi_label_recall"] = task_metrics["recall"]
        elif task_type == "detection":
            task_metrics = detection_threshold_metrics(gold_values, pred_values)
            task_metrics["valid_samples"] = len(pairs)
            metric_value = task_metrics["f1_score"]
            matching_metrics = task_metrics["detailed_metrics"].get(f"IoU_{iou_threshold:.1f}") or detection_metrics(gold_values, pred_values, iou_threshold)
            counts = {key: matching_metrics[key] for key in ("tp", "fp", "fn")}
            by_task[task_type] = {
                "metric": metric_name,
                "value": metric_value,
                "count": len(pairs),
                "metrics": task_metrics,
                "matching": counts,
                "iou_threshold": iou_threshold,
            }
            flat_metrics["f1_iou_0.5"] = metric_value
            flat_metrics["detection_f1_iou_0.5"] = metric_value
            flat_metrics["detection_precision_iou_0.5"] = task_metrics["precision"]
            flat_metrics["detection_recall_iou_0.5"] = task_metrics["recall"]
            flat_metrics["detection_f1_iou_0.3"] = task_metrics["f1_score_at_03"]
            flat_metrics["detection_precision_iou_0.3"] = task_metrics["precision_at_03"]
            flat_metrics["detection_recall_iou_0.3"] = task_metrics["recall_at_03"]
            flat_metrics["detection_average_f1"] = task_metrics["average_f1"]
            flat_metrics["detection_average_precision"] = task_metrics["coco_style_metrics"]["average_precision"]
            flat_metrics["detection_average_recall"] = task_metrics["coco_style_metrics"]["average_recall"]
            for threshold_key, threshold_metrics in task_metrics["detailed_metrics"].items():
                suffix = threshold_key.lower().replace("iou_", "iou_")
                flat_metrics[f"detection_{suffix}_f1_score"] = threshold_metrics["f1_score"]
                flat_metrics[f"detection_{suffix}_precision"] = threshold_metrics["precision"]
                flat_metrics[f"detection_{suffix}_recall"] = threshold_metrics["recall"]
                flat_metrics[f"detection_{suffix}_tp"] = threshold_metrics["tp"]
                flat_metrics[f"detection_{suffix}_fp"] = threshold_metrics["fp"]
                flat_metrics[f"detection_{suffix}_fn"] = threshold_metrics["fn"]
            continue
        elif task_type == "cell_counting":
            task_metrics = numeric_metrics_with_parse_fallback(gold_values, pred_values)
            task_metrics["valid_samples"] = len(pairs)
            metric_value = task_metrics["mean_absolute_error"]
            flat_metrics["cell_counting_mean_absolute_error"] = metric_value
            flat_metrics["cell_counting_root_mean_squared_error"] = task_metrics["root_mean_squared_error"]
        elif task_type == "regression":
            task_metrics = numeric_metrics_with_parse_fallback(gold_values, pred_values)
            task_metrics["valid_samples"] = len(pairs)
            metric_value = task_metrics["mean_absolute_error"]
            flat_metrics["regression_mean_absolute_error"] = metric_value
            flat_metrics["regression_root_mean_squared_error"] = task_metrics["root_mean_squared_error"]
        elif task_type == "report_generation":
            if skip_green_score:
                metric_value = float("nan")
            else:
                metric_value = green_score([str(v or "") for v in gold_values], [str(v or "") for v in pred_values], green_model_name, green_output_dir, green_batch_size, green_max_length)
            flat_metrics["green_score"] = metric_value
            task_metrics = {"green_score": metric_value, "valid_samples": len(pairs)}
            if skip_crimson_score:
                task_metrics["crimson_score"] = float("nan")
            else:
                crimson_metrics = crimson_score(
                    [str(v or "") for v in gold_values],
                    [str(v or "") for v in pred_values],
                    [row for row, _ in pairs],
                    api=crimson_api,
                    model_name=crimson_model_name,
                    batch_size=crimson_batch_size,
                    include_guidelines=crimson_include_guidelines,
                )
                task_metrics.update(crimson_metrics)
            flat_metrics["crimson_score"] = task_metrics["crimson_score"]
        else:
            raise AssertionError(f"Unhandled task type: {task_type}")
        by_task[task_type] = {"metric": metric_name, "value": metric_value, "count": len(pairs), "metrics": task_metrics}

    return {"metrics": flat_metrics, "by_task": by_task, "num_rows": len(ground_truth_rows), "num_tasks": len(by_task)}


def aggregate_split_results(split_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {}
    task_values: dict[str, dict[str, Any]] = {}
    total_rows = 0

    for split, results in split_results.items():
        total_rows += int(results.get("num_rows", 0))
        for metric_name, value in results.get("metrics", {}).items():
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                metric_values.setdefault(metric_name, []).append(float(value))
        for task_type, payload in results.get("by_task", {}).items():
            task_payload = task_values.setdefault(task_type, {
                "metric": payload.get("metric", TASK_METRIC.get(task_type, "")),
                "values": [],
                "count": 0,
                "per_split": {},
            })
            value = payload.get("value")
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                task_payload["values"].append(float(value))
                task_payload["per_split"][split] = float(value)
            task_payload["count"] += int(payload.get("count", 0))

    mean_metrics = {
        metric_name: sum(values) / len(values)
        for metric_name, values in sorted(metric_values.items())
        if values
    }
    mean_by_task = {}
    for task_type, payload in sorted(task_values.items()):
        values = payload.pop("values")
        mean_by_task[task_type] = {
            "metric": payload["metric"],
            "value": sum(values) / len(values) if values else float("nan"),
            "count": payload["count"],
            "per_split": payload["per_split"],
        }

    return {
        "metrics": mean_metrics,
        "mean_metrics": mean_metrics,
        "by_task": mean_by_task,
        "split_results": split_results,
        "num_rows": total_rows,
        "num_splits": len(split_results),
        "num_tasks": len(mean_by_task),
    }


def mae_with_parse_fallback(gold_values: Sequence[Any], pred_values: Sequence[Any]) -> float:
    return numeric_metrics_with_parse_fallback(gold_values, pred_values)["mean_absolute_error"]


def numeric_metrics_with_parse_fallback(gold_values: Sequence[Any], pred_values: Sequence[Any]) -> dict[str, float]:
    gold_numbers = [parse_numeric(value) for value in gold_values]
    pred_numbers = []
    for value in pred_values:
        try:
            pred_numbers.append(parse_numeric(value))
        except ValueError:
            pred_numbers.append(0.0)
    return {
        "mean_absolute_error": mean_absolute_error(gold_numbers, pred_numbers),
        "root_mean_squared_error": root_mean_squared_error(gold_numbers, pred_numbers),
    }


def print_summary(results: Mapping[str, Any], console: Console) -> None:
    if "split_results" in results:
        console.print("Evaluation Summary")
        console.print(f"Splits: {', '.join(results['split_results'])}")
        console.print(f"Rows: {results['num_rows']}")
        console.print("Mean metrics:")
        for metric_name, value in results.get("mean_metrics", {}).items():
            value_text = "nan" if isinstance(value, float) and math.isnan(value) else f"{value:.6f}"
            console.print(f"  {metric_name}: {value_text}")
        console.print("Per-split metrics:")
        for split, split_result in results["split_results"].items():
            metrics = ", ".join(
                f"{name}={value:.6f}" if isinstance(value, (int, float)) and not math.isnan(float(value)) else f"{name}=nan"
                for name, value in split_result.get("metrics", {}).items()
            )
            console.print(f"  {split}: {metrics}")
        return

    console.print("Evaluation Summary")
    console.print(f"Rows: {results['num_rows']}")
    for task_type, payload in results["by_task"].items():
        value = payload["value"]
        value_text = "nan" if isinstance(value, float) and math.isnan(value) else f"{value:.6f}"
        console.print(f"{task_type}: {payload['metric']} = {value_text} (n={payload['count']})")
        if "matching" in payload:
            match = payload["matching"]
            console.print(f"  tp={match['tp']} fp={match['fp']} fn={match['fn']}")
