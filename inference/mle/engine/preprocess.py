import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from mle.vars import ExpConfig
from rich.console import Console


TASK_METRIC = {
    "disease_diagnosis_classification": "balanced_accuracy",
    "multi_label_classification": "f1_score",
    "detection": "f1_iou_0.5",
    "cell_counting": "mean_absolute_error",
    "regression": "mean_absolute_error",
    "report_generation": "green_score",
}

ALL_TASKS = tuple(TASK_METRIC)

# Keep these names identical to the labels used by the released ME-VLIP router.
GLICLASS_LABELS = (
    "classification",
    "detection",
    "regression",
    "counting",
    "instance_detection",
    "multi-label classification",
    "report_generation",
)

TASK_ALIASES = {
    "classification": "disease_diagnosis_classification",
    "cls": "disease_diagnosis_classification",
    "disease diagnosis": "disease_diagnosis_classification",
    "disease diagnosis classification": "disease_diagnosis_classification",
    "disease_diagnosis_classification": "disease_diagnosis_classification",
    "multi-label classification": "multi_label_classification",
    "multi label classification": "multi_label_classification",
    "multi_label_classification": "multi_label_classification",
    "multilabel": "multi_label_classification",
    "instance detection": "detection",
    "instance_detection": "detection",
    "detection": "detection",
    "cell counting": "cell_counting",
    "counting": "cell_counting",
    "count": "cell_counting",
    "cell_counting": "cell_counting",
    "regression": "regression",
    "report": "report_generation",
    "report generation": "report_generation",
    "report_generation": "report_generation",
}

HF_DATASET_COLUMNS = [
    "split",
    "source",
    "case_id",
    "uid",
    "volume_path",
    "image_path",
    "image",
    "images",
    "modality",
    "task_type",
    "metric",
    "subtask",
    "qid",
    "follow_up",
    "question_type",
    "prompt",
    "question",
    "answer",
    "choices",
    "raw_answer",
    "messages",
]


def preprocess(config: ExpConfig, use_wandb: bool, smoke_test: bool, *, console: Console = Console(), **kwargs) -> None:
    """
    This is a template entrypoint for preprocessing. You MUST NOT change its signature, but you may add functions and
    classes to this file.

    All your logs MUST be sent to the provided console. Your implementation MUST support WandB logging and it MUST ONLY
    be enabled if :param use_wandb is `True`.

    :param config: experiment configuration
    :param use_wandb: whether to use wandb for logging
    :param smoke_test: whether to run in smoke test mode
    :param console: the console for logging
    :param kwargs: custom arguments
    """
    input_root = Path(config.dataset_dir)
    output_dir = Path(config.preprocessed_dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = normalize_task_list(kwargs.get("tasks") or ALL_TASKS)
    assistant_content_style = str(kwargs.get("assistant_content_style", "string"))
    allow_missing_images = bool(kwargs.get("allow_missing_images", False))
    include_unanswered = bool(kwargs.get("include_unanswered", False))
    max_rows_per_json = kwargs.get("max_rows_per_json")
    no_hf_dataset = bool(kwargs.get("no_hf_dataset", False))
    no_extract_archives = bool(kwargs.get("no_extract_archives", False))
    if smoke_test:
        console.print("Smoke test mode: limiting preprocessing rows per JSON.")
        if max_rows_per_json is None:
            max_rows_per_json = 32

    task_classifier = make_task_classifier(kwargs, console)

    console.print(f"Preprocessing FLARE-MLLM-2D from {input_root}")
    console.print(f"Writing converted MedGemma SFT data to {output_dir}")
    if not no_extract_archives:
        extract_image_archives(input_root, console)

    json_paths = kwargs.get("json_paths")
    if json_paths:
        question_jsons = [Path(path) for path in json_paths]
    else:
        question_jsons = find_question_jsons(input_root)
    if not question_jsons:
        raise RuntimeError(f"No question JSON files found under {input_root}")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "validation_hidden": [], "testing": []}
    for json_path in question_jsons:
        rows = rows_from_json(
            json_path=json_path,
            input_root=input_root,
            selected_tasks=tasks,
            assistant_content_style=assistant_content_style,
            allow_missing_images=allow_missing_images,
            include_unanswered=include_unanswered,
            max_rows_per_json=int(max_rows_per_json) if max_rows_per_json is not None else None,
            task_classifier=task_classifier,
        )
        counts = Counter(row["split"] for row in rows)
        console.print(f"Loaded {len(rows)} row(s) from {json_path}: {dict(counts)}")
        for row in rows:
            rows_by_split.setdefault(row["split"], []).append(row)

    train_rows = rows_by_split.get("train", [])
    validation_rows = rows_by_split.get("validation", [])
    hidden_rows = rows_by_split.get("validation_hidden", [])
    testing_rows = rows_by_split.get("testing", [])
    if not train_rows:
        raise RuntimeError("No training rows were produced. Check dataset layout, task filters, and answer availability.")

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "validation.jsonl", validation_rows)
    if hidden_rows:
        write_jsonl(output_dir / "validation_hidden.jsonl", hidden_rows)
    if testing_rows:
        write_jsonl(output_dir / "testing.jsonl", testing_rows)

    stats = {
        "dataset": config.dataset_name,
        "model": "google/medgemma-1.5-4b-it",
        "num_train_rows": len(train_rows),
        "num_validation_rows": len(validation_rows),
        "num_validation_hidden_rows": len(hidden_rows),
        "num_testing_rows": len(testing_rows),
        "tasks_requested": tasks,
        "smoke_test": smoke_test,
        "metrics": TASK_METRIC,
        "train_task_counts": dict(Counter(row["task_type"] for row in train_rows)),
        "validation_task_counts": dict(Counter(row["task_type"] for row in validation_rows)),
        "validation_hidden_task_counts": dict(Counter(row["task_type"] for row in hidden_rows)),
        "testing_task_counts": dict(Counter(row["task_type"] for row in testing_rows)),
        "json_files": [str(path) for path in question_jsons],
        "task_classifier": task_classifier.metadata if task_classifier else None,
    }
    write_json(output_dir / "dataset_info.json", stats)

    if not no_hf_dataset:
        save_hf_dataset(output_dir, rows_by_split)

    if use_wandb:
        import wandb

        wandb.init(
            project=kwargs.get("wandb_project", "medgemma15-flare-mllm-2d"),
            name=kwargs.get("wandb_run_name", f"{config.experiment_name}-preprocess"),
            dir=str(output_dir / "wandb"),
            config=stats,
            settings=wandb.Settings(init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "300"))),
        )
        wandb.log({
            "preprocess/train_rows": len(train_rows),
            "preprocess/validation_rows": len(validation_rows),
            "preprocess/validation_hidden_rows": len(hidden_rows),
            "preprocess/testing_rows": len(testing_rows),
        })
        wandb.finish()

    console.print(f"Done. Train={len(train_rows)} validation={len(validation_rows)} hidden={len(hidden_rows)} testing={len(testing_rows)}")


class TaskClassifier:
    """Thin local wrapper around the released ME-VLIP GLiClass task router."""

    def __init__(self, model_path: str, tokenizer_path: str, device: str, threshold: float, mode: str):
        try:
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Task routing requires gliclass==0.1.11. Install the project dependencies before using "
                "task_classifier_path."
            ) from exc
        model_dir = Path(model_path)
        tokenizer_dir = Path(tokenizer_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"GLiClass model directory not found: {model_dir}")
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(f"GLiClass tokenizer directory not found: {tokenizer_dir}")
        self.model = GLiClassModel.from_pretrained(str(model_dir), local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        self.pipeline = ZeroShotClassificationPipeline(
            self.model, self.tokenizer, classification_type="single-label", device=device
        )
        self.threshold = threshold
        self.mode = mode
        self.metadata = {
            "model_path": str(model_dir), "tokenizer_path": str(tokenizer_dir),
            "threshold": threshold, "mode": mode, "labels": list(GLICLASS_LABELS),
        }

    def predict(self, question: Any) -> tuple[str, float]:
        results = self.pipeline(as_text(question), list(GLICLASS_LABELS), threshold=0.1)
        ranked = results[0] if results and isinstance(results[0], list) else results
        if not ranked:
            return "", 0.0
        best = ranked[0]
        return normalize_task_name(best.get("label", "")), float(best.get("score", 0.0))


def default_task_classifier_tokenizer_path(model_path: str) -> Path:
    """Resolve the tokenizer supplied beside the user's ME-VLIP GLiClass checkpoint."""
    model_dir = Path(model_path)
    candidates = (model_dir, model_dir.parent / "deberta-v3-small")
    for candidate in candidates:
        if (candidate / "spm.model").exists():
            return candidate
    return model_dir


def make_task_classifier(kwargs: dict[str, Any], console: Console) -> TaskClassifier | None:
    model_path = kwargs.get("task_classifier_path")
    if not model_path:
        return None
    tokenizer_path = kwargs.get("task_classifier_tokenizer_path") or default_task_classifier_tokenizer_path(str(model_path))
    mode = str(kwargs.get("task_classifier_mode", "override")).strip().lower()
    if mode not in {"override", "fallback"}:
        raise ValueError("task_classifier_mode must be 'override' or 'fallback'")
    threshold = float(kwargs.get("task_classifier_threshold", 0.3))
    console.print(f"Loading ME-VLIP GLiClass task router from {model_path}")
    return TaskClassifier(str(model_path), str(tokenizer_path), str(kwargs.get("task_classifier_device", "cpu")), threshold, mode)


def normalize_task_list(values: Sequence[Any]) -> list[str]:
    tasks = []
    for value in values:
        task = normalize_task_name(value)
        if task not in TASK_METRIC:
            raise ValueError(f"Unsupported task {value!r}. Supported tasks: {', '.join(ALL_TASKS)}")
        tasks.append(task)
    return sorted(set(tasks), key=ALL_TASKS.index)


def normalize_task_name(value: Any) -> str:
    text = as_text(value).strip().lower().replace("_", " ")
    text = re.sub(r"[()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return TASK_ALIASES.get(text) or TASK_ALIASES.get(text.replace(" ", "_")) or text.replace(" ", "_")


def as_text(value: Any, *, empty: str = "") -> str:
    if value is None:
        return empty
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(as_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def first_present(record: dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    lower_to_key = {str(key).lower(): key for key in record}
    for name in names:
        if name in record:
            return record[name]
        key = lower_to_key.get(name.lower())
        if key is not None:
            return record[key]
    return default


def slug(value: Any) -> str:
    text = as_text(value, empty="unknown").strip() or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unknown"


def is_numeric_answer(answer: Any) -> bool:
    if isinstance(answer, (int, float)):
        return True
    if not isinstance(answer, str):
        return False
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer.strip().replace("%", "")))


def looks_like_detection(record: dict[str, Any]) -> bool:
    bbox_keys = {"bbox", "bboxes", "box", "boxes", "bounding_box", "bounding_boxes", "detections"}
    if any(key in record for key in bbox_keys):
        return True
    answer = first_present(record, ["Answer", "answer"])
    if isinstance(answer, dict) and any(key in answer for key in bbox_keys):
        return True
    if isinstance(answer, list) and answer and all(isinstance(item, (dict, list)) for item in answer):
        return True
    task_text = as_text(first_present(record, ["TaskType", "task_type", "task"])).lower()
    return "detection" in task_text


def infer_task_type(raw_task: Any, answer: Any, question: Any, record: dict[str, Any]) -> str:
    task = normalize_task_name(raw_task)
    if task in TASK_METRIC:
        return task
    question_text = as_text(question).lower()
    if looks_like_detection(record):
        return "detection"
    if ("count" in question_text or "how many" in question_text or "number of" in question_text) and is_numeric_answer(answer):
        return "cell_counting"
    if is_numeric_answer(answer):
        return "regression"
    return "disease_diagnosis_classification"


def answer_to_text(answer: Any, task_type: str) -> str:
    if answer is None:
        return ""
    if task_type == "multi_label_classification" and isinstance(answer, list):
        return "; ".join(str(item) for item in answer)
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False)
    return as_text(answer)


def format_choices(choices: Any) -> str:
    if not choices:
        return ""
    if isinstance(choices, str):
        return f"\nOptions: {choices}"
    if not isinstance(choices, list):
        return f"\nOptions: {as_text(choices)}"
    labels = []
    for index, choice in enumerate(choices):
        prefix = chr(ord("A") + index) if index < 26 else str(index + 1)
        labels.append(f"{prefix}) {choice}")
    return "\nOptions: " + " ".join(labels)


def task_instruction(task_type: str, modality: str) -> str:
    modality_text = modality or "medical image"
    if task_type == "multi_label_classification":
        return f"Identify all applicable findings or labels in this {modality_text} image. Select all that apply."
    if task_type == "detection":
        return "Identify and locate the requested findings. Return bounding boxes exactly in the coordinate format requested by the question."
    if task_type == "cell_counting":
        return "Return only the numeric count requested for this image."
    if task_type == "regression":
        return "Return only the requested numeric value, preserving units if present."
    if task_type == "report_generation":
        return f"Generate a concise diagnostic report for this {modality_text} image."
    return f"Answer the medical image classification question for this {modality_text} image. For multiple-choice questions, return the selected option letter or text exactly as annotated."


def make_prompt(question: Any, choices: Any, task_type: str, modality: str) -> str:
    question_text = as_text(question)
    if task_type == "report_generation" and not question_text:
        return task_instruction(task_type, modality)
    return f"{task_instruction(task_type, modality)}\nQuestion: {question_text}{format_choices(choices)}"


def make_messages(image_paths: Sequence[str], prompt: str, answer: str, assistant_content_style: str) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [{"type": "image", "image": path} for path in image_paths]
    user_content.append({"type": "text", "text": prompt})
    assistant_content: Any = answer if assistant_content_style == "string" else [{"type": "text", "text": answer}]
    return [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]


def load_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in ("questions", "data", "annotations", "items"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def find_question_jsons(input_root: Path) -> list[Path]:
    paths = []
    for pattern in ("*.json", "*.JSON"):
        paths.extend(input_root.rglob(pattern))
    split_roots = {"training", "train", "validation-public", "validation_public", "validation-hidden", "validation_hidden", "testing", "test"}
    return sorted(
        path for path in set(paths)
        if not path.name.startswith(".")
        and ".cache" not in path.parts
        and path.name.lower() != "predictions.json"
        and any(part in split_roots for part in path.relative_to(input_root).parts[:-1])
    )


def split_root(input_root: Path, split: str) -> Path | None:
    names_by_split = {
        "train": ("training", "train"),
        "validation": ("validation-public", "validation_public", "validation", "val"),
        "validation_hidden": ("validation-hidden", "validation_hidden", "hidden"),
        "testing": ("testing", "test"),
    }
    for name in names_by_split.get(split, ()):
        candidate = input_root / name
        if candidate.exists():
            return candidate
    return None


def extract_image_archives(input_root: Path, console: Console) -> None:
    archives = sorted(input_root.rglob("imagesTr.zip")) + sorted(input_root.rglob("imagesVal.zip"))
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            first_file = next((member for member in archive.infolist() if not member.is_dir()), None)
            if first_file is None:
                continue
            target_dir = archive_path.parent / Path(first_file.filename).parts[0]
            if target_dir.exists():
                continue
            console.print(f"Extracting {archive_path}")
            for member in archive.infolist():
                target_path = archive_path.parent / member.filename
                try:
                    target_path.resolve().relative_to(archive_path.parent.resolve())
                except ValueError as exc:
                    raise RuntimeError(f"Unsafe archive member {member.filename!r} in {archive_path}") from exc
            archive.extractall(archive_path.parent)


def infer_source(json_path: Path, input_root: Path) -> tuple[str, str]:
    try:
        parts = json_path.relative_to(input_root).parts
    except ValueError:
        parts = json_path.parts
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    if len(parts) >= 2:
        return "unknown", parts[-2]
    return "unknown", json_path.stem


def infer_split_from_path(json_path: Path, input_root: Path) -> str:
    try:
        parts = {part.lower() for part in json_path.relative_to(input_root).parts}
    except ValueError:
        parts = {part.lower() for part in json_path.parts}
    name = json_path.name.lower()
    if "validation-hidden" in parts or "validation_hidden" in parts or "hidden" in name:
        return "validation_hidden"
    if "testing" in parts or "test" in name:
        return "testing"
    if "validation-public" in parts or "validation_public" in parts or "validation" in parts or "val" in name:
        return "validation"
    return "train"


def normalize_split(raw_split: Any) -> str:
    text = as_text(raw_split).strip().lower().replace("-", "_")
    if text in {"validation_hidden", "hidden", "val_hidden"}:
        return "validation_hidden"
    if text in {"testing", "test", "ts"}:
        return "testing"
    if text in {"validation", "validation_public", "val", "valid", "public"}:
        return "validation"
    if text in {"training", "train", "tr"}:
        return "train"
    return text or "train"


def image_names_from_record(record: dict[str, Any]) -> list[str]:
    image_name = first_present(record, ["ImageName", "image_name", "image", "image_path", "ImagePath", "file_name", "filename"])
    if not image_name:
        return []
    if isinstance(image_name, (list, tuple)):
        return [as_text(item).strip() for item in image_name if as_text(item).strip()]
    return [as_text(image_name).strip()]


def resolve_one_image_path(image_text: str, record: dict[str, Any], json_path: Path, input_root: Path, allow_missing: bool) -> str:
    if not image_text:
        if allow_missing:
            return ""
        raise ValueError(f"Missing image field in {json_path}: {record}")
    image_path = Path(image_text)
    if image_path.is_absolute() and image_path.exists():
        return str(image_path.resolve())

    split = normalize_split(first_present(record, ["Split", "split"], infer_split_from_path(json_path, input_root)))
    candidates = [json_path.parent / image_path, input_root / image_path]
    base = split_root(input_root, split)
    if base is not None:
        candidates.append(base / image_path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    basename = image_path.name
    if basename:
        for candidate in json_path.parent.rglob(basename):
            if candidate.exists():
                return str(candidate.resolve())
    if allow_missing:
        return str(candidates[0].resolve())
    raise FileNotFoundError(f"Could not find image {image_text!r} for row in {json_path}")


def resolve_image_paths(record: dict[str, Any], json_path: Path, input_root: Path, allow_missing: bool) -> list[str]:
    image_names = image_names_from_record(record)
    if not image_names:
        if allow_missing:
            return []
        raise ValueError(f"Missing image field in {json_path}: {record}")
    return [path for path in (resolve_one_image_path(name, record, json_path, input_root, allow_missing) for name in image_names) if path]


def row_from_record(
    record: dict[str, Any],
    json_path: Path,
    input_root: Path,
    fallback_split: str,
    selected_tasks: Sequence[str],
    assistant_content_style: str,
    allow_missing_images: bool,
    include_unanswered: bool,
    index: int,
    task_classifier: TaskClassifier | None = None,
) -> dict[str, Any] | None:
    split = normalize_split(first_present(record, ["Split", "split"], fallback_split))
    if fallback_split == "validation_hidden" and split == "validation":
        split = "validation_hidden"
    elif fallback_split == "testing" and split in {"validation", "train"}:
        split = "testing"
    modality, dataset_name = infer_source(json_path, input_root)
    modality = as_text(first_present(record, ["Modality", "modality"], modality))
    question = first_present(record, ["Question", "question", "prompt"], "")
    answer = first_present(record, ["Answer", "answer", "label", "target", "Report", "report"], None)
    if answer is None and not include_unanswered:
        return None

    raw_task = first_present(record, ["TaskType", "task_type", "task"], "")
    annotated_task_type = infer_task_type(raw_task, answer, question, record)
    classifier_task_type, classifier_score = ("", 0.0)
    if task_classifier is not None:
        classifier_task_type, classifier_score = task_classifier.predict(question)
    task_type = annotated_task_type
    if classifier_task_type in TASK_METRIC and classifier_score >= task_classifier.threshold:
        if task_classifier.mode == "override" or not raw_task:
            task_type = classifier_task_type
    if task_type not in selected_tasks:
        return None

    choices = first_present(record, ["Choices", "choices", "Options", "options"], [])
    image_paths = resolve_image_paths(record, json_path, input_root, allow_missing_images)
    image_path = image_paths[0] if image_paths else ""
    image_names = image_names_from_record(record)
    case_stem = Path(image_names[0]).stem if image_names else str(index)
    case_id = slug(first_present(record, ["ID", "Id", "id", "case_id", "CaseID", "qid", "QuestionID"], case_stem or index))
    qid = first_present(record, ["qid", "QuestionID", "question_id", "ID", "Id", "id"], index)
    prompt = make_prompt(question, choices, task_type, modality)
    answer_text = answer_to_text(answer, task_type)
    uid = f"{split}:{dataset_name}:{json_path.stem}:{case_id}:{task_type}:{qid}"

    return {
        "split": split,
        "source": dataset_name,
        "case_id": case_id,
        "uid": uid,
        "volume_path": image_path,
        "image_path": image_path,
        "image": image_path,
        "images": image_paths,
        "modality": modality,
        "task_type": task_type,
        "annotated_task_type": annotated_task_type,
        "classifier_task_type": classifier_task_type,
        "classifier_score": classifier_score,
        "metric": TASK_METRIC[task_type],
        "subtask": as_text(raw_task or task_type),
        "qid": qid,
        "follow_up": "",
        "question_type": as_text(first_present(record, ["QuestionType", "question_type"], "")),
        "prompt": prompt,
        "question": prompt,
        "answer": answer_text,
        "choices": choices or [],
        "raw_answer": answer,
        "messages": make_messages(image_paths, prompt, answer_text, assistant_content_style),
    }


def rows_from_json(
    json_path: Path,
    input_root: Path,
    selected_tasks: Sequence[str],
    assistant_content_style: str,
    allow_missing_images: bool,
    include_unanswered: bool,
    max_rows_per_json: int | None,
    task_classifier: TaskClassifier | None = None,
) -> list[dict[str, Any]]:
    records = load_json(json_path)
    if max_rows_per_json is not None:
        records = records[:max_rows_per_json]
    fallback_split = infer_split_from_path(json_path, input_root)
    rows = []
    for index, record in enumerate(records):
        row = row_from_record(
            record,
            json_path,
            input_root,
            fallback_split,
            selected_tasks,
            assistant_content_style,
            allow_missing_images,
            include_unanswered,
            index,
            task_classifier,
        )
        if row is not None:
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def to_arrow_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def arrow_safe_row(row: dict[str, Any]) -> dict[str, str]:
    return {key: to_arrow_string(row.get(key, "")) for key in HF_DATASET_COLUMNS}


def save_hf_dataset(output_dir: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    try:
        from datasets import Dataset, DatasetDict, Features, Value
    except ImportError as exc:
        raise RuntimeError("Install `datasets` or pass no_hf_dataset=true in custom args.") from exc

    features = Features({key: Value("string") for key in HF_DATASET_COLUMNS})
    dataset = DatasetDict()
    for split, rows in rows_by_split.items():
        if rows:
            dataset[split] = Dataset.from_list([arrow_safe_row(row) for row in rows], features=features)
    dataset.save_to_disk(str(output_dir / "hf_dataset"))
