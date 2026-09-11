import gc
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from rich.console import Console

from mle.engine.evaluate import (
    MODEL_ID,
    TASK_INSTRUCTIONS,
    as_text,
    load_converted_split,
    maybe_json_load,
    normalize_task_list,
    normalize_task_name,
    predictions_out_path_for_split,
    resolve_eval_splits,
    write_json,
    write_jsonl,
)
from mle.vars import ExpConfig
from mle.engine.preprocess import TASK_METRIC, make_task_classifier

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover
    Image = None
    ImageOps = None

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
except Exception:  # pragma: no cover
    AutoModelForImageTextToText = None
    AutoProcessor = None
    BitsAndBytesConfig = None


MOELORA_TASK_ADAPTERS = {
    "disease_diagnosis_classification": "expert_disease_diagnosis_classification",
    "multi_label_classification": "expert_multi_label_classification",
    "report_generation": "expert_report_generation",
    "detection": "expert_detection",
    "cell_counting": "expert_cell_counting",
    "regression": "expert_regression",
}


def infer(config: ExpConfig, tasks: Sequence[str], use_wandb: bool, smoke_test: bool, *, console: Console = Console(),
          **kwargs) -> None:
    """
    This is a template entrypoint for inference. You MUST NOT change its signature, but you may add functions and
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
        console.print("Smoke test mode: limiting inference samples and generation length.")
        kwargs.setdefault("max_samples", 4)
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("image_size", 512)
        kwargs.setdefault("max_new_tokens", 32)

    selected_tasks = normalize_task_list(tasks or kwargs.get("tasks") or TASK_INSTRUCTIONS)
    splits = resolve_eval_splits(kwargs)
    output_dir = Path(
        kwargs.get("infer_output_dir")
        or kwargs.get("predictions_output_dir")
        or kwargs.get("eval_output_dir")
        or Path(config.output_dir) / f"{config.experiment_name}-infer"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_results = {}
    model_bundle = None
    task_classifier = make_task_classifier(kwargs, console)
    wandb_run = start_wandb_run(config, output_dir, selected_tasks, splits, kwargs) if use_wandb else None
    completed = False
    try:
        if wandb_run is not None:
            wandb_run.log({"infer/status": "started", "infer/num_splits": len(splits)})
        if len(splits) > 1:
            model_bundle = load_model_and_processor(config, kwargs)
        for split in splits:
            split_results[split] = infer_one_split(
                config,
                selected_tasks,
                split,
                output_dir,
                console,
                kwargs,
                model_bundle,
                task_classifier=task_classifier,
                wandb_run=wandb_run,
            )

        details = {
            "splits": splits,
            "tasks": selected_tasks,
            "model_variant": selected_model_variant(config, kwargs),
            "split_results": split_results,
            "num_predictions": sum(int(result["num_predictions"]) for result in split_results.values()),
        }
        details_path = Path(kwargs.get("infer_details_json") or output_dir / "inference_details.json")
        write_json(details_path, details)
        console.print(f"Saved inference details to {details_path}")

        if wandb_run is not None:
            wandb_run.log({"infer/status": "completed", "infer/num_predictions": details["num_predictions"]})
            for split, result in split_results.items():
                wandb_run.log({f"infer/{split}/num_predictions": result["num_predictions"]})
            log_wandb_file(wandb_run, details_path, "inference-details")
        completed = True
    finally:
        if model_bundle is not None:
            del model_bundle
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        if wandb_run is not None:
            if not completed:
                wandb_run.log({"infer/status": "failed"})
            wandb_run.finish(exit_code=0 if completed else 1)


def infer_one_split(
    config: ExpConfig,
    selected_tasks: Sequence[str],
    split: str,
    output_dir: Path,
    console: Console,
    kwargs: Mapping[str, Any],
    model_bundle: tuple[Any, Any, str] | None = None,
    task_classifier: Any | None = None,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    console.print(f"Loading {split} rows from {config.preprocessed_dataset_dir}")
    rows = load_converted_split(Path(config.preprocessed_dataset_dir), split, optional_int(kwargs.get("max_samples")))
    rows = route_rows_with_classifier(rows, task_classifier)
    rows = filter_inference_rows(rows, selected_tasks)
    if not rows:
        raise RuntimeError(f"No rows found for split={split!r} and tasks={selected_tasks}")
    console.print(f"Generating predictions for {len(rows)} row(s) across {', '.join(selected_tasks)}")
    if wandb_run is not None:
        wandb_run.log({f"infer/{split}/status": "started", f"infer/{split}/num_rows": len(rows)})

    prediction_records = generate_predictions(config, rows, console, dict(kwargs), model_bundle=model_bundle, split=split, wandb_run=wandb_run)
    predictions_out = predictions_out_path_for_split(kwargs.get("predictions_out"), output_dir, split)
    write_jsonl(predictions_out, prediction_records)
    console.print(f"Saved generated predictions to {predictions_out}")
    submission_path = None
    if kwargs.get("submission_data_root"):
        predictions = {str(record["uid"]): str(record.get("prediction", "")) for record in prediction_records}
        submission_path = output_dir / str(kwargs.get("submission_filename", "predictions.json"))
        write_json(submission_path, build_flare_submission(Path(kwargs["submission_data_root"]), split, predictions))
        console.print(f"Saved FLARE submission to {submission_path}")
    if wandb_run is not None:
        wandb_run.log({f"infer/{split}/status": "completed", f"infer/{split}/num_predictions": len(prediction_records)})
        log_wandb_file(wandb_run, predictions_out, f"predictions-{split}")
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "split": split,
        "predictions": str(predictions_out),
        "num_predictions": len(prediction_records),
        "num_rows": len(rows),
        "model_variant": selected_model_variant(config, kwargs),
        "submission": str(submission_path) if submission_path else None,
    }


def iter_flare_records(value: Any) -> list[dict[str, Any]]:
    """Accept the list or mapping layouts used by the original FLARE JSON files."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        if isinstance(value.get("data"), list):
            return [item for item in value["data"] if isinstance(item, dict)]
        return [item for item in value.values() if isinstance(item, dict)]
    return []


def as_image_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def build_flare_submission(data_root: Path, split: str, predictions: Mapping[str, str]) -> list[dict[str, Any]]:
    """Create Qwen-compatible flat FLARE records with prediction in ``Answer``."""
    output: list[dict[str, Any]] = []
    per_json_qid: dict[Path, int] = {}
    for json_path in sorted(data_root.rglob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            records = iter_flare_records(json.load(handle))
        dataset_name = json_path.parent.name
        for record in records:
            question = str(record.get("Question") or record.get("question") or "").strip()
            if not question:
                continue
            qid = per_json_qid.get(json_path, 0)
            per_json_qid[json_path] = qid + 1
            image_names = [str(item) for item in as_image_list(record.get("ImageName") or record.get("image") or record.get("images"))]
            case_id = str(record.get("CaseID") or record.get("case_id") or (Path(image_names[0]).stem if image_names else qid))
            task_type = normalize_task_name(record.get("TaskType") or record.get("task_type") or "")
            uid = f"{split}:{dataset_name}:{json_path.stem}:{case_id}:{task_type}:{qid}"
            sample = deepcopy(record)
            sample["Answer"] = predictions.get(uid, "")
            output.append(sample)
    return output


def route_rows_with_classifier(rows: Sequence[dict[str, Any]], task_classifier: Any | None) -> list[dict[str, Any]]:
    """Apply the same ME-VLIP router at inference, including unlabeled hidden/test rows."""
    if task_classifier is None:
        return list(rows)
    routed = []
    for source_row in rows:
        row = dict(source_row)
        task_type, score = task_classifier.predict(row.get("question") or row.get("prompt") or "")
        row["classifier_task_type"] = task_type
        row["classifier_score"] = score
        if task_type in TASK_METRIC and score >= task_classifier.threshold:
            if task_classifier.mode == "override" or not row.get("task_type"):
                row["task_type"] = task_type
        routed.append(row)
    return routed


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    out = int(value)
    return out if out > 0 else None


def start_wandb_run(config: ExpConfig, output_dir: Path, selected_tasks: Sequence[str], splits: Sequence[str], kwargs: Mapping[str, Any]):
    import wandb

    wandb_dir = Path(kwargs.get("wandb_dir") or output_dir / "wandb")
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_MODE", str(kwargs.get("wandb_mode", "online")))
    os.environ.setdefault("WANDB_PROJECT", str(kwargs.get("wandb_project", "medgemma15-flare-mllm-2d")))
    if kwargs.get("wandb_entity"):
        os.environ.setdefault("WANDB_ENTITY", str(kwargs["wandb_entity"]))
    if kwargs.get("wandb_tags"):
        os.environ.setdefault("WANDB_TAGS", str(kwargs["wandb_tags"]))
    os.environ.pop("WANDB_DISABLED", None)

    return wandb.init(
        project=str(kwargs.get("wandb_project", os.environ["WANDB_PROJECT"])),
        name=str(kwargs.get("wandb_run_name", f"{config.experiment_name}-infer")),
        dir=str(wandb_dir),
        config={
            "splits": list(splits),
            "tasks": list(selected_tasks),
            "model_variant": selected_model_variant(config, kwargs),
            "model_name_or_path": str(kwargs.get("model_name_or_path", MODEL_ID)),
            "batch_size": int(kwargs.get("batch_size", 1)),
            "max_new_tokens": int(kwargs.get("max_new_tokens", 256)),
        },
        settings=wandb.Settings(init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "300"))),
    )


def log_wandb_file(wandb_run: Any, path: Path, artifact_name: str) -> None:
    try:
        import wandb

        artifact = wandb.Artifact(artifact_name, type="inference-output")
        artifact.add_file(str(path))
        wandb_run.log_artifact(artifact)
    except Exception:
        try:
            wandb_run.save(str(path))
        except Exception:
            pass


def filter_inference_rows(rows: Sequence[dict[str, Any]], tasks: Sequence[str]) -> list[dict[str, Any]]:
    selected = set(tasks)
    out = []
    for row in rows:
        task = normalize_task_name(row.get("task_type", ""))
        if task not in selected:
            continue
        row = dict(row)
        row["task_type"] = task
        out.append(row)
    return out


def get_image_paths(row: Mapping[str, Any]) -> list[str]:
    images = maybe_json_load(row.get("images"))
    if isinstance(images, list):
        paths = [str(path).strip() for path in images if str(path).strip()]
        if paths:
            return paths
    if isinstance(images, str) and images.strip():
        return [images.strip()]
    for key in ("image_path", "image", "volume_path"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    raise KeyError(f"No image path found for uid={row.get('uid')}")


def load_image(path: str, image_size: int, resize_mode: str):
    if Image is None or ImageOps is None:
        raise RuntimeError("Pillow is required for inference.")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    if image_size and image_size > 0:
        resample = Image.Resampling.BICUBIC
        if resize_mode == "square":
            image = image.resize((image_size, image_size), resample)
        elif resize_mode == "longest":
            image.thumbnail((image_size, image_size), resample)
        elif resize_mode != "none":
            raise ValueError(f"Unknown resize_mode: {resize_mode}")
    return image


def row_images(row: Mapping[str, Any], image_size: int, resize_mode: str, max_images_per_sample: int):
    paths = get_image_paths(row)
    if max_images_per_sample > 0:
        paths = paths[:max_images_per_sample]
    return [load_image(path, image_size, resize_mode) for path in paths]


def build_prompt(row: Mapping[str, Any]) -> str:
    task = normalize_task_name(row.get("task_type", ""))
    prompt = as_text(row.get("prompt") or row.get("question") or "")
    prompt_style = str(row.get("_prompt_style", "default")).strip().lower()
    original_question = extract_original_question(prompt)
    if prompt_style in {"question_only", "question-only", "raw_question", "raw-question"}:
        return original_question
    if prompt_style in {"task_type_question", "task-type-question", "task_tag_question", "task-tag-question"}:
        return f"[{router_task_label(task)}]\n{original_question}"
    if prompt_style not in {"", "default", "instruction"}:
        raise ValueError(
            "prompt_style must be one of: default, question_only, task_type_question"
        )
    choices = maybe_json_load(row.get("choices"))
    if not isinstance(choices, list):
        choices = []
    parts = [TASK_INSTRUCTIONS.get(task, "Answer the medical imaging question using the provided image.")]
    if prompt:
        parts.append(prompt)
    if choices and "options:" not in prompt.lower():
        parts.append("Options: " + "; ".join(str(choice) for choice in choices))
    return "\n\n".join(parts)


def extract_original_question(prompt: str) -> str:
    """Remove the preprocessing instruction while preserving the source question verbatim."""
    marker = "\nQuestion: "
    if marker in prompt:
        return prompt.split(marker, 1)[1]
    return prompt


def router_task_label(task: str) -> str:
    """Use the task tags expected by the released ME-VLIP question router."""
    return {
        "disease_diagnosis_classification": "classification",
        "multi_label_classification": "multi-label classification",
        "detection": "detection",
        "cell_counting": "counting",
        "regression": "regression",
        "report_generation": "report_generation",
    }.get(task, task)


def make_generation_messages(num_images: int, prompt: str, system_prompt: str) -> list[dict[str, Any]]:
    content = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": prompt})
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}, {"role": "user", "content": content}]


def load_model_and_processor(config: ExpConfig, kwargs: Mapping[str, Any]):
    missing = []
    base_model = should_infer_base_model(kwargs)
    adapter_path = resolve_adapter_path(config, kwargs)
    if torch is None:
        missing.append("torch")
    if AutoModelForImageTextToText is None or AutoProcessor is None or BitsAndBytesConfig is None:
        missing.append("transformers")
    if not base_model and adapter_path and PeftModel is None:
        missing.append("peft")
    if missing:
        raise RuntimeError("Missing inference dependencies: " + ", ".join(sorted(set(missing))))

    device = str(kwargs.get("device", "auto"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Pass device=cpu for a slow CPU-only dry run.")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else (torch.float16 if device == "cuda" else torch.float32)
    load_in_4bit = bool(kwargs.get("load_in_4bit", device == "cuda"))
    quant_config = None
    if load_in_4bit and device == "cuda":
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)

    model_name = str(kwargs.get("model_name_or_path", MODEL_ID))
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        quantization_config=quant_config,
        trust_remote_code=True,
        local_files_only=bool(kwargs.get("local_files_only", True)),
    )
    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is required to infer with a fine-tuned adapter. Pass --base_model to infer without it.")
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=bool(kwargs.get("local_files_only", True)))
        checkpoint = Path(adapter_path)
        expert_dirs = {
            task: checkpoint / adapter_name
            for task, adapter_name in MOELORA_TASK_ADAPTERS.items()
        }
        available_experts = {task: path for task, path in expert_dirs.items() if (path / "adapter_config.json").is_file()}
        if available_experts:
            if set(available_experts) != set(MOELORA_TASK_ADAPTERS):
                raise RuntimeError(
                    "Incomplete MedGemma MoE-LoRA checkpoint: "
                    f"found experts for {sorted(available_experts)}, expected {sorted(MOELORA_TASK_ADAPTERS)}."
                )
            for task, expert_dir in expert_dirs.items():
                model.load_adapter(str(expert_dir), adapter_name=MOELORA_TASK_ADAPTERS[task], is_trainable=False)
            # Some curated bundles preserve the exact shared/default LoRA that
            # accompanied a task checkpoint.  Load those optional task-specific
            # shared adapters so inference can reproduce the original
            # ``checkpoint root + target expert`` pairing instead of forcing all
            # experts to use the root/default adapter.
            task_shared_adapters = {}
            for task in MOELORA_TASK_ADAPTERS:
                shared_name = f"shared_{task}"
                shared_dir = checkpoint / shared_name
                if (shared_dir / "adapter_config.json").is_file():
                    model.load_adapter(str(shared_dir), adapter_name=shared_name, is_trainable=False)
                    task_shared_adapters[task] = shared_name
            model.base_model.set_adapter(["default", MOELORA_TASK_ADAPTERS["disease_diagnosis_classification"]])
            model._flare_moelora_task_adapters = dict(MOELORA_TASK_ADAPTERS)
            model._flare_moelora_shared_adapters = task_shared_adapters
    if device == "cpu":
        model.to(device)
    model.eval()

    # A LoRA adapter may save tokenizer files but not the base image processor
    # assets (e.g. preprocessor_config.json). Always load the processor from the
    # local MedGemma base directory to avoid an unintended Hub fallback.
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=bool(kwargs.get("local_files_only", True)),
    )
    processor.tokenizer.padding_side = "left" if int(kwargs.get("batch_size", 1)) > 1 else "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    return model, processor, device


def activate_moelora_task(model: Any, task_name: str) -> None:
    """Activate the shared/default LoRA plus the expert for one FLARE task."""
    task_adapters = getattr(model, "_flare_moelora_task_adapters", None)
    if task_adapters is None:
        return
    task_name = normalize_task_name(task_name)
    adapter_name = task_adapters.get(task_name)
    if adapter_name is None:
        raise ValueError(
            f"No MedGemma MoE-LoRA expert for task {task_name!r}; "
            f"available={sorted(task_adapters)}"
        )
    shared_adapters = getattr(model, "_flare_moelora_shared_adapters", {})
    shared_adapter = shared_adapters.get(task_name, "default")
    model.base_model.set_adapter([shared_adapter, adapter_name])


def should_infer_base_model(kwargs: Mapping[str, Any]) -> bool:
    for key in ("base_model", "base_model_only", "infer_base_model", "evaluate_base_model", "no_adapter"):
        if parse_bool(kwargs.get(key, False)):
            return True
    return False


def resolve_adapter_path(config: ExpConfig, kwargs: Mapping[str, Any]) -> str | None:
    if should_infer_base_model(kwargs):
        return None
    adapter_path = kwargs.get("adapter_path")
    if adapter_path:
        return str(adapter_path)
    candidate = Path(kwargs.get("model_output_dir") or Path(config.output_dir) / f"{config.experiment_name}-medgemma15-lora") / "final"
    return str(candidate) if candidate.exists() else None


def selected_model_variant(config: ExpConfig, kwargs: Mapping[str, Any]) -> str:
    if should_infer_base_model(kwargs):
        return "base"
    return "adapter" if resolve_adapter_path(config, kwargs) else "base"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def generate_predictions(
    config: ExpConfig,
    rows: Sequence[dict[str, Any]],
    console: Console,
    kwargs: dict[str, Any],
    model_bundle: tuple[Any, Any, str] | None = None,
    split: str = "inference",
    wandb_run: Any | None = None,
) -> list[dict[str, Any]]:
    model, processor, device = model_bundle or load_model_and_processor(config, kwargs)
    image_size = int(kwargs.get("image_size", 896))
    resize_mode = str(kwargs.get("resize_mode", "square"))
    max_images_per_sample = int(kwargs.get("max_images_per_sample", 1))
    batch_size = max(1, int(kwargs.get("batch_size", 1)))
    prompt_style = str(kwargs.get("prompt_style", "default"))
    system_prompt = str(kwargs.get("system_prompt", "You are an expert medical imaging assistant."))
    predictions = []
    log_every = max(1, int(kwargs.get("log_every", 25)))
    next_log_at = log_every

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch_tasks = {normalize_task_name(row.get("task_type", "")) for row in batch_rows}
        if len(batch_tasks) != 1 and getattr(model, "_flare_moelora_task_adapters", None) is not None:
            raise ValueError(
                "MedGemma MoE-LoRA inference batches must contain one task. "
                "Use batch_size=1 or group rows by task."
            )
        activate_moelora_task(model, next(iter(batch_tasks)))
        texts = []
        batch_images = []
        for row in batch_rows:
            row = dict(row)
            row["_prompt_style"] = prompt_style
            images = row_images(row, image_size, resize_mode, max_images_per_sample)
            messages = make_generation_messages(len(images), build_prompt(row), system_prompt)
            if hasattr(processor, "apply_chat_template"):
                text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                text = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            texts.append(text)
            batch_images.append(images)
        inputs = processor(text=texts, images=batch_images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": int(kwargs.get("max_new_tokens", 256)),
            "do_sample": float(kwargs.get("temperature", 0.0)) > 0,
            "top_p": float(kwargs.get("top_p", 1.0)),
            "pad_token_id": processor.tokenizer.pad_token_id,
        }
        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = float(kwargs.get("temperature", 0.0))
        with torch.no_grad():
            generated = model.generate(**inputs, **generation_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        decoded = processor.tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        for row, prediction in zip(batch_rows, decoded):
            predictions.append({
                "uid": row["uid"], "prediction": prediction.strip(), "task_type": row.get("task_type", ""),
                "case_id": row.get("case_id", ""), "classifier_task_type": row.get("classifier_task_type", ""),
                "classifier_score": row.get("classifier_score", 0.0),
            })
        done = min(start + len(batch_rows), len(rows))
        if done == len(rows) or done >= next_log_at:
            console.print(f"Generated {done}/{len(rows)} predictions")
            if wandb_run is not None:
                wandb_run.log({
                    "infer/progress": done / len(rows),
                    "infer/generated_predictions": done,
                    f"infer/{split}/progress": done / len(rows),
                    f"infer/{split}/generated_predictions": done,
                })
            while next_log_at <= done:
                next_log_at += log_every
    return predictions
