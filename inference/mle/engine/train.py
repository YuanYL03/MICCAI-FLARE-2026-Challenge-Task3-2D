from __future__ import annotations

import inspect
import json
import math
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console
from typing import Any, Sequence

from mle.vars import ExpConfig

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
    from datasets import Dataset, load_from_disk
except Exception:  # pragma: no cover
    Dataset = None
    load_from_disk = None

try:
    from torch.utils.data import Sampler
except Exception:  # pragma: no cover
    Sampler = object

try:
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
except Exception:  # pragma: no cover
    LoraConfig = None
    PeftModel = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, EarlyStoppingCallback
except Exception:  # pragma: no cover
    AutoModelForImageTextToText = None
    AutoProcessor = None
    BitsAndBytesConfig = None
    EarlyStoppingCallback = None

try:
    from trl import SFTConfig, SFTTrainer
except Exception:  # pragma: no cover
    SFTConfig = None
    SFTTrainer = None

# these constants MUST NOT be removed but may be modified
DEFAULT_NUM_EPOCHS: int = 1
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_LEARNING_RATE: float = 2e-4

MODEL_ID = "google/medgemma-1.5-4b-it"

TASK_INSTRUCTIONS = {
    "disease_diagnosis_classification": (
        "You are given a medical image. Answer the classification question using only the provided image. "
        "If options are provided, return only the correct option text or class label."
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

# Keep these names stable across all three stages.  The root/default adapter is
# the shared LoRA; each named adapter is one task expert.
MOELORA_TASK_ADAPTERS = {
    "disease_diagnosis_classification": "expert_disease_diagnosis_classification",
    "multi_label_classification": "expert_multi_label_classification",
    "report_generation": "expert_report_generation",
    "detection": "expert_detection",
    "cell_counting": "expert_cell_counting",
    "regression": "expert_regression",
}


def train(
        config: ExpConfig,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        use_wandb: bool,
        smoke_test: bool,
        *,
        console: Console = Console(),
        **kwargs,
) -> None:
    """
    This is a template entrypoint for training. You MUST NOT change its signature, but you may add functions and classes
    to this file.

    All your logs MUST be sent to the provided console. Your implementation MUST support WandB logging and it MUST ONLY
    be enabled if :param use_wandb is `True`.

    :param config: experiment configuration
    :param num_epochs: the number of epochs to train for
    :param batch_size: the batch size for training
    :param learning_rate: the learning rate for training
    :param use_wandb: whether to use wandb for logging
    :param smoke_test: whether to run in smoke test mode
    :param console: the console for logging
    :param kwargs: custom arguments
    """
    missing = dependency_gaps()
    if missing:
        raise RuntimeError("Missing training dependencies: " + ", ".join(missing))
    if not torch.cuda.is_available() and not bool(kwargs.get("allow_cpu", False)):
        raise RuntimeError(
            "CUDA is required for MedGemma fine-tuning. Use a Slurm GPU job or pass allow_cpu=true for tiny dry runs.")

    # In torchrun/DDP, Transformers + bitsandbytes otherwise default each
    # process to cuda:0 while loading a quantized model. Bind before model
    # construction so LOCAL_RANK=1 owns cuda:1.
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)

    output_dir = Path(kwargs.get("model_output_dir") or kwargs.get("output_dir") or Path(
        config.output_dir) / f"{config.experiment_name}-medgemma15-lora")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(kwargs.get("seed", 42))
    seed_everything(seed)
    configure_wandb(config, output_dir, use_wandb, kwargs)
    model_name_or_path = str(kwargs.get("model_name_or_path", MODEL_ID))
    image_size = int(kwargs.get("image_size", 512 if smoke_test else 896))
    resize_mode = str(kwargs.get("resize_mode", "square"))
    max_images_per_sample = int(kwargs.get("max_images_per_sample", 1))
    max_length = int(kwargs.get("max_length", 0))
    max_train_samples = optional_int(kwargs.get("max_train_samples", 8 if smoke_test else None))
    max_eval_samples = optional_int(kwargs.get("max_eval_samples", 4 if smoke_test else 256))
    per_device_eval_batch_size = int(kwargs.get("per_device_eval_batch_size", 1))
    gradient_accumulation_steps = int(kwargs.get("gradient_accumulation_steps", 1 if smoke_test else 16))
    max_steps = int(kwargs.get("max_steps", 2 if smoke_test else -1))
    per_device_train_batch_size = int(kwargs.get("per_device_train_batch_size", batch_size))
    moelora_enable = bool(kwargs.get("moelora_enable", False))
    moelora_source_adapter = str(kwargs.get("moelora_source_adapter", "")).strip()
    moelora_preserve_optimizer_state = bool(kwargs.get("moelora_preserve_optimizer_state", False))
    moelora_target_task = None
    if moelora_source_adapter:
        if not moelora_enable:
            raise ValueError("moelora_source_adapter requires moelora_enable=true.")
        moelora_target_task = normalize_task_type({"task_type": kwargs.get("moelora_target_task", "")})
        if moelora_target_task not in MOELORA_TASK_ADAPTERS:
            raise ValueError(
                "moelora_target_task must be one of "
                f"{sorted(MOELORA_TASK_ADAPTERS)}."
            )
    if smoke_test:
        console.print("Smoke test mode: limiting training samples, steps, and evaluation workload.")

    console.print(f"Loading converted FLARE-MLLM-2D data from {config.preprocessed_dataset_dir}")
    train_dataset, eval_dataset = load_splits(Path(config.preprocessed_dataset_dir), max_train_samples,
                                              max_eval_samples)
    if moelora_target_task is not None:
        train_dataset = train_dataset.filter(
            lambda row: normalize_task_type(row) == moelora_target_task,
            desc=f"Selecting {moelora_target_task} rows",
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(
                lambda row: normalize_task_type(row) == moelora_target_task,
                desc=f"Selecting validation {moelora_target_task} rows",
            )
        if len(train_dataset) == 0:
            raise RuntimeError(f"No training rows found for {moelora_target_task}.")
    console.print(f"Loaded {len(train_dataset)} training row(s)" + (
        f" and {len(eval_dataset)} validation row(s)" if eval_dataset else ""))

    dtype = choose_dtype(kwargs.get("precision", kwargs.get("torch_dtype", kwargs.get("dtype", "auto"))), console)
    attn_implementations = choose_attention_backends(str(kwargs.get("attn_implementation", "auto")))
    quant_config = make_quant_config(bool(kwargs.get("load_in_4bit", True)), dtype)
    warmup_steps = resolve_warmup_steps(
        kwargs.get("warmup_steps"),
        kwargs.get("warmup_ratio", 0.03),
        len(train_dataset),
        per_device_train_batch_size,
        gradient_accumulation_steps,
        float(num_epochs),
        max_steps,
    )

    console.print(f"Loading {model_name_or_path} with attention={attn_label(attn_implementations[0])}")
    model = load_model_with_attention_fallback(
        model_name_or_path,
        attn_implementations,
        console,
        torch_dtype=dtype,
        device_map=None if torch.cuda.is_available() else "cpu",
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    gradient_checkpointing = bool(kwargs.get("gradient_checkpointing", True))
    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    target_modules: Any = kwargs.get("target_modules", "all-linear")
    if isinstance(target_modules, str) and "," in target_modules:
        target_modules = [item.strip() for item in target_modules.split(",") if item.strip()]
    modules_to_save = split_csv(kwargs.get("modules_to_save", ""))
    peft_config = LoraConfig(
        r=int(kwargs.get("lora_rank", 8 if smoke_test else 16)),
        lora_alpha=int(kwargs.get("lora_alpha", 8 if smoke_test else 16)),
        lora_dropout=float(kwargs.get("lora_dropout", 0.05)),
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save or None,
    )

    collator_class = MoELoraImageSFTCollator if moelora_enable else ImageSFTCollator
    collator = collator_class(
        processor=processor,
        image_size=image_size,
        resize_mode=resize_mode,
        max_images_per_sample=max_images_per_sample,
        max_length=max_length if max_length > 0 else None,
        mask_prompt_tokens=bool(kwargs.get("mask_prompt_tokens", True)),
    )

    requested_eval_strategy = str(kwargs.get("eval_strategy", "")).strip().lower()
    eval_strategy = requested_eval_strategy or ("steps" if eval_dataset is not None else "no")
    if eval_strategy not in {"no", "steps", "epoch"}:
        raise ValueError("eval_strategy must be one of: no, steps, epoch")
    early_stopping_patience = int(kwargs.get("early_stopping_patience", 0))
    use_early_stopping = eval_dataset is not None and early_stopping_patience > 0
    if use_early_stopping and EarlyStoppingCallback is None:
        raise RuntimeError("EarlyStoppingCallback is unavailable; install or upgrade transformers.")

    training_args = make_sft_config_compat(
        output_dir=str(output_dir),
        run_name=str(kwargs.get("run_name", config.experiment_name)),
        num_train_epochs=float(num_epochs),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(learning_rate),
        weight_decay=float(kwargs.get("weight_decay", 0.0)),
        warmup_steps=warmup_steps,
        lr_scheduler_type=str(kwargs.get("lr_scheduler_type", "linear")),
        optim=str(kwargs.get("optim", "paged_adamw_8bit" if quant_config is not None else "adamw_torch_fused")),
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        max_grad_norm=float(kwargs.get("max_grad_norm", 0.3)),
        logging_steps=int(kwargs.get("logging_steps", 1 if smoke_test else 10)),
        save_strategy=str(kwargs.get("save_strategy", "steps")),
        save_steps=int(kwargs.get("save_steps", 100000 if smoke_test else 200)),
        save_total_limit=int(kwargs.get("save_total_limit", 1 if smoke_test else 3)),
        eval_strategy=eval_strategy,
        eval_steps=int(kwargs.get("eval_steps", 100000 if smoke_test else 200)),
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss" if use_early_stopping else None,
        greater_is_better=False if use_early_stopping else None,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        label_names=["labels"],
        report_to="wandb" if use_wandb else "none",
        push_to_hub=bool(kwargs.get("push_to_hub", False)),
        hub_model_id=kwargs.get("hub_model_id"),
        dataloader_num_workers=int(kwargs.get("dataloader_num_workers", 0 if smoke_test else 0)),
        dataloader_pin_memory=bool(kwargs.get("dataloader_pin_memory", False)),
        group_by_length=False,
        packing=False,
        max_seq_length=max_length if max_length > 0 else None,
    )

    callbacks = []
    if use_early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=float(kwargs.get("early_stopping_threshold", 0.0)),
        ))

    if moelora_source_adapter:
        if PeftModel is None:
            raise RuntimeError("PEFT PeftModel is unavailable for MoE-LoRA continuation training.")
        source_checkpoint = Path(moelora_source_adapter)
        if not (source_checkpoint / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing shared/default adapter in {source_checkpoint}")
        model = PeftModel.from_pretrained(model, str(source_checkpoint), is_trainable=False, local_files_only=True)
        for task_name, adapter_name in MOELORA_TASK_ADAPTERS.items():
            expert_dir = source_checkpoint / adapter_name
            if not (expert_dir / "adapter_config.json").is_file():
                raise FileNotFoundError(f"Missing {adapter_name} in {source_checkpoint}")
            model.load_adapter(
                str(expert_dir),
                adapter_name=adapter_name,
                is_trainable=(task_name == moelora_target_task),
            )
        target_adapter = MOELORA_TASK_ADAPTERS[moelora_target_task]
        trainable_lora_tensors = 0
        for name, parameter in model.named_parameters():
            # To import the Stage-1 optimizer moments, first construct an
            # optimizer over the exact same full LoRA parameter ordering. The
            # non-target adapters are frozen immediately after that import.
            parameter.requires_grad = bool(
                "lora_" in name
                and (moelora_preserve_optimizer_state or f".{target_adapter}." in name)
            )
            trainable_lora_tensors += int(parameter.requires_grad)
        if trainable_lora_tensors == 0:
            raise RuntimeError(f"No trainable LoRA tensors found for {target_adapter}.")
        model.base_model.set_adapter(["default", target_adapter])
        console.print(
            f"Loaded frozen shared MoE-LoRA from {source_checkpoint}; frozen shared/default "
            f"and all non-target experts; training {target_adapter} only "
            f"({trainable_lora_tensors} initially trainable LoRA tensors)."
        )
    elif moelora_enable:
        if get_peft_model is None:
            raise RuntimeError("PEFT get_peft_model is unavailable for MoE-LoRA training.")
        model = get_peft_model(model, peft_config)
        for adapter_name in MOELORA_TASK_ADAPTERS.values():
            model.add_adapter(adapter_name, deepcopy(peft_config))
        # The optimizer is constructed by Trainer after this point. Keep every
        # LoRA tensor in it; the routed forward pass makes only its expert and
        # the shared/default adapter contribute gradients on each micro-batch.
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True
        model.base_model.set_adapter(["default", next(iter(MOELORA_TASK_ADAPTERS.values()))])
        console.print(
            "Configured additive MoE-LoRA: shared/default + "
            + ", ".join(MOELORA_TASK_ADAPTERS.values())
        )

    trainer = make_sft_trainer_compat(
        trainer_cls=MedGemmaMoELoraSFTTrainer if moelora_enable else SFTTrainer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None if moelora_enable else peft_config,
        processing_class=processor,
        data_collator=collator,
        callbacks=callbacks,
    )
    if moelora_source_adapter and moelora_preserve_optimizer_state:
        optimizer_path = Path(moelora_source_adapter) / "optimizer.pt"
        if not optimizer_path.is_file():
            raise FileNotFoundError(f"Requested optimizer-state preservation but missing {optimizer_path}")
        # The Stage-1 optimizer covers shared + every expert in this exact
        # adapter creation order. Import moments before switching gradients to
        # the sole target expert; the new LR/scheduler remains intentionally
        # fresh for the continuation experiment.
        trainer.create_optimizer()
        try:
            optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        except TypeError:  # older PyTorch versions
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
        trainer.optimizer.load_state_dict(optimizer_state)
        for group in trainer.optimizer.param_groups:
            group["lr"] = float(learning_rate)
            group.pop("initial_lr", None)
        target_adapter = MOELORA_TASK_ADAPTERS[moelora_target_task]
        final_trainable = 0
        for name, parameter in trainer.model.named_parameters():
            parameter.requires_grad = bool("lora_" in name and f".{target_adapter}." in name)
            final_trainable += int(parameter.requires_grad)
        if final_trainable == 0:
            raise RuntimeError(f"No target-expert LoRA tensors remain trainable for {target_adapter}.")
        console.print(
            f"Restored Stage-1 optimizer moments from {optimizer_path}; reset LR to {learning_rate:g}; "
            f"frozen shared/default and all non-target experts; training {target_adapter} only "
            f"({final_trainable} LoRA tensors)."
        )
    prepare_trainable_parameters_for_amp(trainer.model, dtype, console)

    console.print("Starting MedGemma LoRA fine-tuning")
    trainer.train(resume_from_checkpoint=kwargs.get("resume_from_checkpoint"))
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    console.print(f"Saved final adapter and processor to {final_dir}")


def dependency_gaps() -> list[str]:
    gaps = []
    if torch is None:
        gaps.append("torch")
    if Image is None or ImageOps is None:
        gaps.append("Pillow")
    if Dataset is None or load_from_disk is None:
        gaps.append("datasets")
    if LoraConfig is None or prepare_model_for_kbit_training is None:
        gaps.append("peft")
    if AutoModelForImageTextToText is None or AutoProcessor is None or BitsAndBytesConfig is None:
        gaps.append("transformers")
    if SFTConfig is None or SFTTrainer is None:
        gaps.append("trl")
    return gaps


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    out = int(value)
    return out if out > 0 else None


def optional_nonnegative_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    out = int(value)
    return out if out >= 0 else None


def split_csv(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_wandb(config: ExpConfig, output_dir: Path, use_wandb: bool, kwargs: dict[str, Any]) -> None:
    if use_wandb:
        os.environ.setdefault("WANDB_DIR", str(output_dir / "wandb"))
        os.environ.setdefault("WANDB_PROJECT", str(kwargs.get("wandb_project", "medgemma15-flare-mllm-2d")))
        os.environ.setdefault("WANDB_RUN_NAME", str(kwargs.get("wandb_run_name", config.experiment_name)))
        os.environ.setdefault("WANDB_INIT_TIMEOUT", str(kwargs.get("wandb_init_timeout", 300)))
        for env_name, key in (("WANDB_ENTITY", "wandb_entity"), ("WANDB_MODE", "wandb_mode"),
                              ("WANDB_TAGS", "wandb_tags")):
            if kwargs.get(key):
                os.environ.setdefault(env_name, str(kwargs[key]))
        os.environ.setdefault("WANDB_LOG_MODEL", str(kwargs.get("wandb_log_model", "checkpoint")))
    else:
        os.environ.setdefault("WANDB_DISABLED", "true")


def make_sft_config_compat(**kwargs):
    sig = inspect.signature(SFTConfig.__init__)
    params = sig.parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    allowed = set(params) - {"self"}
    config = dict(kwargs)
    if "max_seq_length" in config and "max_seq_length" not in allowed and "max_length" in allowed:
        config["max_length"] = config.pop("max_seq_length")
    if "eval_strategy" in config and "eval_strategy" not in allowed and "evaluation_strategy" in allowed:
        config["evaluation_strategy"] = config.pop("eval_strategy")
    if not accepts_kwargs:
        config = {key: value for key, value in config.items() if key in allowed}
    return SFTConfig(**config)


def make_sft_trainer_compat(trainer_cls=SFTTrainer, **kwargs):
    sig = inspect.signature(trainer_cls.__init__)
    params = sig.parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    allowed = set(params) - {"self"}
    config = dict(kwargs)
    if "processing_class" in config and "processing_class" not in allowed and "tokenizer" in allowed:
        config["tokenizer"] = config.pop("processing_class")
    if not accepts_kwargs:
        config = {key: value for key, value in config.items() if key in allowed}
    return trainer_cls(**config)


def choose_dtype(requested: Any = "auto", console: Console | None = None):
    requested_text = str(requested or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "torch.bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
        "torch.float16": "fp16",
        "half": "fp16",
        "fp32": "fp32",
        "float32": "fp32",
        "torch.float32": "fp32",
        "full": "fp32",
    }
    if requested_text not in aliases:
        raise ValueError("precision must be one of: auto, bf16, fp16, fp32")
    precision = aliases[requested_text]
    if not torch.cuda.is_available():
        if precision != "fp32" and console is not None:
            console.print("CUDA is unavailable; using fp32 precision.")
        return torch.float32

    major, minor = torch.cuda.get_device_capability()
    supports_bf16 = cuda_supports_bf16()
    if precision == "auto":
        dtype = torch.bfloat16 if supports_bf16 else torch.float16
    elif precision == "bf16":
        if not supports_bf16:
            if console is not None:
                console.print(f"bf16 was requested but CUDA capability sm_{major}{minor} does not support it; using fp16.")
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    if console is not None:
        console.print(f"Using {dtype_label(dtype)} precision on CUDA capability sm_{major}{minor}.")
    return dtype


def cuda_supports_bf16() -> bool:
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        major, _minor = torch.cuda.get_device_capability()
        return major >= 8


def dtype_label(dtype: Any) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype)


def resolve_warmup_steps(
        requested_steps: Any,
        requested_ratio: Any,
        train_size: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        num_epochs: float,
        max_steps: int,
) -> int:
    warmup_steps = optional_nonnegative_int(requested_steps)
    if warmup_steps is not None:
        return warmup_steps
    warmup_ratio = float(requested_ratio or 0.0)
    if warmup_ratio <= 0.0:
        return 0
    if max_steps > 0:
        total_steps = max_steps
    else:
        effective_batch_size = max(1, per_device_train_batch_size) * max(1, gradient_accumulation_steps)
        steps_per_epoch = max(1, math.ceil(train_size / effective_batch_size))
        total_steps = max(1, math.ceil(steps_per_epoch * max(0.0, num_epochs)))
    return int(math.ceil(total_steps * warmup_ratio))


def prepare_trainable_parameters_for_amp(model: Any, dtype: Any, console: Console) -> None:
    if dtype != torch.float16:
        return
    converted_tensors = 0
    converted_params = 0
    for parameter in model.parameters():
        if not parameter.requires_grad or not parameter.is_floating_point():
            continue
        if parameter.dtype not in {torch.float16, torch.bfloat16}:
            continue
        converted_tensors += 1
        converted_params += parameter.numel()
        with torch.no_grad():
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()
    if converted_tensors:
        console.print(
            f"Promoted {converted_tensors} trainable tensor(s) ({converted_params:,} parameters) to fp32 for fp16 AMP."
        )


def choose_attention_backends(requested: str) -> list[str | None]:
    requested = requested.strip().lower()
    if requested == "auto":
        return ["sdpa", "eager", None]
    if requested in {"default", "none", "transformers-default"}:
        return [None]
    if requested == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            return ["flash_attention_2", "sdpa", "eager", None]
        except Exception:
            return ["sdpa", "eager", None]
    return unique_attention_backends([requested, "sdpa", "eager", None])


def unique_attention_backends(backends: Sequence[str | None]) -> list[str | None]:
    out = []
    for backend in backends:
        if backend not in out:
            out.append(backend)
    return out


def attn_label(attn_implementation: str | None) -> str:
    return attn_implementation or "transformers-default"


def load_model_with_attention_fallback(model_name_or_path: str, attn_implementations: Sequence[str | None],
                                       console: Console, **kwargs):
    last_error: Exception | None = None
    for index, attn_implementation in enumerate(attn_implementations):
        model_kwargs = dict(kwargs)
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        try:
            if index > 0:
                console.print(f"Retrying model load with attention={attn_label(attn_implementation)}")
            return AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)
        except ValueError as exc:
            last_error = exc
            message = str(exc)
            if "attn" not in message.lower() and "attention" not in message.lower():
                raise
            console.print(f"Attention backend {attn_label(attn_implementation)} is unsupported here: {message}")
    if last_error is not None:
        raise last_error
    raise RuntimeError("No attention backends were provided for model loading.")


def make_quant_config(load_in_4bit: bool, dtype: Any):
    if not load_in_4bit or not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )


def maybe_json_load(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
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


def arrow_safe_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def arrow_safe_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    keys = sorted({key for row in rows for key in row})
    return [{key: arrow_safe_value(row.get(key)) for key in keys} for row in rows]


def find_dataset_sources(data_dir: Path, train_name: str = "train", val_name: str = "validation") -> tuple[
    Path, Path | None, str]:
    for base in (data_dir / "hf_dataset", data_dir):
        train_dir = base / train_name
        val_dir = base / val_name
        if train_dir.exists():
            return train_dir, val_dir if val_dir.exists() else None, "hf"
    train_jsonl = data_dir / f"{train_name}.jsonl"
    val_jsonl = data_dir / f"{val_name}.jsonl"
    if train_jsonl.exists():
        return train_jsonl, val_jsonl if val_jsonl.exists() else None, "jsonl"
    raise FileNotFoundError(f"Could not find train split under {data_dir}")


def load_splits(data_dir: Path, max_train_samples: int | None, max_eval_samples: int | None):
    train_src, val_src, kind = find_dataset_sources(data_dir)
    if kind == "hf":
        train_dataset = load_from_disk(str(train_src))
        eval_dataset = load_from_disk(str(val_src)) if val_src else None
    else:
        train_dataset = Dataset.from_list(arrow_safe_records(read_jsonl(train_src)))
        eval_dataset = Dataset.from_list(arrow_safe_records(read_jsonl(val_src))) if val_src else None
    if max_train_samples:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if eval_dataset is not None and max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))
    return train_dataset, eval_dataset


def get_image_paths(row: dict[str, Any]) -> list[str]:
    images = maybe_json_load(row.get("images"), default=[])
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
    raise KeyError(f"No image path found for uid={row.get('uid', '<unknown>')}")


def load_image(path: str, image_size: int, resize_mode: str):
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


def load_row_images(row: dict[str, Any], image_size: int, resize_mode: str, max_images_per_sample: int):
    paths = get_image_paths(row)
    if max_images_per_sample and max_images_per_sample > 0:
        paths = paths[:max_images_per_sample]
    return [load_image(path, image_size, resize_mode) for path in paths]


def normalize_task_type(row: dict[str, Any]) -> str:
    task = row.get("task_type") or row.get("task") or row.get("subtask") or "disease_diagnosis_classification"
    task = str(task).strip().lower().replace(" ", "_").replace("-", "_")
    return {
        "classification": "disease_diagnosis_classification",
        "cls": "disease_diagnosis_classification",
        "multi_label": "multi_label_classification",
        "multilabel": "multi_label_classification",
        "instance_detection": "detection",
        "count": "cell_counting",
        "counting": "cell_counting",
        "report": "report_generation",
    }.get(task, task)


def build_prompt(row: dict[str, Any]) -> str:
    task = normalize_task_type(row)
    prompt = as_text(row.get("prompt") or row.get("question") or "")
    choices = maybe_json_load(row.get("choices"), default=[])
    if not isinstance(choices, list):
        choices = []
    parts = [TASK_INSTRUCTIONS.get(task, "Answer the medical imaging question using the provided image.")]
    if prompt:
        parts.append(prompt)
    if choices and "options:" not in prompt.lower():
        parts.append("Options: " + "; ".join(str(choice) for choice in choices))
    return "\n\n".join(parts)


def extract_answer(row: dict[str, Any]) -> str:
    answer = maybe_json_load(row.get("raw_answer"), default=None)
    if answer is None:
        answer = row.get("answer", "")
    if isinstance(answer, list):
        return "; ".join(str(item) for item in answer)
    if isinstance(answer, dict):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def make_messages(num_images: int, prompt: str, answer: str, system_prompt: str) -> list[dict[str, Any]]:
    content = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def find_subsequence(sequence, pattern: Sequence[int]) -> int:
    if not pattern:
        return -1
    target = torch.tensor(pattern, device=sequence.device, dtype=sequence.dtype)
    length = len(pattern)
    for index in range(0, sequence.numel() - length + 1):
        if torch.equal(sequence[index:index + length], target):
            return index
    return -1


@dataclass
class ImageSFTCollator:
    processor: Any
    image_size: int = 896
    resize_mode: str = "square"
    max_images_per_sample: int = 1
    max_length: int | None = None
    system_prompt: str = "You are an expert medical imaging assistant."
    mask_prompt_tokens: bool = True

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        batch_images = []
        for example in examples:
            images = load_row_images(example, self.image_size, self.resize_mode, self.max_images_per_sample)
            messages = make_messages(len(images), build_prompt(example), extract_answer(example), self.system_prompt)
            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False).strip()
            else:
                text = self.processor.tokenizer.apply_chat_template(messages, add_generation_prompt=False,
                                                                    tokenize=False).strip()
            texts.append(text)
            batch_images.append(images)

        proc_kwargs = {"text": texts, "images": batch_images, "return_tensors": "pt", "padding": True}
        if self.max_length:
            proc_kwargs.update({"truncation": True, "max_length": self.max_length})
        batch = self.processor(**proc_kwargs)
        labels = batch["input_ids"].clone()
        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        for token_id in image_token_ids(tokenizer):
            labels[labels == token_id] = -100

        if self.mask_prompt_tokens:
            marker_lists = [tokenizer.encode(marker, add_special_tokens=False) for marker in
                            ("<start_of_turn>model\n", "model\n")]
            for row_index in range(labels.shape[0]):
                for marker_tokens in marker_lists:
                    found = find_subsequence(batch["input_ids"][row_index], marker_tokens)
                    if found >= 0:
                        labels[row_index, :found + len(marker_tokens)] = -100
                        break

        batch["labels"] = labels
        return batch


@dataclass
class MoELoraImageSFTCollator(ImageSFTCollator):
    """Attach a single task route to each MoE-LoRA micro-batch."""

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        task_names = [normalize_task_type(example) for example in examples]
        if len(set(task_names)) != 1:
            raise ValueError(
                "A MedGemma MoE-LoRA micro-batch contains multiple tasks; "
                "use per_device_train_batch_size=1."
            )
        task_name = task_names[0]
        if task_name not in MOELORA_TASK_ADAPTERS:
            raise ValueError(
                f"Unsupported MedGemma MoE-LoRA task {task_name!r}; "
                f"expected one of {sorted(MOELORA_TASK_ADAPTERS)}."
            )
        batch = super().__call__(examples)
        batch["moelora_task"] = task_name
        return batch


class TaskGroupedSampler(Sampler):
    """Shuffle complete, task-homogeneous mini-batches without mixing tasks."""

    def __init__(self, dataset, batch_size: int, seed: int = 42):
        if batch_size < 1:
            raise ValueError("TaskGroupedSampler batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.groups: dict[str, list[int]] = {}
        for index in range(len(dataset)):
            task_name = normalize_task_type(dataset[index])
            if task_name not in MOELORA_TASK_ADAPTERS:
                raise ValueError(
                    f"Unsupported MedGemma MoE-LoRA task {task_name!r} at dataset index {index}."
                )
            self.groups.setdefault(task_name, []).append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in self.groups.values():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            full = len(shuffled) - (len(shuffled) % self.batch_size)
            batches.extend(
                shuffled[start:start + self.batch_size]
                for start in range(0, full, self.batch_size)
            )
        rng.shuffle(batches)
        for batch in batches:
            yield from batch

    def __len__(self) -> int:
        return sum((len(indices) // self.batch_size) * self.batch_size for indices in self.groups.values())


class MedGemmaMoELoraSFTTrainer(SFTTrainer):
    """Route each micro-batch through the shared adapter plus its task expert."""

    def _get_train_sampler(self, *args, **kwargs):
        if self.train_dataset is None:
            return None
        return TaskGroupedSampler(
            self.train_dataset,
            batch_size=int(self.args.per_device_train_batch_size),
            seed=int(self.args.seed),
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        task_name = inputs.pop("moelora_task")
        adapter_name = MOELORA_TASK_ADAPTERS.get(task_name)
        if adapter_name is None:
            raise ValueError(f"Unknown MedGemma MoE-LoRA task: {task_name!r}")
        peft_model = self.accelerator.unwrap_model(model, keep_torch_compile=False)
        # PeftModel.set_adapter accepts one adapter in some PEFT releases, but
        # its underlying LoraModel supports the additive shared+expert route.
        peft_model.base_model.set_adapter(["default", adapter_name])
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


def image_token_ids(tokenizer: Any) -> set[int]:
    token_ids = set()
    special_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    for key in ("boi_token", "image_token"):
        token = special_map.get(key)
        if token:
            try:
                token_ids.add(int(tokenizer.convert_tokens_to_ids(token)))
            except Exception:
                pass
    token_ids.add(262144)
    return {token_id for token_id in token_ids if token_id >= 0}
