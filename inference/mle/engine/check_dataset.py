from pathlib import Path

from mle.vars import ExpConfig


def check_dataset(config: ExpConfig) -> str:
    """
    This function checks the availability of the dataset.
    :param config: experiment configuration
    :return: a string indicating the availability of the dataset: if available, it must start with "OK" followed by optional details; otherwise not available
    """
    dataset_dir = Path(config.dataset_dir)
    if not dataset_dir.exists():
        return f"Not found: {dataset_dir}"
    if not dataset_dir.is_dir():
        return f"Invalid: {dataset_dir} is not a directory"

    train_dir = first_existing(dataset_dir, "training", "train")
    validation_dir = first_existing(dataset_dir, "validation-public", "validation_public", "validation", "val")
    if train_dir is None:
        return "Invalid: missing the training split"
    if validation_dir is None:
        return "Invalid: missing the public validation split"

    question_files = list(dataset_dir.rglob("*.json"))
    if not question_files:
        return "Invalid: no question JSON files found"

    extras = []
    if first_existing(dataset_dir, "validation-hidden", "validation_hidden"):
        extras.append("hidden validation")
    if first_existing(dataset_dir, "testing", "test"):
        extras.append("testing")
    suffix = f" with {', '.join(extras)}" if extras else ""
    return f"OK{suffix}; {len(question_files)} JSON file(s)"


def check_preprocessed_dataset(config: ExpConfig) -> str:
    """
    This function checks the availability of the preprocessed dataset.
    :param config: experiment configuration
    :return: a string indicating the availability of the preprocessed dataset: see `check_dataset` for requirements
    """
    preprocessed_dir = Path(config.preprocessed_dataset_dir)
    if not preprocessed_dir.exists():
        return f"Not found: {preprocessed_dir}"
    if not (preprocessed_dir / "train.jsonl").exists() and not (preprocessed_dir / "hf_dataset" / "train").exists():
        return f"Invalid: missing train.jsonl or hf_dataset/train in {preprocessed_dir}"
    details = []
    for split in ("validation", "validation_hidden", "testing"):
        if (preprocessed_dir / f"{split}.jsonl").exists() or (preprocessed_dir / "hf_dataset" / split).exists():
            details.append(split)
    suffix = f" with {', '.join(details)}" if details else ""
    return f"OK{suffix}"


def first_existing(root: Path, *names: str) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None
