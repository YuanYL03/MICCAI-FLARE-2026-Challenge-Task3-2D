from typing import Any, Callable

import torch
from erbium.api import get_all_gpu_info
from pynvml import NVMLError
from rich.console import Console
from rich.table import Table

from mle.engine import check_dataset, check_preprocessed_dataset
from mle.vars import ExpConfig


def check_environment(config: ExpConfig, *, fn_cd: Callable[[ExpConfig], str] = check_dataset,
                      fn_cpd: Callable[[ExpConfig], str] = check_preprocessed_dataset) -> dict[str, Any]:
    try:
        gpus = get_all_gpu_info()
    except NVMLError:
        gpus = {}
    return {"dataset": fn_cd(config), "preprocessed_dataset": fn_cpd(config), "gpus": gpus,
            "cuda": torch.version.cuda}


def print_environment_check_results(results: dict[str, Any], *, console: Console = Console()) -> None:
    table = Table(title="Available GPUs")
    table.add_column("Name (ID)", justify="left")
    table.add_column("Total Memory (GB)", justify="center", style="cyan")
    table.add_column("Utilization (%)", justify="center", style="magenta")
    table.add_column("Memory Utilization (%)", justify="center", style="green")
    for info in results["gpus"].values():
        table.add_row(
            f"{info.name} ({info.device_id})", f"{info.total_memory_gb:.1f}", f"{info.utilization_percent:.2f}",
            f"{info.memory_utilization_percent:.2f}"
        )
    console.print(table)
    console.print(f"Dataset availability: {results["dataset"]}")
    console.print(f"Preprocessed dataset availability: {results["preprocessed_dataset"]}")
    console.print(f"CUDA version: {results["cuda"]}")


def check_satisfied_or_throw(results: dict[str, Any], dataset: bool, preprocessed_dataset: bool, cuda: bool) -> None:
    if dataset and not results["dataset"].startswith("OK"):
        raise RuntimeError(f"Dataset check failed: {results["dataset"]}")
    if preprocessed_dataset and not results["preprocessed_dataset"].startswith("OK"):
        raise RuntimeError(f"Preprocessed dataset check failed: {results["preprocessed_dataset"]}")
    if cuda and not results["cuda"]:
        raise RuntimeError("CUDA not available")
