from typing import Sequence

from mle.vars import ExpConfig
from mle.sanity import check_environment, print_environment_check_results, check_satisfied_or_throw
from mle.engine import preprocess as _preprocess, train as _train, infer as _infer, evaluate as _evaluate


def preprocess(config: ExpConfig, use_wandb: bool, smoke_test: bool, **kwargs) -> None:
    results = check_environment(config)
    print_environment_check_results(results)
    check_satisfied_or_throw(results, True, False, False)
    _preprocess(config, use_wandb, smoke_test, **kwargs)


def train(config: ExpConfig, num_epochs: int, batch_size: int, learning_rate: float, use_wandb: bool, smoke_test: bool,
          **kwargs) -> None:
    results = check_environment(config)
    print_environment_check_results(results)
    check_satisfied_or_throw(results, False, True, True)
    _train(config, num_epochs, batch_size, learning_rate, use_wandb, smoke_test, **kwargs)


def infer(config: ExpConfig, tasks: Sequence[str], use_wandb: bool, smoke_test: bool, **kwargs) -> None:
    results = check_environment(config)
    print_environment_check_results(results)
    check_satisfied_or_throw(results, False, True, True)
    _infer(config, tasks, use_wandb, smoke_test, **kwargs)


def evaluate(config: ExpConfig, tasks: Sequence[str], use_wandb: bool, smoke_test: bool, **kwargs) -> None:
    results = check_environment(config)
    print_environment_check_results(results)
    check_satisfied_or_throw(results, False, True, True)
    _evaluate(config, tasks, use_wandb, smoke_test, **kwargs)
