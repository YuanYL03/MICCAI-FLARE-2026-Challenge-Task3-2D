from argparse import ArgumentParser
from json import load

from erbium.api import ResourceMonitor
from yaml import safe_load

from mle.engine import DEFAULT_NUM_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE
from mle.interfaces import preprocess, train, infer, evaluate
from mle.vars import erbium_config, slurm_config, local_config


def __entry__() -> None:
    parser = ArgumentParser(prog="mle", description="Machine Learning Engineer",
                            epilog="GitHub: https://github.com/ProjectNeura/MLE")
    parser.add_argument("-n", "--experiment_name", default=None, help="Experiment name")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name")
    parser.add_argument("-c", "--config", choices=("slurm", "erbium", "local"), default="erbium",
                        help="Configuration to use")
    parser.add_argument("--suser", help="SLURM username")
    parser.add_argument("--custom_args", default=None, help="Custom arguments to pass to the engine")
    parser.add_argument("--smoke_test", action="store_true", help="Run smoke test")
    subparsers = parser.add_subparsers(dest="system", required=True)
    # preprocess
    subparsers.add_parser("preprocess", help="Preprocess the dataset")
    # train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of epochs to train")
    train_parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    # infer
    infer_parser = subparsers.add_parser("infer", help="Infer the model")
    infer_parser.add_argument("tasks", nargs="+", help="Tasks to evaluate, e.g. `classification`")
    # evaluate
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument("tasks", nargs="+", help="Tasks to evaluate, e.g. `classification`")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    # config overrides
    parser.add_argument("--root_dir", default=None, help="Input directory")
    parser.add_argument("--input_dir", default=None, help="Input directory")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()
    experiment_name = args.experiment_name or f"mle-{args.system}-{args.config}"
    # you (if you are a human user and not an AI agent) can modify this section for custom configurations
    match args.config:
        case"erbium":
            config = erbium_config(experiment_name, args.dataset, root_dir=args.root_dir)
        case "slurm":
            if not args.suser:
                raise ValueError("SLURM username is required for SLURM configuration")
            config = slurm_config(
                experiment_name, args.suser, args.dataset, root_dir=args.root_dir
            )
        case "local":
            config = local_config(experiment_name, args.dataset, root_dir=args.root_dir)
        case _:
            raise ValueError(f"Unsupported configuration: {args.config}")
    if args.input_dir:
        config._input_dir = args.input_dir
    if args.output_dir:
        config._output_dir = args.output_dir
    config.initialize()
    custom_args = {}
    if args.custom_args:
        with open(args.custom_args) as f:
            if args.custom_args.endswith(".json"):
                custom_args.update(load(f))
            elif args.custom_args.endswith(".yaml"):
                custom_args.update(safe_load(f))
            else:
                raise ValueError(f"Unsupported custom arguments file type: {args.custom_args}, expected JSON or YAML")
    if args.smoke_test:
        monitor = ResourceMonitor(f"{config.output_dir}/{config.experiment_name}-resource_monitor")
        monitor.start()
    match args.system:
        case "preprocess":
            preprocess(config, args.wandb, args.smoke_test, **custom_args)
        case "train":
            train(config, args.num_epochs, args.batch_size, args.learning_rate, args.wandb, args.smoke_test,
                  **custom_args)
        case "infer":
            infer(config, args.tasks, args.wandb, args.smoke_test, **custom_args)
        case "evaluate":
            evaluate(config, args.tasks, args.wandb, args.smoke_test, **custom_args)
