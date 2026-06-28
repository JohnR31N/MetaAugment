from __future__ import annotations

import argparse

from meta_augment.config import load_config, override_config
from meta_augment.training.trainer import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MetaAugment with JAX/Flax.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--workdir", type=str, default=None, help="Directory for logs and checkpoints.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory for CIFAR data.")
    parser.add_argument("--competitor", type=str, default=None, help="Training competitor name.")
    parser.add_argument(
        "--model-architecture",
        type=str,
        default=None,
        help="wide_resnet, preact_resnet, or preact_resnet18.",
    )
    parser.add_argument("--model-depth", type=int, default=None, help="Override model.depth.")
    parser.add_argument("--model-width", type=int, default=None, help="Override model.width.")
    parser.add_argument("--epochs", type=int, default=None, help="Override optim.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override data.eval_batch_size.")
    parser.add_argument("--save-every-epochs", type=int, default=None, help="Override system.save_every_epochs.")
    parser.add_argument("--keep-checkpoints", type=int, default=None, help="How many checkpoints to keep.")
    parser.add_argument("--no-pmap", action="store_true", help="Use one local device only.")
    parser.add_argument("--init-distributed", action="store_true", help="Call jax.distributed.initialize().")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a tiny synthetic-data job to test imports, JIT, and gradients.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.workdir is not None:
        config = override_config(config, "system.workdir", args.workdir)
    if args.data_dir is not None:
        config = override_config(config, "data.data_dir", args.data_dir)
    if args.competitor is not None:
        config = override_config(config, "competitor.name", args.competitor)
    if args.model_architecture is not None:
        config = override_config(config, "model.architecture", args.model_architecture)
    if args.model_depth is not None:
        config = override_config(config, "model.depth", args.model_depth)
    if args.model_width is not None:
        config = override_config(config, "model.width", args.model_width)
    if args.epochs is not None:
        config = override_config(config, "optim.epochs", args.epochs)
    if args.batch_size is not None:
        config = override_config(config, "data.batch_size", args.batch_size)
    if args.eval_batch_size is not None:
        config = override_config(config, "data.eval_batch_size", args.eval_batch_size)
    if args.save_every_epochs is not None:
        config = override_config(config, "system.save_every_epochs", args.save_every_epochs)
    if args.keep_checkpoints is not None:
        config = override_config(config, "system.keep_checkpoints", args.keep_checkpoints)
    if args.no_pmap:
        config = override_config(config, "system.use_pmap", False)
    if args.init_distributed:
        config = override_config(config, "system.init_distributed", True)
    if args.fast_dev_run:
        config = override_config(config, "data.dataset", "synthetic")
        config = override_config(config, "data.val_size", 32)
        config = override_config(config, "data.batch_size", 8)
        config = override_config(config, "data.eval_batch_size", 8)
        architecture = config.model.architecture.lower()
        if args.model_architecture is None:
            config = override_config(config, "model.architecture", "wide_resnet")
            architecture = "wide_resnet"
        if args.model_depth is None:
            depth = 20 if architecture in {"preact_resnet", "preact-resnet", "preact"} else 10
            if architecture in {"preact_resnet18", "preact-resnet18", "preact18", "preact_resnet_18"}:
                depth = 18
            config = override_config(config, "model.depth", depth)
        if args.model_width is None:
            config = override_config(config, "model.width", 1)
        config = override_config(config, "optim.epochs", 1)
        config = override_config(config, "system.log_every", 1)
        config = override_config(config, "system.eval_every_epochs", 1)
        config = override_config(config, "system.save_every_epochs", 1)
        config = override_config(config, "system.keep_checkpoints", 1)
        config = override_config(config, "system.workdir", "runs/fast_dev_run")
    train_and_evaluate(config)


if __name__ == "__main__":
    main()
