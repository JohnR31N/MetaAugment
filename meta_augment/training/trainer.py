from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from flax.training import train_state

from meta_augment.config import Config, to_dict
from meta_augment.data.augmentations import NUM_OPS, apply_metaaugment, initial_sampler_probs
from meta_augment.data.loaders.cifar_loader import (
    CyclingBatcher,
    iter_batches,
    load_cifar_dataset,
    shard_batch,
)
from meta_augment.data.preprocessors.cifar_preprocessor import normalize_images
from meta_augment.networks.policy_network import PolicyNetwork
from meta_augment.networks.task_networks.backbones.wide_resnet_28_10 import WideResNet
from meta_augment.training.losses import cross_entropy_loss, normalized_policy_weights
from meta_augment.training.metrics import topk_accuracy, topk_accuracy_sum
from meta_augment.training.optimizers import cosine_schedule, sgd_tx


class TaskTrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict[str, Any]


class PolicyTrainState(train_state.TrainState):
    pass


def _host0_print(*values: Any) -> None:
    if jax.process_index() == 0:
        print(*values, flush=True)


def _softplus_inverse(value: float) -> float:
    return float(np.log(np.expm1(value)))


def _create_states(
    config: Config,
    num_classes: int,
    steps_per_epoch: int,
    axis_name: str | None,
) -> tuple[WideResNet, PolicyNetwork, TaskTrainState, PolicyTrainState]:
    rng = jax.random.PRNGKey(config.system.seed)
    rng_model, rng_policy, rng_dropout = jax.random.split(rng, 3)
    model = WideResNet(
        depth=config.model.depth,
        width=config.model.width,
        num_classes=num_classes,
        dropout_rate=config.model.dropout_rate,
        axis_name=axis_name,
    )
    variables = model.init(
        {"params": rng_model, "dropout": rng_dropout},
        jnp.ones((1, 32, 32, 3), dtype=jnp.float32),
        train=False,
        return_features=True,
    )

    task_schedule = cosine_schedule(
        base_learning_rate=config.optim.task_learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=config.optim.epochs,
        warmup_epochs=config.optim.warmup_epochs,
    )
    task_state = TaskTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=sgd_tx(
            learning_rate=task_schedule,
            momentum=config.optim.momentum,
            weight_decay=config.optim.weight_decay,
            max_grad_norm=config.optim.max_grad_norm,
        ),
        batch_stats=variables["batch_stats"],
    )

    _, dummy_features = model.apply(
        variables,
        jnp.ones((1, 32, 32, 3), dtype=jnp.float32),
        train=False,
        return_features=True,
    )
    policy = PolicyNetwork()
    policy_params = policy.init(
        rng_policy,
        dummy_features,
        jnp.ones((1, NUM_OPS * 2), dtype=jnp.float32),
    )["params"]
    meta_params = {
        "policy": policy_params,
        "inner_log_lr": jnp.asarray(_softplus_inverse(config.optim.inner_learning_rate), dtype=jnp.float32),
    }
    policy_state = PolicyTrainState.create(
        apply_fn=policy.apply,
        params=meta_params,
        tx=sgd_tx(
            learning_rate=config.optim.policy_learning_rate,
            momentum=config.optim.momentum,
            weight_decay=config.optim.weight_decay,
            max_grad_norm=config.optim.max_grad_norm,
        ),
    )
    return model, policy, task_state, policy_state


def _create_train_step(
    model: WideResNet,
    policy: PolicyNetwork,
    config: Config,
    *,
    axis_name: str,
):
    num_classes = 10 if config.data.dataset.lower() in {"cifar10", "synthetic"} else 100
    dataset = config.data.dataset.lower()

    def forward_with_features(params, batch_stats, images, train, rng):
        variables = {"params": params, "batch_stats": batch_stats}
        if train:
            (logits, features), new_state = model.apply(
                variables,
                images,
                train=True,
                return_features=True,
                mutable=["batch_stats"],
                rngs={"dropout": rng},
            )
            return logits, features, new_state["batch_stats"]
        logits, features = model.apply(
            variables,
            images,
            train=False,
            return_features=True,
        )
        return logits, features, batch_stats

    def weighted_aug_loss(task_params, batch_stats, policy_params, images, labels, embeds, rng):
        logits, features, new_batch_stats = forward_with_features(
            task_params, batch_stats, normalize_images(images, dataset), True, rng
        )
        raw_weights = policy.apply(
            {"params": policy_params},
            jax.lax.stop_gradient(features),
            embeds,
        )
        losses = cross_entropy_loss(
            logits,
            labels,
            num_classes=num_classes,
            label_smoothing=config.optim.label_smoothing,
        )
        weights = normalized_policy_weights(raw_weights)
        loss = jnp.sum(weights * losses)
        return loss, (logits, raw_weights, new_batch_stats)

    def step(task_state, policy_state, sampler_probs, train_batch, val_batch, rng):
        rng_aug, rng_inner, rng_task = jax.random.split(rng, 3)
        aug_images, aug_labels, embeds, pair_ids = apply_metaaugment(
            train_batch["image"],
            train_batch["label"],
            rng_aug,
            sampler_probs,
            num_transforms_per_sample=config.augment.num_transforms_per_sample,
            cutout_size=config.augment.cutout_size,
            translate_const=config.augment.translate_const,
        )

        def policy_loss_fn(meta_params):
            inner_lr = jax.nn.softplus(meta_params["inner_log_lr"])

            def inner_task_loss(params):
                loss, _ = weighted_aug_loss(
                    params,
                    task_state.batch_stats,
                    meta_params["policy"],
                    aug_images,
                    aug_labels,
                    embeds,
                    rng_inner,
                )
                return loss

            task_grads = jax.grad(inner_task_loss)(task_state.params)
            pseudo_params = optax.apply_updates(
                task_state.params,
                jax.tree_util.tree_map(lambda grad: -inner_lr * grad, task_grads),
            )
            val_logits = model.apply(
                {"params": pseudo_params, "batch_stats": task_state.batch_stats},
                normalize_images(val_batch["image"], dataset),
                train=False,
                return_features=False,
            )
            losses = cross_entropy_loss(
                val_logits,
                val_batch["label"],
                num_classes=num_classes,
                label_smoothing=config.optim.label_smoothing,
            )
            return jnp.mean(losses)

        policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_state.params)
        policy_grads = jax.lax.pmean(policy_grads, axis_name)
        new_policy_state = policy_state.apply_gradients(grads=policy_grads)

        def task_loss_fn(params):
            return weighted_aug_loss(
                params,
                task_state.batch_stats,
                new_policy_state.params["policy"],
                aug_images,
                aug_labels,
                embeds,
                rng_task,
            )

        (task_loss, (logits, raw_weights, new_batch_stats)), task_grads = jax.value_and_grad(
            task_loss_fn,
            has_aux=True,
        )(task_state.params)
        task_grads = jax.lax.pmean(task_grads, axis_name)
        new_batch_stats = jax.lax.pmean(new_batch_stats, axis_name)
        new_task_state = task_state.apply_gradients(
            grads=task_grads,
            batch_stats=new_batch_stats,
        )

        flat_ids = pair_ids
        pair_sums = jnp.zeros((NUM_OPS * NUM_OPS,), dtype=jnp.float32).at[flat_ids].add(raw_weights)
        pair_counts = jnp.zeros((NUM_OPS * NUM_OPS,), dtype=jnp.float32).at[flat_ids].add(1.0)
        pair_sums = jax.lax.psum(pair_sums.reshape((NUM_OPS, NUM_OPS)), axis_name)
        pair_counts = jax.lax.psum(pair_counts.reshape((NUM_OPS, NUM_OPS)), axis_name)

        metrics = {
            "task_loss": jax.lax.pmean(task_loss, axis_name),
            "policy_loss": jax.lax.pmean(policy_loss, axis_name),
            "train_top1": jax.lax.pmean(topk_accuracy(logits, aug_labels, k=1), axis_name),
            "train_top5": jax.lax.pmean(topk_accuracy(logits, aug_labels, k=5), axis_name),
            "mean_policy_weight": jax.lax.pmean(jnp.mean(raw_weights), axis_name),
            "inner_lr": jax.lax.pmean(jax.nn.softplus(new_policy_state.params["inner_log_lr"]), axis_name),
        }
        return new_task_state, new_policy_state, metrics, pair_sums, pair_counts

    return jax.pmap(step, axis_name=axis_name)


def _create_eval_step(model: WideResNet, config: Config, *, axis_name: str):
    num_classes = 10 if config.data.dataset.lower() in {"cifar10", "synthetic"} else 100
    dataset = config.data.dataset.lower()

    def step(task_state, batch):
        logits = model.apply(
            {"params": task_state.params, "batch_stats": task_state.batch_stats},
            normalize_images(batch["image"], dataset),
            train=False,
            return_features=False,
        )
        losses = cross_entropy_loss(logits, batch["label"], num_classes=num_classes)
        mask = batch["mask"]
        metrics = {
            "loss_sum": jax.lax.psum(jnp.sum(losses * mask), axis_name),
            "top1_sum": jax.lax.psum(topk_accuracy_sum(logits, batch["label"], mask, k=1), axis_name),
            "top5_sum": jax.lax.psum(topk_accuracy_sum(logits, batch["label"], mask, k=5), axis_name),
            "count": jax.lax.psum(jnp.sum(mask), axis_name),
        }
        return metrics

    return jax.pmap(step, axis_name=axis_name)


def _save_checkpoint(
    workdir: Path,
    epoch: int,
    task_state: TaskTrainState,
    policy_state: PolicyTrainState,
    sampler_probs: jnp.ndarray,
    config: Config,
) -> None:
    if jax.process_index() != 0:
        return
    workdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "task": {
            "step": int(jax.device_get(task_state.step)),
            "params": jax.device_get(task_state.params),
            "batch_stats": jax.device_get(task_state.batch_stats),
            "opt_state": jax.device_get(task_state.opt_state),
        },
        "policy": {
            "step": int(jax.device_get(policy_state.step)),
            "params": jax.device_get(policy_state.params),
            "opt_state": jax.device_get(policy_state.opt_state),
        },
        "sampler_probs": jax.device_get(sampler_probs),
        "config": to_dict(config),
    }
    with (workdir / f"checkpoint_{epoch:04d}.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    with (workdir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(to_dict(config), handle, indent=2)


def _evaluate(
    eval_step,
    task_state,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    device_count: int,
) -> dict[str, float]:
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_count = 0.0
    for batch in iter_batches(
        images,
        labels,
        batch_size,
        rng,
        shuffle=False,
        drop_remainder=False,
    ):
        metrics = eval_step(task_state, shard_batch(batch, device_count))
        metrics = jax.device_get(jax_utils.unreplicate(metrics))
        total_loss += float(metrics["loss_sum"])
        total_top1 += float(metrics["top1_sum"])
        total_top5 += float(metrics["top5_sum"])
        total_count += float(metrics["count"])
    return {
        "loss": total_loss / max(1.0, total_count),
        "top1": total_top1 / max(1.0, total_count),
        "top5": total_top5 / max(1.0, total_count),
    }


def _update_sampler_probs(history: list[np.ndarray], epsilon: float) -> np.ndarray:
    sums = np.zeros((NUM_OPS, NUM_OPS), dtype=np.float64)
    counts = np.zeros((NUM_OPS, NUM_OPS), dtype=np.float64)
    for item in history:
        sums += item[..., 0]
        counts += item[..., 1]
    values = sums / np.maximum(counts, 1.0)
    values = np.where(counts > 0.0, values, 1.0)
    values = values / np.maximum(values.sum(), 1.0e-12)
    uniform = np.full_like(values, 1.0 / values.size)
    return ((1.0 - epsilon) * values + epsilon * uniform).astype(np.float32)


def train_and_evaluate(config: Config) -> None:
    if config.system.init_distributed:
        jax.distributed.initialize()

    dataset = load_cifar_dataset(
        config.data.dataset,
        config.data.data_dir,
        val_size=config.data.val_size,
        seed=config.data.seed,
    )
    device_count = jax.local_device_count() if config.system.use_pmap else 1
    if config.data.batch_size % device_count != 0:
        raise ValueError("data.batch_size must be divisible by the local device count")
    if config.data.eval_batch_size % device_count != 0:
        raise ValueError("data.eval_batch_size must be divisible by the local device count")

    steps_per_epoch = dataset.train_size // config.data.batch_size
    if steps_per_epoch <= 0:
        raise ValueError("Training set is smaller than one batch")

    axis_name = "device"
    model, policy, task_state, policy_state = _create_states(
        config,
        dataset.num_classes,
        steps_per_epoch,
        axis_name=axis_name,
    )
    train_step = _create_train_step(model, policy, config, axis_name=axis_name)
    eval_step = _create_eval_step(model, config, axis_name=axis_name)

    task_state = jax_utils.replicate(task_state)
    policy_state = jax_utils.replicate(policy_state)
    sampler_probs = jax_utils.replicate(initial_sampler_probs())

    workdir = Path(config.system.workdir)
    rng = np.random.default_rng(config.system.seed)
    val_batcher = CyclingBatcher(
        dataset.val_images,
        dataset.val_labels,
        config.data.batch_size,
        rng,
        shuffle=True,
    )
    step_rng = jax.random.PRNGKey(config.system.seed)
    sampler_history: list[np.ndarray] = []
    _host0_print(
        f"Starting MetaAugment on {config.data.dataset}: "
        f"{dataset.train_size} train, {dataset.val_size} val, {dataset.test_size} test, "
        f"{device_count} local device(s)."
    )

    for epoch in range(1, config.optim.epochs + 1):
        start_time = time.time()
        epoch_pair_sums = np.zeros((NUM_OPS, NUM_OPS), dtype=np.float64)
        epoch_pair_counts = np.zeros((NUM_OPS, NUM_OPS), dtype=np.float64)
        train_metrics = []
        train_iter = iter_batches(
            dataset.train_images,
            dataset.train_labels,
            config.data.batch_size,
            rng,
            shuffle=True,
            drop_remainder=True,
        )
        for step, train_batch in enumerate(train_iter, start=1):
            val_batch = val_batcher.next()
            step_rng, *device_rngs = jax.random.split(step_rng, device_count + 1)
            device_rngs = jnp.asarray(device_rngs)
            task_state, policy_state, metrics, pair_sums, pair_counts = train_step(
                task_state,
                policy_state,
                sampler_probs,
                shard_batch(train_batch, device_count),
                shard_batch(val_batch, device_count),
                device_rngs,
            )
            host_metrics = jax.device_get(jax_utils.unreplicate(metrics))
            train_metrics.append(host_metrics)
            host_sums = jax.device_get(jax_utils.unreplicate(pair_sums))
            host_counts = jax.device_get(jax_utils.unreplicate(pair_counts))
            epoch_pair_sums += host_sums
            epoch_pair_counts += host_counts

            if step % config.system.log_every == 0:
                _host0_print(
                    f"epoch {epoch:03d} step {step:04d}/{steps_per_epoch} "
                    f"task_loss={float(host_metrics['task_loss']):.4f} "
                    f"policy_loss={float(host_metrics['policy_loss']):.4f} "
                    f"top1={float(host_metrics['train_top1']):.4f} "
                    f"top5={float(host_metrics['train_top5']):.4f} "
                    f"inner_lr={float(host_metrics['inner_lr']):.5f}"
                )

        sampler_history.append(np.stack([epoch_pair_sums, epoch_pair_counts], axis=-1))
        max_history = config.augment.sampler_history_epochs
        if len(sampler_history) > max_history:
            sampler_history = sampler_history[-max_history:]
        if epoch % config.augment.sampler_update_epochs == 0:
            updated = _update_sampler_probs(sampler_history, config.augment.epsilon)
            sampler_probs = jax_utils.replicate(jnp.asarray(updated))

        mean_metrics = {
            key: float(np.mean([np.asarray(metrics[key]) for metrics in train_metrics]))
            for key in train_metrics[0]
        }
        message = (
            f"epoch {epoch:03d} done in {time.time() - start_time:.1f}s "
            f"task_loss={mean_metrics['task_loss']:.4f} "
            f"policy_loss={mean_metrics['policy_loss']:.4f} "
            f"top1={mean_metrics['train_top1']:.4f} "
            f"top5={mean_metrics['train_top5']:.4f}"
        )

        if epoch % config.system.eval_every_epochs == 0:
            val_metrics = _evaluate(
                eval_step,
                task_state,
                dataset.val_images,
                dataset.val_labels,
                config.data.eval_batch_size,
                rng,
                device_count,
            )
            test_metrics = _evaluate(
                eval_step,
                task_state,
                dataset.test_images,
                dataset.test_labels,
                config.data.eval_batch_size,
                rng,
                device_count,
            )
            message += (
                f" val_top1={val_metrics['top1']:.4f} "
                f"val_top5={val_metrics['top5']:.4f} "
                f"test_top1={test_metrics['top1']:.4f} "
                f"test_top5={test_metrics['top5']:.4f}"
            )
        _host0_print(message)

        if epoch % config.system.save_every_epochs == 0 or epoch == config.optim.epochs:
            _save_checkpoint(
                workdir,
                epoch,
                jax_utils.unreplicate(task_state),
                jax_utils.unreplicate(policy_state),
                jax_utils.unreplicate(sampler_probs),
                config,
            )

    final_probs = np.asarray(jax.device_get(jax_utils.unreplicate(sampler_probs)))
    if jax.process_index() == 0:
        workdir.mkdir(parents=True, exist_ok=True)
        np.save(workdir / "sampler_probs.npy", final_probs)
