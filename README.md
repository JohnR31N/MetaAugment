# MetaAugment

Unofficial JAX/Flax implementation of **MetaAugment: Sample-Aware Data
Augmentation Policy Learning** (arXiv:2012.12076).

This repo targets CIFAR-10/100 reproduction with WRN-28-10 and is structured
for Google Cloud TPU VMs via JAX/XLA.

## What is implemented

- WRN-28-10 task network with shared features for the policy network.
- The 14 AutoAugment/RandAugment-style image operations used by MetaAugment.
- 28-dimensional transformation embeddings.
- Policy MLP with two 100-unit branches and sigmoid weights.
- MetaAugment bilevel step:
  - sample a transformation pair from the learned sampler distribution,
  - compute an inner pseudo-update of the task network,
  - update the policy network on a validation batch,
  - update the task network with the updated policy weights.
- Sampler distribution update with epsilon exploration.
- CIFAR-10/100 download, deterministic 1,000-example validation split,
  checkpointing, and TPU-friendly `pmap` training.

The paper trains CIFAR-10 WRN-28-10 for 600 epochs and CIFAR-100 WRN-28-10 for
200 epochs with SGD momentum 0.9, weight decay 5e-4, batch size 128, cosine
task learning rate decay from 0.1, fixed policy learning rate 1e-3, and epsilon
0.1. Those defaults are in `configs/`.

## Local setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins `orbax-checkpoint==0.10.3` because newer Orbax wheels
currently include very deep test fixture paths that can exceed Windows path
limits in a nested `.venv`. This implementation saves checkpoints directly with
pickle and does not rely on Orbax checkpoint APIs.

Run a quick synthetic smoke test:

```bash
python -m meta_augment.cli.train --config configs/smoke.yaml
```

Train CIFAR-10:

```bash
python -m meta_augment.cli.train --config configs/cifar10_wrn28_10_metaaugment.yaml
```

Train CIFAR-100:

```bash
python -m meta_augment.cli.train --config configs/cifar100_wrn28_10_metaaugment.yaml
```

## Google Cloud TPU VM

Create a TPU VM with a recent JAX-compatible image, SSH in, clone this repo,
then install TPU JAX instead of CPU JAX:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-tpu.txt
```

Or run `bash scripts/setup_tpu_vm.sh`.

Run:

```bash
python -m meta_augment.cli.train \
  --config configs/cifar10_wrn28_10_metaaugment.yaml \
  --data-dir /tmp/metaaugment_data \
  --workdir /tmp/metaaugment_runs/cifar10
```

For multi-host TPU slices, run the same command on every worker and add
`--init-distributed` if your TPU environment does not initialize JAX
distributed automatically.

## Notes

- Checkpoints are written as Python pickles in `workdir`.
- `sampler_probs.npy` stores the final learned 14 x 14 transformation
  distribution.
- The CPU requirement uses `jax[cpu]`; on TPU VMs install `jax[tpu]` with the
  Google libtpu wheel link shown above.
