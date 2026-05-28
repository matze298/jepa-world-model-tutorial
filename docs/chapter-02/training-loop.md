# 2.9 Training Loop

We now have the components required for minimal I-JEPA training:

- image patchification,
- positional embeddings,
- context and target mask sampling,
- minimal ViT encoder,
- EMA target encoder,
- JEPA predictor,
- latent losses,
- representation diagnostics.

This section assembles them into a complete training loop.

The goal is not yet large-scale training. The goal is a transparent, correct loop that we can run locally, debug in marimo, and later scale in Chapter 3.

The core training step is:

```python
context_indices, target_indices = sample_mask_batch(...)

pred_repr, target_repr = model(
    images=images,
    context_indices=context_indices,
    target_indices=target_indices,
)

loss = latent_loss(pred_repr, target_repr)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()

update_ema(...)
```

This is the first point where the entire minimal JEPA system comes together.

---

## 2.9.1 What the Training Loop Must Do

Each training step must:

1. load a batch of images,
2. sample context and target masks,
3. run the online encoder on context patches,
4. run the target encoder on target patches without gradients,
5. run the predictor,
6. compute the latent prediction loss,
7. update online encoder and predictor parameters,
8. update the target encoder by EMA,
9. compute diagnostics,
10. log results.

In pseudocode:

```python
for images in train_loader:
    images = images.to(device)

    context_indices, target_indices = sample_mask_batch(
        config=mask_config,
        batch_size=images.size(0),
        device=images.device,
    )

    pred_repr, target_repr = model(
        images=images,
        context_indices=context_indices,
        target_indices=target_indices,
    )

    loss = latent_loss(pred_repr, target_repr)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    tau = ema_schedule(...)
    update_ema(model.online_encoder, model.target_encoder, tau)

    logs = diagnostics(...)
```

This is intentionally plain PyTorch.

We want every JEPA-specific operation to remain visible.

---

## 2.9.2 Implementing the Top-Level Model

Create:

```text
src/jepa_world_model/model.py
```

Add:

```python
from __future__ import annotations

import torch
import torch.nn as nn

from jepa_world_model.ema import initialize_target_encoder
```

Define the top-level model:

```python
class MinimalIJEPA(nn.Module):
    """
    Minimal image JEPA model.

    Components:
        online_encoder:
            Processes context patches and receives gradients.

        target_encoder:
            Processes target patches and is updated by EMA.

        predictor:
            Predicts target representations from context representations
            and target positions.
    """

    def __init__(
        self,
        online_encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: nn.Module,
    ):
        super().__init__()

        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor

        initialize_target_encoder(
            online=self.online_encoder,
            target=self.target_encoder,
        )

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images:
                [B, C, H, W]

            context_indices:
                [B, N_ctx]

            target_indices:
                [B, N_tgt]

        Returns:
            pred_repr:
                [B, N_tgt, D]

            target_repr:
                [B, N_tgt, D]
        """
        context_repr = self.online_encoder(
            images=images,
            patch_indices=context_indices,
        )

        with torch.no_grad():
            target_repr = self.target_encoder(
                images=images,
                patch_indices=target_indices,
            )

        pred_repr = self.predictor(
            context_repr=context_repr,
            context_indices=context_indices,
            target_indices=target_indices,
        )

        return pred_repr, target_repr
```

The model does **not** compute the loss internally.

The model does **not** update EMA internally.

Those belong in the training loop.

This keeps the forward pass pure and easier to test.

---

## 2.9.3 Building the Model from Config

Add a model builder to `model.py`.

```python
from jepa_world_model.config import MinimalJEPAConfig
from jepa_world_model.predictor import JEPAPredictor
from jepa_world_model.vit import MinimalViTEncoder
```

Then:

```python
def build_minimal_ijepa(
    cfg: MinimalJEPAConfig,
) -> MinimalIJEPA:
    """
    Build the minimal I-JEPA model from config.
    """
    online_encoder = MinimalViTEncoder(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
        embed_dim=cfg.encoder_dim,
        depth=cfg.encoder_depth,
        num_heads=cfg.encoder_heads,
        mlp_ratio=cfg.mlp_ratio,
    )

    target_encoder = MinimalViTEncoder(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
        embed_dim=cfg.encoder_dim,
        depth=cfg.encoder_depth,
        num_heads=cfg.encoder_heads,
        mlp_ratio=cfg.mlp_ratio,
    )

    grid_size = cfg.image_size // cfg.patch_size

    predictor = JEPAPredictor(
        grid_size=grid_size,
        encoder_dim=cfg.encoder_dim,
        predictor_dim=cfg.predictor_dim,
        depth=cfg.predictor_depth,
        num_heads=cfg.predictor_heads,
        mlp_ratio=cfg.mlp_ratio,
    )

    return MinimalIJEPA(
        online_encoder=online_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
    )
```

This gives experiments one simple entry point:

```python
model = build_minimal_ijepa(cfg)
```

---

## 2.9.4 Optimizer Setup

Only the online encoder and predictor should be optimized.

The target encoder is updated by EMA, not gradient descent.

Add:

```python
def trainable_jepa_parameters(
    model: MinimalIJEPA,
):
    """
    Return parameters that should be optimized by gradient descent.
    """
    yield from model.online_encoder.parameters()
    yield from model.predictor.parameters()
```

Then:

```python
optimizer = torch.optim.AdamW(
    trainable_jepa_parameters(model),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
)
```

Avoid:

```python
optimizer = torch.optim.AdamW(model.parameters(), ...)
```

because it includes target encoder parameters. They are frozen, but being explicit prevents confusion.

---

## 2.9.5 Mask Config from Model Config

The mask sampler uses `BlockMaskConfig`.

Add a helper:

```python
from jepa_world_model.masks import BlockMaskConfig
```

```python
def build_mask_config(
    cfg: MinimalJEPAConfig,
) -> BlockMaskConfig:
    grid_size = cfg.image_size // cfg.patch_size

    return BlockMaskConfig(
        grid_height=grid_size,
        grid_width=grid_size,
        num_target_blocks=cfg.num_target_blocks,
        target_block_height=cfg.target_block_height,
        target_block_width=cfg.target_block_width,
        context_ratio=cfg.context_ratio,
    )
```

Usage:

```python
mask_config = build_mask_config(cfg)
```

---

## 2.9.6 The Training Step

Create:

```text
src/jepa_world_model/training.py
```

Add imports:

```python
from __future__ import annotations

import torch

from jepa_world_model.diagnostics import (
    assert_finite_tensor,
    jepa_diagnostics,
    mask_stats,
)
from jepa_world_model.ema import cosine_ema_tau, parameter_distance, update_ema
from jepa_world_model.losses import latent_loss
from jepa_world_model.masks import (
    BlockMaskConfig,
    assert_no_mask_overlap,
    sample_mask_batch,
)
from jepa_world_model.model import MinimalIJEPA
```

Now implement the training step:

```python
def train_step(
    model: MinimalIJEPA,
    images: torch.Tensor,
    mask_config: BlockMaskConfig,
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    loss_type: str,
    ema_tau_base: float,
    ema_tau_final: float,
    check_masks: bool = True,
    log_parameter_distance: bool = False,
) -> dict[str, float]:
    """
    Run one JEPA training step.

    Args:
        model:
            MinimalIJEPA model.

        images:
            Image batch [B, C, H, W].

        mask_config:
            Context/target mask configuration.

        optimizer:
            Optimizer for online encoder and predictor.

        step:
            Global training step.

        total_steps:
            Total number of training steps.

        loss_type:
            Latent loss type.

        ema_tau_base:
            Initial EMA momentum.

        ema_tau_final:
            Final EMA momentum.

        check_masks:
            Whether to assert no context-target overlap.

        log_parameter_distance:
            Whether to compute online-target parameter distance.

    Returns:
        logs:
            Dictionary of scalar logs.
    """
    model.train()

    batch_size = images.size(0)

    context_indices, target_indices = sample_mask_batch(
        config=mask_config,
        batch_size=batch_size,
        device=images.device,
    )

    if check_masks:
        assert_no_mask_overlap(
            context_indices=context_indices,
            target_indices=target_indices,
        )

    pred_repr, target_repr = model(
        images=images,
        context_indices=context_indices,
        target_indices=target_indices,
    )

    assert_finite_tensor(pred_repr, "pred_repr")
    assert_finite_tensor(target_repr, "target_repr")

    loss = latent_loss(
        pred=pred_repr,
        target=target_repr,
        loss_type=loss_type,
    )

    assert_finite_tensor(loss, "loss")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    ema_tau = cosine_ema_tau(
        step=step,
        total_steps=total_steps,
        tau_base=ema_tau_base,
        tau_final=ema_tau_final,
    )

    update_ema(
        online=model.online_encoder,
        target=model.target_encoder,
        tau=ema_tau,
    )

    with torch.no_grad():
        logs = {
            "loss": loss.item(),
            "ema_tau": ema_tau,
        }

        logs.update(
            jepa_diagnostics(
                pred=pred_repr,
                target=target_repr,
            )
        )

        logs.update(
            mask_stats(
                context_indices=context_indices,
                target_indices=target_indices,
                num_patches=mask_config.num_patches,
            )
        )

        if log_parameter_distance:
            logs.update(
                parameter_distance(
                    online=model.online_encoder,
                    target=model.target_encoder,
                )
            )

    return logs
```

This function is the central training primitive.

It intentionally contains JEPA-specific steps:

- mask sampling,
- target no-grad forward through model,
- latent loss,
- EMA update,
- diagnostics.

---

## 2.9.7 Why Mask Sampling Happens Inside the Training Step

We sample new masks every step.

This means the same image can produce different prediction tasks across epochs.

For example, one epoch may hide patches in the lower-left region. Another may hide patches in the upper-right region.

This improves the diversity of self-supervised targets.

The mask sampler should use:

```python
device=images.device
```

so that indices live on the same device as image tensors and model activations.

---

## 2.9.8 Logging Formatter

Add a simple log formatter to `training.py` or `utils.py`.

```python
def format_logs(
    logs: dict[str, float],
    keys: list[str] | None = None,
) -> str:
    """
    Format scalar logs for console printing.
    """
    if keys is None:
        keys = [
            "loss",
            "pred_target/cosine",
            "target/std_mean",
            "pred/std_mean",
            "target/effective_rank",
            "mask/overlap_fraction",
            "ema_tau",
        ]

    parts = []

    for key in keys:
        if key not in logs:
            continue

        value = logs[key]

        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")

    return " | ".join(parts)
```

Usage:

```python
print(format_logs(logs))
```

Example output:

```text
loss=0.1832 | pred_target/cosine=0.2147 | target/std_mean=0.5861 | pred/std_mean=0.3189 | target/effective_rank=41.2237 | mask/overlap_fraction=0.0000 | ema_tau=0.9960
```

---

## 2.9.9 Checkpointing

For the minimal implementation, checkpointing can be simple.

Create:

```text
src/jepa_world_model/checkpointing.py
```

Add:

```python
from __future__ import annotations

from pathlib import Path

import torch

from jepa_world_model.config import MinimalJEPAConfig
from jepa_world_model.model import MinimalIJEPA
```

Save checkpoint:

```python
def save_checkpoint(
    path: str | Path,
    model: MinimalIJEPA,
    optimizer: torch.optim.Optimizer,
    cfg: MinimalJEPAConfig,
    step: int,
    epoch: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "step": step,
        "epoch": epoch,
    }

    torch.save(
        checkpoint,
        path,
    )
```

Load checkpoint:

```python
def load_checkpoint(
    path: str | Path,
    model: MinimalIJEPA,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
```

This saves:

- online encoder,
- target encoder,
- predictor,
- optimizer state,
- config,
- step,
- epoch.

The target encoder must be saved because it is part of the training state.

---

## 2.9.10 Data Loading

We will use torchvision datasets for the minimal image experiments.

Create:

```text
src/jepa_world_model/data.py
```

Add:

```python
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, STL10

from jepa_world_model.config import MinimalJEPAConfig
```

Dataset builder:

```python
def build_image_transform(
    image_size: int,
) -> T.Compose:
    """
    Build simple image transform for JEPA pretraining.
    """
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]
    )
```

STL-10 loader:

```python
def build_stl10_dataloader(
    cfg: MinimalJEPAConfig,
    split: str = "unlabeled",
    shuffle: bool = True,
) -> DataLoader:
    transform = build_image_transform(
        image_size=cfg.image_size,
    )

    dataset = STL10(
        root=cfg.data_dir,
        split=split,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
```

CIFAR-10 loader:

```python
def build_cifar10_dataloader(
    cfg: MinimalJEPAConfig,
    train: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    transform = build_image_transform(
        image_size=cfg.image_size,
    )

    dataset = CIFAR10(
        root=cfg.data_dir,
        train=train,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
```

Generic builder:

```python
def build_pretrain_dataloader(
    cfg: MinimalJEPAConfig,
    dataset_name: str = "stl10",
) -> DataLoader:
    if dataset_name == "stl10":
        return build_stl10_dataloader(
            cfg=cfg,
            split="unlabeled",
            shuffle=True,
        )

    if dataset_name == "cifar10":
        return build_cifar10_dataloader(
            cfg=cfg,
            train=True,
            shuffle=True,
        )

    raise ValueError(f"Unknown dataset_name: {dataset_name}")
```

The dataloader returns:

```python
images, labels
```

For JEPA pretraining, labels are ignored.

---

## 2.9.11 Minimal Training Script

Create:

```text
experiments/train_minimal.py
```

Add:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from jepa_world_model.checkpointing import save_checkpoint
from jepa_world_model.config import MinimalJEPAConfig
from jepa_world_model.data import build_pretrain_dataloader
from jepa_world_model.model import (
    build_mask_config,
    build_minimal_ijepa,
    trainable_jepa_parameters,
)
from jepa_world_model.presets import local_debug_config
from jepa_world_model.training import format_logs, train_step
from jepa_world_model.utils import get_device, seed_everything
```

Argument parsing:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preset",
        type=str,
        default="local_debug",
        choices=["local_debug", "default"],
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "stl10"],
    )

    parser.add_argument(
        "--loss-type",
        type=str,
        default="smooth_l1",
        choices=["mse", "smooth_l1", "cosine", "combined"],
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
    )

    return parser.parse_args()
```

Config selection:

```python
def build_config(preset: str) -> MinimalJEPAConfig:
    if preset == "local_debug":
        return local_debug_config()

    if preset == "default":
        return MinimalJEPAConfig()

    raise ValueError(f"Unknown preset: {preset}")
```

Main:

```python
def main() -> None:
    args = parse_args()

    cfg = build_config(args.preset)

    seed_everything(cfg.seed)

    device = get_device()

    print(f"Using device: {device}")
    print(f"Using config: {cfg}")

    dataloader = build_pretrain_dataloader(
        cfg=cfg,
        dataset_name=args.dataset,
    )

    model = build_minimal_ijepa(cfg).to(device)

    optimizer = torch.optim.AdamW(
        trainable_jepa_parameters(model),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    mask_config = build_mask_config(cfg)

    total_steps = cfg.epochs * len(dataloader)
    global_step = 0

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(cfg.epochs):
        progress = tqdm(
            dataloader,
            desc=f"epoch {epoch + 1}/{cfg.epochs}",
        )

        for batch in progress:
            images = batch[0].to(
                device,
                non_blocking=True,
            )

            logs = train_step(
                model=model,
                images=images,
                mask_config=mask_config,
                optimizer=optimizer,
                step=global_step,
                total_steps=total_steps,
                loss_type=args.loss_type,
                ema_tau_base=cfg.ema_tau_base,
                ema_tau_final=cfg.ema_tau_final,
                check_masks=True,
                log_parameter_distance=False,
            )

            if global_step % args.log_every == 0:
                progress.set_postfix_str(
                    format_logs(logs)
                )

            global_step += 1

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                path=checkpoint_dir / f"minimal_ijepa_epoch_{epoch + 1}.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                step=global_step,
                epoch=epoch + 1,
            )


if __name__ == "__main__":
    main()
```

Run:

```bash
python experiments/train_minimal.py --preset local_debug --dataset cifar10
```

This should run a tiny local debug experiment.

---

## 2.9.12 Presets

Create or update:

```text
src/jepa_world_model/presets.py
```

Add:

```python
from jepa_world_model.config import MinimalJEPAConfig


def local_debug_config() -> MinimalJEPAConfig:
    return MinimalJEPAConfig(
        image_size=32,
        patch_size=4,
        encoder_dim=64,
        encoder_depth=2,
        encoder_heads=4,
        predictor_dim=64,
        predictor_depth=1,
        predictor_heads=4,
        num_target_blocks=2,
        target_block_height=2,
        target_block_width=2,
        context_ratio=0.5,
        batch_size=16,
        epochs=2,
        num_workers=0,
        learning_rate=1e-3,
        weight_decay=0.05,
    )


def cloud_run_config() -> MinimalJEPAConfig:
    return MinimalJEPAConfig(
        image_size=96,
        patch_size=8,
        encoder_dim=384,
        encoder_depth=8,
        encoder_heads=6,
        predictor_dim=256,
        predictor_depth=4,
        predictor_heads=8,
        num_target_blocks=4,
        target_block_height=3,
        target_block_width=3,
        context_ratio=0.6,
        batch_size=256,
        epochs=300,
        num_workers=8,
        learning_rate=5e-4,
        weight_decay=0.05,
        data_dir="/workspace/data",
        run_dir="/workspace/runs",
        checkpoint_dir="/workspace/checkpoints",
    )
```

This keeps local and cloud configs separate without introducing Hydra yet.

---

## 2.9.13 Cloud Training Script

Create:

```text
experiments/train_cloud.py
```

For now, this can be a thin wrapper around the same logic as `train_minimal.py`.

A simple version:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from jepa_world_model.checkpointing import save_checkpoint
from jepa_world_model.data import build_pretrain_dataloader
from jepa_world_model.model import (
    build_mask_config,
    build_minimal_ijepa,
    trainable_jepa_parameters,
)
from jepa_world_model.presets import cloud_run_config
from jepa_world_model.training import format_logs, train_step
from jepa_world_model.utils import get_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="stl10",
        choices=["cifar10", "stl10"],
    )

    parser.add_argument(
        "--loss-type",
        type=str,
        default="smooth_l1",
        choices=["mse", "smooth_l1", "cosine", "combined"],
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = cloud_run_config()

    seed_everything(cfg.seed)

    device = get_device()

    if device.type != "cuda":
        raise RuntimeError(
            f"Cloud training expected CUDA, got device={device}."
        )

    print(f"Using device: {device}")
    print(f"Using config: {cfg}")

    dataloader = build_pretrain_dataloader(
        cfg=cfg,
        dataset_name=args.dataset,
    )

    model = build_minimal_ijepa(cfg).to(device)

    optimizer = torch.optim.AdamW(
        trainable_jepa_parameters(model),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    mask_config = build_mask_config(cfg)

    total_steps = cfg.epochs * len(dataloader)
    global_step = 0

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(cfg.epochs):
        progress = tqdm(
            dataloader,
            desc=f"epoch {epoch + 1}/{cfg.epochs}",
        )

        for batch in progress:
            images = batch[0].to(
                device,
                non_blocking=True,
            )

            logs = train_step(
                model=model,
                images=images,
                mask_config=mask_config,
                optimizer=optimizer,
                step=global_step,
                total_steps=total_steps,
                loss_type=args.loss_type,
                ema_tau_base=cfg.ema_tau_base,
                ema_tau_final=cfg.ema_tau_final,
                check_masks=False,
                log_parameter_distance=False,
            )

            if global_step % args.log_every == 0:
                progress.set_postfix_str(
                    format_logs(logs)
                )

            global_step += 1

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                path=checkpoint_dir / f"minimal_ijepa_epoch_{epoch + 1}.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                step=global_step,
                epoch=epoch + 1,
            )


if __name__ == "__main__":
    main()
```

Run on cloud:

```bash
python experiments/train_cloud.py --dataset stl10
```

This is still plain PyTorch. Chapter 3 will introduce better experiment infrastructure.

---

## 2.9.14 Tiny Overfit Test

Before running real training, overfit a tiny batch.

The goal is not representation quality. The goal is to verify that:

- forward pass works,
- backward pass works,
- optimizer updates parameters,
- EMA updates target encoder,
- loss can decrease.

A tiny overfit script can use one batch repeatedly.

Create:

```text
experiments/overfit_tiny_batch.py
```

Sketch:

```python
from __future__ import annotations

import torch

from jepa_world_model.data import build_pretrain_dataloader
from jepa_world_model.model import (
    build_mask_config,
    build_minimal_ijepa,
    trainable_jepa_parameters,
)
from jepa_world_model.presets import local_debug_config
from jepa_world_model.training import format_logs, train_step
from jepa_world_model.utils import get_device, seed_everything


def main() -> None:
    cfg = local_debug_config()

    seed_everything(cfg.seed)

    device = get_device()

    dataloader = build_pretrain_dataloader(
        cfg=cfg,
        dataset_name="cifar10",
    )

    batch = next(iter(dataloader))
    images = batch[0].to(device)

    model = build_minimal_ijepa(cfg).to(device)

    optimizer = torch.optim.AdamW(
        trainable_jepa_parameters(model),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    mask_config = build_mask_config(cfg)

    total_steps = 200

    for step in range(total_steps):
        logs = train_step(
            model=model,
            images=images,
            mask_config=mask_config,
            optimizer=optimizer,
            step=step,
            total_steps=total_steps,
            loss_type="smooth_l1",
            ema_tau_base=cfg.ema_tau_base,
            ema_tau_final=cfg.ema_tau_final,
            check_masks=True,
            log_parameter_distance=step % 20 == 0,
        )

        if step % 10 == 0:
            print(
                f"step={step} | {format_logs(logs)}"
            )


if __name__ == "__main__":
    main()
```

Run:

```bash
python experiments/overfit_tiny_batch.py
```

Expected behavior:

- loss should generally decrease,
- representation stats should remain nonzero,
- mask overlap should stay zero,
- no NaNs should appear.

This is one of the most important sanity checks.

---

## 2.9.15 Unit Tests for Model Assembly

Create:

```text
tests/test_model.py
```

Add:

```python
import torch

from jepa_world_model.model import (
    build_mask_config,
    build_minimal_ijepa,
    trainable_jepa_parameters,
)
from jepa_world_model.presets import local_debug_config


def test_build_minimal_ijepa_forward():
    cfg = local_debug_config()

    model = build_minimal_ijepa(cfg)
    mask_config = build_mask_config(cfg)

    images = torch.randn(
        2,
        cfg.in_channels,
        cfg.image_size,
        cfg.image_size,
    )

    from jepa_world_model.masks import sample_mask_batch

    context_indices, target_indices = sample_mask_batch(
        config=mask_config,
        batch_size=images.size(0),
        device=torch.device("cpu"),
    )

    pred_repr, target_repr = model(
        images=images,
        context_indices=context_indices,
        target_indices=target_indices,
    )

    assert pred_repr.shape == target_repr.shape
    assert pred_repr.shape[0] == 2
    assert pred_repr.shape[-1] == cfg.encoder_dim


def test_target_encoder_frozen():
    cfg = local_debug_config()
    model = build_minimal_ijepa(cfg)

    assert all(
        not p.requires_grad
        for p in model.target_encoder.parameters()
    )


def test_trainable_parameters_excludes_target_encoder():
    cfg = local_debug_config()
    model = build_minimal_ijepa(cfg)

    trainable_ids = {
        id(p)
        for p in trainable_jepa_parameters(model)
    }

    target_ids = {
        id(p)
        for p in model.target_encoder.parameters()
    }

    assert trainable_ids.isdisjoint(target_ids)
```

Run:

```bash
pytest tests/test_model.py
```

---

## 2.9.16 Unit Test for One Training Step

Create:

```text
tests/test_training.py
```

Add:

```python
import torch

from jepa_world_model.model import (
    build_mask_config,
    build_minimal_ijepa,
    trainable_jepa_parameters,
)
from jepa_world_model.presets import local_debug_config
from jepa_world_model.training import train_step


def test_train_step_runs():
    cfg = local_debug_config()

    model = build_minimal_ijepa(cfg)

    optimizer = torch.optim.AdamW(
        trainable_jepa_parameters(model),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    mask_config = build_mask_config(cfg)

    images = torch.randn(
        2,
        cfg.in_channels,
        cfg.image_size,
        cfg.image_size,
    )

    logs = train_step(
        model=model,
        images=images,
        mask_config=mask_config,
        optimizer=optimizer,
        step=0,
        total_steps=10,
        loss_type="smooth_l1",
        ema_tau_base=cfg.ema_tau_base,
        ema_tau_final=cfg.ema_tau_final,
        check_masks=True,
        log_parameter_distance=True,
    )

    assert "loss" in logs
    assert "pred_target/cosine" in logs
    assert "mask/overlap_fraction" in logs
    assert logs["mask/overlap_fraction"] == 0.0
```

Run:

```bash
pytest tests/test_training.py
```

This test confirms the entire training step works.

---

## 2.9.17 Common Bugs

### Bug 1: Optimizer includes target encoder

Avoid:

```python
optimizer = torch.optim.AdamW(model.parameters(), ...)
```

Use:

```python
optimizer = torch.optim.AdamW(
    trainable_jepa_parameters(model),
    ...
)
```

---

### Bug 2: EMA update happens before optimizer step

Use:

```python
loss.backward()
optimizer.step()
update_ema(...)
```

---

### Bug 3: Mask tensors on CPU while images are on GPU

Sample masks with:

```python
device=images.device
```

---

### Bug 4: Loss decreases but target representation collapses

Always log:

```text
target/std_mean
target/dead_dim_fraction
target/effective_rank
```

---

### Bug 5: Checkpoint misses target encoder

Save:

```python
model.state_dict()
```

not just the online encoder. The target encoder is part of training state.

---

### Bug 6: `torch.load` and config objects

Saving a dataclass object inside a checkpoint is convenient for local use, but for long-term reproducibility a plain dictionary or YAML is safer.

For Chapter 2, saving the dataclass is fine.

Chapter 3 will improve checkpoint metadata.

---

## 2.9.18 Summary

This section assembled the minimal I-JEPA training loop.

We implemented:

- the top-level `MinimalIJEPA` model,
- model construction from config,
- trainable parameter selection,
- mask config construction,
- one-step training function,
- log formatting,
- simple checkpointing,
- dataset loaders,
- local training script,
- cloud training script,
- tiny overfit script,
- model and training tests.

We now have a trainable minimal JEPA system.

The next section evaluates the learned representation using nearest-neighbor retrieval and linear probing.

---

## References and Further Reading

- Mahmoud Assran et al., **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.
  <https://arxiv.org/abs/2301.08243>

- Facebook Research, **Official I-JEPA Codebase**.
  <https://github.com/facebookresearch/ijepa>

- PyTorch, **AdamW**.
  <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>

- PyTorch, **Saving and Loading Models**.
  <https://pytorch.org/tutorials/beginner/saving_loading_models.html>

- TorchVision, **Datasets**.
  <https://pytorch.org/vision/stable/datasets.html>
