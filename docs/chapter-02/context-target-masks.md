# 2.4 Context and Target Mask Sampling

JEPA is a predictive learning problem.

The model receives a **context** and predicts a **target representation**. For images, both context and target are defined as sets of patch indices.

The masking strategy determines the actual learning task.

If the target patches are too small or too local, the model may solve the task through texture continuation. If the target patches overlap with the context, the model can cheat. If the context is too small, the task may be impossible. If the context is too large, the task may become trivial.

So mask sampling is not just data preprocessing. It is part of the algorithm.

In this section, we implement context and target mask sampling for minimal I-JEPA.

---

## 2.4.1 What the Mask Sampler Must Produce

Given a patch grid of size:

\[
G \times G
\]

with:

\[
N = G^2
\]

patches, the sampler must produce two tensors:

```python
context_indices.shape
# [B, N_ctx]

target_indices.shape
# [B, N_tgt]
```

where:

- \(B\) is batch size,
- \(N_{\mathrm{ctx}}\) is number of context patches,
- \(N_{\mathrm{tgt}}\) is number of target patches.

The context and target sets must not overlap:

\[
\mathcal{C} \cap \mathcal{T} = \emptyset
\]

Here \(\mathcal{C}\) is the set of context patch indices and \(\mathcal{T}\) is the set of target patch indices for one image.

In code:

```python
context_indices, target_indices = sampler(
    batch_size=batch_size,
    device=device,
)
```

The encoder will use:

```python
context_repr = online_encoder(images, context_indices)
```

The target encoder will use:

```python
target_repr = target_encoder(images, target_indices)
```

The predictor will use:

```python
pred_repr = predictor(
    context_repr=context_repr,
    context_indices=context_indices,
    target_indices=target_indices,
)
```

This means the mask sampler controls what the model sees and what it must predict.

---

## 2.4.2 Masking in MAE vs I-JEPA

Masked Autoencoders commonly use random patch masking.

A typical MAE-style mask randomly selects a large fraction of patches to hide and asks the decoder to reconstruct the missing pixels.

JEPA uses a different kind of masking.

In I-JEPA, the model predicts representations of target blocks from a context block or context region. Target blocks are spatially coherent. This matters because a coherent block is more likely to correspond to meaningful object or scene structure than isolated random patches.

The contrast is:

```text
MAE-style random masking:
    many scattered patches hidden

JEPA-style block masking:
    coherent target regions hidden
```

For this minimal implementation, we will use:

- rectangular target blocks,
- a context set sampled from the remaining patches,
- explicit overlap checks,
- fixed-size output tensors.

---

## 2.4.3 Design Choices for the Minimal Sampler

We want the sampler to be simple, correct, and inspectable.

The minimal sampler will:

1. sample several rectangular target blocks,
2. merge those blocks into a target set,
3. remove target patches from the candidate context pool,
4. sample a fixed number of context patches from the remaining patches,
5. return fixed-size tensors.

The config parameters are:

```python
grid_size: int
num_target_blocks: int
target_block_height: int
target_block_width: int
context_ratio: float
```

Example:

```text
image_size = 96
patch_size = 8
grid_size = 12
num_patches = 144

num_target_blocks = 4
target_block_height = 3
target_block_width = 3
context_ratio = 0.6
```

Each target block contains:

\[
3 \times 3 = 9
\]

patches.

Four blocks contain up to:

\[
4 \times 9 = 36
\]

target patches.

If blocks overlap, the unique target set may be smaller. For the minimal implementation, we will resample until we get the desired target count.

The context count is:

\[
N_{\mathrm{ctx}}
=
\lfloor \rho N \rfloor
\]

where \(\rho\) is `context_ratio`.

For `context_ratio = 0.6` and \(N=144\):

\[
N_{\mathrm{ctx}} = 86
\]

---

## 2.4.4 Fixed-Size vs Variable-Size Masks

A target made from multiple blocks can have variable size if blocks overlap.

For example, two \(3 \times 3\) blocks may overlap by several patches. Then the unique target set is smaller than expected.

Variable-size masks complicate batching.

For the minimal implementation, we will use fixed-size masks.

The sampler will aim for:

```python
num_target_patches = num_target_blocks * target_block_height * target_block_width
```

If sampled target blocks overlap, we will resample until we obtain exactly that many unique target patches.

This is simple and avoids padding.

Later, a more general implementation can allow variable target lengths with padding masks.

---

## 2.4.5 Implementing `masks.py`

Create:

```text
src/jepa_world_model/masks.py
```

Start with basic imports:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
```

Define the mask config:

```python
@dataclass(frozen=True)
class BlockMaskConfig:
    grid_height: int
    grid_width: int
    num_target_blocks: int
    target_block_height: int
    target_block_width: int
    context_ratio: float = 0.6
    max_attempts: int = 100

    @property
    def num_patches(self) -> int:
        return self.grid_height * self.grid_width

    @property
    def num_target_patches(self) -> int:
        return (
            self.num_target_blocks
            * self.target_block_height
            * self.target_block_width
        )

    @property
    def num_context_patches(self) -> int:
        return int(self.context_ratio * self.num_patches)

    def validate(self) -> None:
        if self.grid_height <= 0 or self.grid_width <= 0:
            raise ValueError("Grid dimensions must be positive.")

        if self.num_target_blocks <= 0:
            raise ValueError("num_target_blocks must be positive.")

        if self.target_block_height <= 0 or self.target_block_width <= 0:
            raise ValueError("Target block dimensions must be positive.")

        if self.target_block_height > self.grid_height:
            raise ValueError("target_block_height cannot exceed grid_height.")

        if self.target_block_width > self.grid_width:
            raise ValueError("target_block_width cannot exceed grid_width.")

        if not 0.0 < self.context_ratio < 1.0:
            raise ValueError("context_ratio must be in (0, 1).")

        if self.num_target_patches >= self.num_patches:
            raise ValueError("Target patches must be fewer than total patches.")

        if self.num_context_patches + self.num_target_patches > self.num_patches:
            raise ValueError(
                "context patches + target patches exceeds total patches. "
                "Reduce context_ratio or target block count/size."
            )
```

This config validates that the mask request is feasible.

---

## 2.4.6 Sampling One Rectangular Block

A rectangular block is defined by:

- top row,
- left column,
- block height,
- block width.

The flattened index is:

\[
i = rG_w + c
\]

where \(G_w\) is the grid width.

Add:

```python
def sample_rectangular_block(
    grid_height: int,
    grid_width: int,
    block_height: int,
    block_width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample a rectangular block of patch indices.

    Returns:
        indices:
            Tensor of shape [block_height * block_width].
    """
    if block_height > grid_height or block_width > grid_width:
        raise ValueError(
            "Block dimensions cannot exceed grid dimensions. "
            f"Got block=({block_height}, {block_width}), "
            f"grid=({grid_height}, {grid_width})."
        )

    max_row = grid_height - block_height
    max_col = grid_width - block_width

    row = torch.randint(
        low=0,
        high=max_row + 1,
        size=(),
        device=device,
    )

    col = torch.randint(
        low=0,
        high=max_col + 1,
        size=(),
        device=device,
    )

    rows = torch.arange(
        row,
        row + block_height,
        device=device,
    )

    cols = torch.arange(
        col,
        col + block_width,
        device=device,
    )

    rr, cc = torch.meshgrid(
        rows,
        cols,
        indexing="ij",
    )

    indices = rr * grid_width + cc

    return indices.flatten().long()
```

This function samples a single target block.

---

## 2.4.7 Sampling Target Indices

Now sample multiple target blocks and merge them.

```python
def sample_target_indices(
    config: BlockMaskConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample fixed-size target indices from rectangular blocks.

    Returns:
        target_indices:
            Tensor of shape [num_target_patches].
    """
    config.validate()

    for _ in range(config.max_attempts):
        blocks = []

        for _block_idx in range(config.num_target_blocks):
            block = sample_rectangular_block(
                grid_height=config.grid_height,
                grid_width=config.grid_width,
                block_height=config.target_block_height,
                block_width=config.target_block_width,
                device=device,
            )
            blocks.append(block)

        indices = torch.cat(blocks, dim=0)
        unique_indices = torch.unique(indices, sorted=False)

        if unique_indices.numel() == config.num_target_patches:
            return unique_indices

    raise RuntimeError(
        "Failed to sample non-overlapping target blocks. "
        "Try reducing num_target_blocks or target block size."
    )
```

This implementation resamples until the target blocks do not overlap.

It is intentionally simple.

Later, we can implement more efficient non-overlapping block placement.

---

## 2.4.8 Sampling Context Indices

Context indices are sampled from patches not in the target set.

```python
def sample_context_indices(
    config: BlockMaskConfig,
    target_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample context indices from patches not included in the target.

    Returns:
        context_indices:
            Tensor of shape [num_context_patches].
    """
    config.validate()

    all_indices = torch.arange(
        config.num_patches,
        device=device,
        dtype=torch.long,
    )

    is_target = torch.isin(
        all_indices,
        target_indices,
    )

    candidate_context = all_indices[~is_target]

    if candidate_context.numel() < config.num_context_patches:
        raise RuntimeError(
            "Not enough candidate context patches. "
            f"Need {config.num_context_patches}, "
            f"available {candidate_context.numel()}."
        )

    perm = torch.randperm(
        candidate_context.numel(),
        device=device,
    )

    selected = candidate_context[
        perm[: config.num_context_patches]
    ]

    return selected.long()
```

This samples random context patches from the non-target region.

---

## 2.4.9 Sampling a Single Mask Pair

The full single-example sampler is:

```python
def sample_single_mask_pair(
    config: BlockMaskConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample one context-target mask pair.

    Returns:
        context_indices:
            [num_context_patches]

        target_indices:
            [num_target_patches]
    """
    target_indices = sample_target_indices(
        config=config,
        device=device,
    )

    context_indices = sample_context_indices(
        config=config,
        target_indices=target_indices,
        device=device,
    )

    return context_indices, target_indices
```

This function creates one training mask.

---

## 2.4.10 Sampling a Batch of Masks

For a batch, we sample independent masks per image.

```python
def sample_mask_batch(
    config: BlockMaskConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of context and target masks.

    Returns:
        context_indices:
            [B, num_context_patches]

        target_indices:
            [B, num_target_patches]
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    context_masks = []
    target_masks = []

    for _ in range(batch_size):
        context_indices, target_indices = sample_single_mask_pair(
            config=config,
            device=device,
        )

        context_masks.append(context_indices)
        target_masks.append(target_indices)

    context_batch = torch.stack(
        context_masks,
        dim=0,
    )

    target_batch = torch.stack(
        target_masks,
        dim=0,
    )

    return context_batch.long(), target_batch.long()
```

This uses a Python loop over the batch. That is acceptable for the minimal implementation because mask sampling is not the main computational cost.

A later optimized version can vectorize this.

---

## 2.4.11 Overlap Checks

Mask leakage is catastrophic for JEPA.

The context and target sets must not overlap.

Add:

```python
@torch.no_grad()
def mask_overlap_fraction(
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> float:
    """
    Compute the average fraction of target indices that also appear in context.

    Args:
        context_indices:
            [B, N_ctx]

        target_indices:
            [B, N_tgt]

    Returns:
        Average overlap fraction across batch.
    """
    if context_indices.dim() != 2:
        raise ValueError(
            f"Expected context_indices [B, N_ctx], got {context_indices.shape}."
        )

    if target_indices.dim() != 2:
        raise ValueError(
            f"Expected target_indices [B, N_tgt], got {target_indices.shape}."
        )

    if context_indices.size(0) != target_indices.size(0):
        raise ValueError(
            "Batch size mismatch between context_indices and target_indices."
        )

    overlaps = []

    for batch_idx in range(context_indices.size(0)):
        overlap = torch.isin(
            target_indices[batch_idx],
            context_indices[batch_idx],
        )

        overlaps.append(overlap.float().mean())

    return torch.stack(overlaps).mean().item()
```

Add a strict assertion:

```python
def assert_no_mask_overlap(
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> None:
    """
    Raise an error if any context-target overlap exists.
    """
    overlap = mask_overlap_fraction(
        context_indices=context_indices,
        target_indices=target_indices,
    )

    if overlap != 0.0:
        raise ValueError(
            f"Mask leakage detected. Overlap fraction = {overlap:.6f}."
        )
```

During development, call this often.

```python
assert_no_mask_overlap(context_indices, target_indices)
```

During large-scale training, you can log the overlap fraction periodically.

---

## 2.4.12 Boolean Mask Conversion

Sometimes it is useful to convert index masks to boolean masks.

For example:

```python
target_bool.shape
# [B, N]
```

where:

```python
target_bool[b, i] = True
```

if patch \(i\) is a target patch for batch item \(b\).

Add:

```python
def indices_to_bool_mask(
    indices: torch.Tensor,
    num_patches: int,
) -> torch.Tensor:
    """
    Convert patch indices to a boolean mask.

    Args:
        indices:
            [B, K]

        num_patches:
            Total number of patches N.

    Returns:
        mask:
            [B, N] boolean tensor.
    """
    if indices.dim() != 2:
        raise ValueError(
            f"Expected indices with shape [B, K], got {indices.shape}."
        )

    batch_size = indices.size(0)

    mask = torch.zeros(
        batch_size,
        num_patches,
        device=indices.device,
        dtype=torch.bool,
    )

    mask.scatter_(
        dim=1,
        index=indices.long(),
        value=True,
    )

    return mask
```

This is useful for visualization and diagnostics.

---

## 2.4.13 Visualizing Masks

Visualizing masks is essential.

A mask visualization should show:

- context patches,
- target patches,
- unused patches.

Add:

```python
import matplotlib.pyplot as plt


def visualize_mask(
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
    grid_height: int,
    grid_width: int,
    batch_index: int = 0,
) -> None:
    """
    Visualize context and target masks for one batch item.

    Values:
        0 = unused
        1 = context
        2 = target
    """
    if context_indices.dim() != 2 or target_indices.dim() != 2:
        raise ValueError("Expected batched indices with shape [B, K].")

    mask = torch.zeros(
        grid_height * grid_width,
        dtype=torch.long,
    )

    ctx = context_indices[batch_index].detach().cpu().long()
    tgt = target_indices[batch_index].detach().cpu().long()

    mask[ctx] = 1
    mask[tgt] = 2

    mask = mask.reshape(
        grid_height,
        grid_width,
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(mask)
    plt.title("Mask: 0=unused, 1=context, 2=target")
    plt.axis("off")
    plt.colorbar()
    plt.show()
```

This visualization is not meant to be beautiful. It is meant to catch bugs.

A healthy mask should show target blocks and scattered context patches with no overlap.

---

## 2.4.14 Masking Images for Visualization

It is also useful to apply masks to an image.

For example, show target patches in black and context patches visible.

A simple utility:

```python
def apply_patch_mask_to_image(
    image: torch.Tensor,
    visible_indices: torch.Tensor,
    grid_height: int,
    grid_width: int,
    patch_size: int,
    mask_value: float = 0.0,
) -> torch.Tensor:
    """
    Return an image where patches not in visible_indices are masked.

    Args:
        image:
            [C, H, W]

        visible_indices:
            [K]

    Returns:
        masked_image:
            [C, H, W]
    """
    if image.dim() != 3:
        raise ValueError(
            f"Expected image [C, H, W], got {image.shape}."
        )

    masked = torch.full_like(
        image,
        fill_value=mask_value,
    )

    visible = torch.zeros(
        grid_height * grid_width,
        dtype=torch.bool,
        device=image.device,
    )

    visible[visible_indices.long()] = True

    for idx in visible_indices.long():
        row = int(idx.item()) // grid_width
        col = int(idx.item()) % grid_width

        h0 = row * patch_size
        h1 = h0 + patch_size
        w0 = col * patch_size
        w1 = w0 + patch_size

        masked[:, h0:h1, w0:w1] = image[:, h0:h1, w0:w1]

    return masked
```

For context visualization:

```python
masked_context_image = apply_patch_mask_to_image(
    image=image,
    visible_indices=context_indices[0],
    grid_height=12,
    grid_width=12,
    patch_size=8,
)
```

For target visualization:

```python
masked_target_image = apply_patch_mask_to_image(
    image=image,
    visible_indices=target_indices[0],
    grid_height=12,
    grid_width=12,
    patch_size=8,
)
```

This function uses a loop over visible indices, which is fine for visualization.

---

## 2.4.15 marimo Debug Notebook

Extend:

```text
notebooks/01_visualize_patches_and_masks.py
```

Add:

```python
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.datasets import STL10

from jepa_world_model.masks import (
    BlockMaskConfig,
    apply_patch_mask_to_image,
    assert_no_mask_overlap,
    sample_mask_batch,
    visualize_mask,
)
```

Load one image:

```python
dataset = STL10(
    root="data",
    split="unlabeled",
    download=True,
    transform=T.Compose([
        T.Resize((96, 96)),
        T.ToTensor(),
    ]),
)

image, _ = dataset[0]
```

Create a mask config:

```python
config = BlockMaskConfig(
    grid_height=12,
    grid_width=12,
    num_target_blocks=4,
    target_block_height=3,
    target_block_width=3,
    context_ratio=0.6,
)
```

Sample masks:

```python
context_indices, target_indices = sample_mask_batch(
    config=config,
    batch_size=4,
    device=torch.device("cpu"),
)

context_indices.shape, target_indices.shape
```

Check overlap:

```python
assert_no_mask_overlap(
    context_indices,
    target_indices,
)
```

Visualize the mask:

```python
visualize_mask(
    context_indices=context_indices,
    target_indices=target_indices,
    grid_height=12,
    grid_width=12,
    batch_index=0,
)
```

Visualize visible context:

```python
context_image = apply_patch_mask_to_image(
    image=image,
    visible_indices=context_indices[0],
    grid_height=12,
    grid_width=12,
    patch_size=8,
)

plt.figure(figsize=(5, 5))
plt.imshow(context_image.permute(1, 2, 0))
plt.axis("off")
plt.title("Visible context patches")
plt.show()
```

Visualize target patches:

```python
target_image = apply_patch_mask_to_image(
    image=image,
    visible_indices=target_indices[0],
    grid_height=12,
    grid_width=12,
    patch_size=8,
)

plt.figure(figsize=(5, 5))
plt.imshow(target_image.permute(1, 2, 0))
plt.axis("off")
plt.title("Target patches")
plt.show()
```

This notebook lets us inspect whether the masks create a meaningful prediction problem.

---

## 2.4.16 Unit Tests

Create:

```text
tests/test_masks.py
```

Add:

```python
import torch

from jepa_world_model.masks import (
    BlockMaskConfig,
    assert_no_mask_overlap,
    indices_to_bool_mask,
    mask_overlap_fraction,
    sample_mask_batch,
    sample_rectangular_block,
    sample_single_mask_pair,
)


def test_block_mask_config_properties():
    config = BlockMaskConfig(
        grid_height=12,
        grid_width=12,
        num_target_blocks=4,
        target_block_height=3,
        target_block_width=3,
        context_ratio=0.6,
    )

    assert config.num_patches == 144
    assert config.num_target_patches == 36
    assert config.num_context_patches == 86


def test_sample_rectangular_block_shape():
    block = sample_rectangular_block(
        grid_height=12,
        grid_width=12,
        block_height=3,
        block_width=3,
        device=torch.device("cpu"),
    )

    assert block.shape == (9,)
    assert block.min() >= 0
    assert block.max() < 144


def test_sample_single_mask_pair_shapes():
    config = BlockMaskConfig(
        grid_height=12,
        grid_width=12,
        num_target_blocks=4,
        target_block_height=3,
        target_block_width=3,
        context_ratio=0.6,
    )

    context, target = sample_single_mask_pair(
        config=config,
        device=torch.device("cpu"),
    )

    assert context.shape == (config.num_context_patches,)
    assert target.shape == (config.num_target_patches,)


def test_sample_mask_batch_shapes():
    config = BlockMaskConfig(
        grid_height=12,
        grid_width=12,
        num_target_blocks=4,
        target_block_height=3,
        target_block_width=3,
        context_ratio=0.6,
    )

    context, target = sample_mask_batch(
        config=config,
        batch_size=8,
        device=torch.device("cpu"),
    )

    assert context.shape == (8, config.num_context_patches)
    assert target.shape == (8, config.num_target_patches)


def test_masks_do_not_overlap():
    config = BlockMaskConfig(
        grid_height=12,
        grid_width=12,
        num_target_blocks=4,
        target_block_height=3,
        target_block_width=3,
        context_ratio=0.6,
    )

    context, target = sample_mask_batch(
        config=config,
        batch_size=8,
        device=torch.device("cpu"),
    )

    overlap = mask_overlap_fraction(
        context_indices=context,
        target_indices=target,
    )

    assert overlap == 0.0

    assert_no_mask_overlap(
        context_indices=context,
        target_indices=target,
    )


def test_indices_to_bool_mask():
    indices = torch.tensor([
        [0, 2, 4],
        [1, 3, 5],
    ])

    mask = indices_to_bool_mask(
        indices=indices,
        num_patches=6,
    )

    assert mask.shape == (2, 6)
    assert mask.dtype == torch.bool

    expected = torch.tensor([
        [True, False, True, False, True, False],
        [False, True, False, True, False, True],
    ])

    torch.testing.assert_close(mask, expected)
```

Run:

```bash
pytest tests/test_masks.py
```

---

## 2.4.17 First Integration with Patch Tokens

The mask sampler should work with `gather_patches`.

A simple integration test:

```python
from jepa_world_model.patchify import gather_patches


def test_masks_gather_patch_tokens():
    config = BlockMaskConfig(
        grid_height=8,
        grid_width=8,
        num_target_blocks=2,
        target_block_height=2,
        target_block_width=2,
        context_ratio=0.5,
    )

    context, target = sample_mask_batch(
        config=config,
        batch_size=4,
        device=torch.device("cpu"),
    )

    tokens = torch.randn(4, 64, 32)

    context_tokens = gather_patches(
        tokens,
        context,
    )

    target_tokens = gather_patches(
        tokens,
        target,
    )

    assert context_tokens.shape == (4, config.num_context_patches, 32)
    assert target_tokens.shape == (4, config.num_target_patches, 32)
```

This test verifies that masks are compatible with token selection.

---

## 2.4.18 Common Bugs

### Bug 1: Context and target overlap

This is mask leakage.

Always check:

```python
assert_no_mask_overlap(context_indices, target_indices)
```

especially while developing a new mask sampler.

---

### Bug 2: Target blocks overlap unexpectedly

If target blocks overlap, the unique target count becomes smaller than expected.

The minimal sampler resamples until blocks do not overlap.

If sampling fails often, reduce:

```python
num_target_blocks
target_block_height
target_block_width
```

or increase grid size.

---

### Bug 3: Context ratio too high

If:

```text
num_context_patches + num_target_patches > num_patches
```

then the mask configuration is impossible.

Reduce `context_ratio` or target size.

---

### Bug 4: Device mismatch

Masks should be created on the same device as the tensors they index.

Use:

```python
device=images.device
```

when sampling masks during training.

---

### Bug 5: Hidden variable target length

If target blocks overlap and the implementation silently returns fewer target tokens, batching may break later.

The minimal implementation avoids this by requiring a fixed target count.

---

## 2.4.19 How This Differs from the Official I-JEPA Sampler

The official I-JEPA implementation uses a more carefully tuned masking strategy, including scale and aspect-ratio sampling. The goal is to generate prediction tasks that encourage semantic representation learning.

Our minimal sampler is simpler.

It uses fixed rectangular target blocks and random context patches from the remaining image.

This is enough to implement the JEPA mechanism clearly.

Later, we can extend the sampler with:

- random target block scales,
- random aspect ratios,
- multiple context blocks,
- minimum distance constraints,
- target padding masks,
- per-sample variable target counts.

For now, correctness and clarity are more important than full parity.

---

## 2.4.20 Summary

Mask sampling defines the JEPA prediction task.

In this section, we implemented:

- `BlockMaskConfig`,
- rectangular target block sampling,
- target mask sampling,
- context sampling from non-target patches,
- batch mask sampling,
- overlap diagnostics,
- boolean mask conversion,
- mask visualization,
- image masking for visualization,
- unit tests,
- integration with `gather_patches`.

The next section builds the minimal ViT encoder that will process context and target patch tokens.

---

## References and Further Reading

- Mahmoud Assran et al., **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.
  <https://arxiv.org/abs/2301.08243>

- Facebook Research, **Official I-JEPA Codebase**.
  <https://github.com/facebookresearch/ijepa>

- Kaiming He et al., **Masked Autoencoders Are Scalable Vision Learners**, 2021.
  <https://arxiv.org/abs/2111.06377>

- PyTorch, **torch.isin documentation**.
  <https://pytorch.org/docs/stable/generated/torch.isin.html>

- PyTorch, **torch.scatter documentation**.
  <https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html>
