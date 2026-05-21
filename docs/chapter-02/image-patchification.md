# 2.2 Image Patchification

The first implementation step is turning an image into a sequence of patches.

A Vision Transformer does not process an image as a 2D grid of pixels directly. It divides the image into fixed-size patches, flattens each patch, projects each patch into an embedding vector, and then processes the resulting sequence of patch tokens.

For JEPA, patchification is especially important because context and target masks are defined over patch indices.

The image:

\[
x \in \mathbb{R}^{B \times C \times H \times W}
\]

becomes a sequence:

\[
p \in \mathbb{R}^{B \times N \times (C P^2)}
\]

where:

- \(B\) is batch size,
- \(C\) is number of channels,
- \(H\) is image height,
- \(W\) is image width,
- \(P\) is patch size,
- \(N\) is the number of patches.

For square images with \(H = W\):

\[
N = \left(\frac{H}{P}\right)^2
\]

For example, with \(96 \times 96\) images and \(8 \times 8\) patches:

\[
N = \left(\frac{96}{8}\right)^2 = 12^2 = 144
\]

Each patch contains:

\[
C P^2 = 3 \times 8^2 = 192
\]

raw values.

After patchification, the model can select arbitrary patch subsets:

\[
x_{\mathcal{C}}
\]

for context patches, and:

\[
x_{\mathcal{T}}
\]

for target patches.

This section implements patchification carefully and tests that it is reversible.

---

## 2.2.1 Why Patchification Matters for JEPA

In I-JEPA, the model does not usually mask individual pixels. It masks image regions at the patch level.

The core data flow is:

```text
image
  ↓
patch sequence
  ↓
select context patch indices
  ↓
online encoder
```

and:

```text
image
  ↓
patch sequence
  ↓
select target patch indices
  ↓
target encoder
```

This means patch indices become the coordinate system of the whole model.

The same indexing convention must be used by:

- patchification,
- context masks,
- target masks,
- positional embeddings,
- predictor target queries,
- mask visualization.

If patch indexing is inconsistent, the model may receive the wrong positional information or predict the wrong target locations.

So we start with explicit, tested patch utilities.

---

## 2.2.2 Patch Grid Convention

Assume an image is split into a grid of patches.

For a square image:

\[
H = W
\]

and patch size:

\[
P
\]

the grid size is:

\[
G = H / P
\]

The patch grid has shape:

\[
G \times G
\]

We flatten this grid row-major.

The patch at row \(r\) and column \(c\) has index:

\[
i = rG + c
\]

So a \(4 \times 4\) patch grid is indexed as:

```text
 0   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14  15
```

In code:

```python
def patch_index(row: int, col: int, grid_size: int) -> int:
    return row * grid_size + col
```

The inverse is:

```python
def patch_row_col(index: int, grid_size: int) -> tuple[int, int]:
    row = index // grid_size
    col = index % grid_size
    return row, col
```

This indexing convention will be used throughout Chapter 2.

---

## 2.2.3 Shape Convention

We will implement patchification for images with shape:

```python
images.shape
# [B, C, H, W]
```

The patchified output should have shape:

```python
patches.shape
# [B, N, C * P * P]
```

where:

```python
N = (H // P) * (W // P)
```

For square images:

```python
G = H // P
N = G * G
```

Example:

```python
B = 16
C = 3
H = W = 96
P = 8

images.shape
# [16, 3, 96, 96]

patches.shape
# [16, 144, 192]
```

For JEPA, we also need to gather selected patches.

Given:

```python
patches.shape
# [B, N, patch_dim]

indices.shape
# [B, K]
```

we want:

```python
selected.shape
# [B, K, patch_dim]
```

where each batch item may select different patch indices.

---

## 2.2.4 Implementing `patchify.py`

Create:

```text
src/jepa_world_model/patchify.py
```

Add the following implementation.

```python
from __future__ import annotations

import torch


def image_to_patches(
    images: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """
    Convert a batch of images into flattened patches.

    Args:
        images:
            Tensor of shape [B, C, H, W].

        patch_size:
            Spatial size P of each square patch.

    Returns:
        patches:
            Tensor of shape [B, N, C * P * P], where
            N = (H // P) * (W // P).

    Raises:
        ValueError:
            If images is not rank 4 or H/W are not divisible by patch_size.
    """
    if images.dim() != 4:
        raise ValueError(
            f"Expected images with shape [B, C, H, W], got {images.shape}."
        )

    batch_size, channels, height, width = images.shape

    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image height and width must be divisible by patch_size. "
            f"Got height={height}, width={width}, patch_size={patch_size}."
        )

    grid_h = height // patch_size
    grid_w = width // patch_size

    patches = images.reshape(
        batch_size,
        channels,
        grid_h,
        patch_size,
        grid_w,
        patch_size,
    )

    patches = patches.permute(
        0, 2, 4, 1, 3, 5
    )

    patches = patches.reshape(
        batch_size,
        grid_h * grid_w,
        channels * patch_size * patch_size,
    )

    return patches


def patches_to_image(
    patches: torch.Tensor,
    patch_size: int,
    image_height: int,
    image_width: int,
    channels: int,
) -> torch.Tensor:
    """
    Reconstruct images from flattened patches.

    Args:
        patches:
            Tensor of shape [B, N, C * P * P].

        patch_size:
            Spatial size P of each square patch.

        image_height:
            Reconstructed image height.

        image_width:
            Reconstructed image width.

        channels:
            Number of image channels.

    Returns:
        images:
            Tensor of shape [B, C, H, W].
    """
    if patches.dim() != 3:
        raise ValueError(
            f"Expected patches with shape [B, N, patch_dim], got {patches.shape}."
        )

    batch_size, num_patches, patch_dim = patches.shape

    expected_patch_dim = channels * patch_size * patch_size
    if patch_dim != expected_patch_dim:
        raise ValueError(
            f"Patch dimension mismatch. Expected {expected_patch_dim}, "
            f"got {patch_dim}."
        )

    if image_height % patch_size != 0 or image_width % patch_size != 0:
        raise ValueError(
            f"Image height and width must be divisible by patch_size. "
            f"Got image_height={image_height}, image_width={image_width}, "
            f"patch_size={patch_size}."
        )

    grid_h = image_height // patch_size
    grid_w = image_width // patch_size

    expected_num_patches = grid_h * grid_w
    if num_patches != expected_num_patches:
        raise ValueError(
            f"Number of patches mismatch. Expected {expected_num_patches}, "
            f"got {num_patches}."
        )

    images = patches.reshape(
        batch_size,
        grid_h,
        grid_w,
        channels,
        patch_size,
        patch_size,
    )

    images = images.permute(
        0, 3, 1, 4, 2, 5
    )

    images = images.reshape(
        batch_size,
        channels,
        image_height,
        image_width,
    )

    return images
```

The two key operations are:

```python
reshape
permute
reshape
```

This avoids Python loops and keeps patchification differentiable.

---

## 2.2.5 Understanding the Reshape

The input image tensor is:

```python
[B, C, H, W]
```

We first split height and width into grid and patch dimensions:

```python
[B, C, G_h, P, G_w, P]
```

Then we permute to bring the patch grid dimensions before the patch content dimensions:

```python
[B, G_h, G_w, C, P, P]
```

Finally, we flatten:

```python
[B, G_h * G_w, C * P * P]
```

This gives row-major patch ordering because \(G_h\) and \(G_w\) appear in that order before flattening.

For reconstruction, we reverse the operations.

---

## 2.2.6 Patch Index Utilities

Add the following to `patchify.py`.

```python
def num_patches(
    image_height: int,
    image_width: int,
    patch_size: int,
) -> int:
    """
    Return the number of patches for an image size and patch size.
    """
    if image_height % patch_size != 0 or image_width % patch_size != 0:
        raise ValueError(
            f"Image size must be divisible by patch_size. "
            f"Got image_height={image_height}, image_width={image_width}, "
            f"patch_size={patch_size}."
        )

    return (image_height // patch_size) * (image_width // patch_size)


def grid_size(
    image_height: int,
    image_width: int,
    patch_size: int,
) -> tuple[int, int]:
    """
    Return patch grid size as (grid_h, grid_w).
    """
    if image_height % patch_size != 0 or image_width % patch_size != 0:
        raise ValueError(
            f"Image size must be divisible by patch_size. "
            f"Got image_height={image_height}, image_width={image_width}, "
            f"patch_size={patch_size}."
        )

    return image_height // patch_size, image_width // patch_size


def patch_index(
    row: int,
    col: int,
    grid_width: int,
) -> int:
    """
    Convert patch row/column into flattened row-major index.
    """
    return row * grid_width + col


def patch_row_col(
    index: int,
    grid_width: int,
) -> tuple[int, int]:
    """
    Convert flattened row-major patch index into row/column.
    """
    row = index // grid_width
    col = index % grid_width
    return row, col
```

For square images, `grid_width == grid_size`.

For rectangular images, row-major indexing still works with `grid_width`.

---

## 2.2.7 Gathering Selected Patches

JEPA needs to select context and target patches.

Suppose:

```python
patches.shape
# [B, N, D]

indices.shape
# [B, K]
```

We need:

```python
selected.shape
# [B, K, D]
```

Add:

```python
def gather_patches(
    patches: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Gather selected patches for each batch item.

    Args:
        patches:
            Tensor of shape [B, N, D].

        indices:
            Long tensor of shape [B, K].

    Returns:
        selected:
            Tensor of shape [B, K, D].
    """
    if patches.dim() != 3:
        raise ValueError(
            f"Expected patches with shape [B, N, D], got {patches.shape}."
        )

    if indices.dim() != 2:
        raise ValueError(
            f"Expected indices with shape [B, K], got {indices.shape}."
        )

    batch_size, _, dim = patches.shape

    if indices.size(0) != batch_size:
        raise ValueError(
            f"Batch size mismatch: patches has B={batch_size}, "
            f"indices has B={indices.size(0)}."
        )

    if indices.dtype != torch.long:
        indices = indices.long()

    expanded_indices = indices.unsqueeze(-1).expand(
        -1,
        -1,
        dim,
    )

    return torch.gather(
        patches,
        dim=1,
        index=expanded_indices,
    )
```

This function will later be used by both the online encoder and the target encoder.

Example:

```python
patches = image_to_patches(images, patch_size=8)

context_patches = gather_patches(
    patches,
    context_indices,
)

target_patches = gather_patches(
    patches,
    target_indices,
)
```

---

## 2.2.8 Patch Embedding

Raw flattened patches are not yet transformer tokens.

A transformer expects vectors of dimension \(D\). We therefore apply a learned linear projection:

\[
e_i = W p_i + b
\]

where:

- \(p_i \in \mathbb{R}^{C P^2}\),
- \(e_i \in \mathbb{R}^{D}\).

A minimal patch embedding module:

```python
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Convert images into patch embeddings.

    Input:
        images [B, C, H, W]

    Output:
        tokens [B, N, D]
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size must be divisible by patch_size. "
                f"Got image_size={image_size}, patch_size={patch_size}."
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = in_channels * patch_size * patch_size

        self.proj = nn.Linear(
            self.patch_dim,
            embed_dim,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = image_to_patches(
            images,
            patch_size=self.patch_size,
        )

        tokens = self.proj(patches)

        return tokens
```

This implementation is very explicit.

Many ViT implementations use a convolution for patch embedding:

```python
nn.Conv2d(
    in_channels,
    embed_dim,
    kernel_size=patch_size,
    stride=patch_size,
)
```

The convolution version is efficient and equivalent to a linear projection over non-overlapping patches. We use the explicit patchify-plus-linear version first because it is easier to inspect and test.

Later, if needed, we can replace it with a convolutional patch embedding.

---

## 2.2.9 Convolutional Patch Embedding Alternative

The convolutional version is:

```python
class ConvPatchEmbed(nn.Module):
    """
    Patch embedding using a Conv2d projection.

    This is equivalent to applying a learned linear projection to
    each non-overlapping patch.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size must be divisible by patch_size. "
                f"Got image_size={image_size}, patch_size={patch_size}."
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.proj(images)

        # [B, D, G, G] -> [B, G, G, D]
        x = x.permute(0, 2, 3, 1)

        # [B, G, G, D] -> [B, N, D]
        x = x.reshape(images.size(0), self.num_patches, self.embed_dim)

        return x
```

Either patch embedding can be used by the encoder.

For the minimal tutorial implementation, the explicit `PatchEmbed` is easier to reason about. For performance, `ConvPatchEmbed` is often preferable.

---

## 2.2.10 Visualizing Patches

Patchification should be visualized before moving on.

For a single image, we can display its patches in a grid.

Add a utility to `patchify.py` or use it in a marimo notebook:

```python
import matplotlib.pyplot as plt


def show_patch_grid(
    image: torch.Tensor,
    patch_size: int,
) -> None:
    """
    Visualize all patches from a single image.

    Args:
        image:
            Tensor of shape [C, H, W].

        patch_size:
            Patch size.
    """
    if image.dim() != 3:
        raise ValueError(
            f"Expected image with shape [C, H, W], got {image.shape}."
        )

    image_batch = image.unsqueeze(0)
    patches = image_to_patches(image_batch, patch_size=patch_size)

    channels, height, width = image.shape
    grid_h, grid_w = grid_size(height, width, patch_size)

    patches = patches.squeeze(0)
    patches = patches.reshape(
        grid_h,
        grid_w,
        channels,
        patch_size,
        patch_size,
    )

    fig, axes = plt.subplots(
        grid_h,
        grid_w,
        figsize=(grid_w, grid_h),
    )

    for row in range(grid_h):
        for col in range(grid_w):
            ax = axes[row, col]
            patch = patches[row, col]

            if channels == 1:
                ax.imshow(patch.squeeze(0).cpu(), cmap="gray")
            else:
                patch = patch.permute(1, 2, 0).cpu()
                patch = patch.clamp(0, 1)
                ax.imshow(patch)

            ax.axis("off")

    plt.tight_layout()
    plt.show()
```

For \(96 \times 96\) images with patch size 8, this creates a \(12 \times 12\) grid.

This visualization helps catch:

- wrong patch ordering,
- wrong channel ordering,
- broken normalization,
- unexpected image shape.

---

## 2.2.11 marimo Debug Notebook

Create:

```text
notebooks/01_visualize_patches_and_masks.py
```

Open it with:

```bash
uv run marimo edit notebooks/01_visualize_patches_and_masks.py
```

The notebook should load one image and inspect patchification.

Example cells:

```python
import torch
import torchvision.transforms as T
from torchvision.datasets import STL10

from jepa_world_model.patchify import (
    image_to_patches,
    patches_to_image,
    show_patch_grid,
)
```

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
image.shape
```

```python
patch_size = 8

patches = image_to_patches(
    image.unsqueeze(0),
    patch_size=patch_size,
)

patches.shape
```

```python
reconstructed = patches_to_image(
    patches,
    patch_size=patch_size,
    image_height=96,
    image_width=96,
    channels=3,
)

torch.testing.assert_close(
    reconstructed,
    image.unsqueeze(0),
)
```

```python
show_patch_grid(
    image,
    patch_size=patch_size,
)
```

This notebook becomes the first visual sanity check.

---

## 2.2.12 Unit Tests

Create:

```text
tests/test_patchify.py
```

Add:

```python
import pytest
import torch

from jepa_world_model.patchify import (
    gather_patches,
    grid_size,
    image_to_patches,
    num_patches,
    patch_index,
    patch_row_col,
    patches_to_image,
)


def test_num_patches_square_image():
    assert num_patches(
        image_height=96,
        image_width=96,
        patch_size=8,
    ) == 144


def test_grid_size_square_image():
    assert grid_size(
        image_height=96,
        image_width=96,
        patch_size=8,
    ) == (12, 12)


def test_patch_index_roundtrip():
    grid_width = 12

    for row in range(12):
        for col in range(12):
            idx = patch_index(row, col, grid_width)
            recovered_row, recovered_col = patch_row_col(idx, grid_width)

            assert recovered_row == row
            assert recovered_col == col


def test_patchify_shape():
    images = torch.randn(4, 3, 96, 96)

    patches = image_to_patches(
        images,
        patch_size=8,
    )

    assert patches.shape == (4, 144, 192)


def test_patchify_unpatchify_roundtrip():
    images = torch.randn(2, 3, 32, 32)

    patches = image_to_patches(
        images,
        patch_size=4,
    )

    reconstructed = patches_to_image(
        patches,
        patch_size=4,
        image_height=32,
        image_width=32,
        channels=3,
    )

    torch.testing.assert_close(reconstructed, images)


def test_patchify_rejects_bad_size():
    images = torch.randn(2, 3, 30, 30)

    with pytest.raises(ValueError):
        image_to_patches(
            images,
            patch_size=8,
        )


def test_gather_patches_shape():
    patches = torch.randn(2, 16, 32)

    indices = torch.tensor([
        [0, 1, 2],
        [5, 6, 7],
    ])

    selected = gather_patches(
        patches,
        indices,
    )

    assert selected.shape == (2, 3, 32)


def test_gather_patches_values():
    patches = torch.arange(
        2 * 4 * 1,
        dtype=torch.float32,
    ).reshape(2, 4, 1)

    indices = torch.tensor([
        [0, 2],
        [1, 3],
    ])

    selected = gather_patches(
        patches,
        indices,
    )

    expected = torch.tensor([
        [[0.0], [2.0]],
        [[5.0], [7.0]],
    ])

    torch.testing.assert_close(selected, expected)
```

Run:

```bash
uv run pytest tests/test_patchify.py
```

Expected result:

```text
8 passed
```

---

## 2.2.13 Smoke Test for `PatchEmbed`

Add to `tests/test_patchify.py`:

```python
from jepa_world_model.patchify import PatchEmbed


def test_patch_embed_shape():
    module = PatchEmbed(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=64,
    )

    images = torch.randn(2, 3, 32, 32)

    tokens = module(images)

    assert tokens.shape == (2, 64, 64)
```

Why:

```text
image_size = 32
patch_size = 4
grid_size = 8
num_patches = 8 * 8 = 64
embed_dim = 64
```

So:

```python
tokens.shape
# [2, 64, 64]
```

---

## 2.2.14 Patchification and JEPA Encoders

The encoder will eventually perform:

```python
tokens = patch_embed(images)
tokens = tokens + pos_embed
selected = gather_patches(tokens, patch_indices)
encoded = transformer(selected)
```

The target encoder will do the same with target indices.

The context encoder:

```python
context_tokens = gather_patches(tokens, context_indices)
```

The target encoder:

```python
target_tokens = gather_patches(tokens, target_indices)
```

This means `gather_patches` must work for any token sequence, not only raw flattened patches.

The same function applies to:

```python
raw_patches.shape
# [B, N, C * P * P]

embedded_tokens.shape
# [B, N, D]

positional_embeddings.shape
# [B, N, D]
```

As long as the tensor is shaped:

```python
[B, N, D]
```

we can gather along the patch dimension.

---

## 2.2.15 Common Bugs

### Bug 1: Wrong patch order

If patches are flattened column-major instead of row-major, positional embeddings and masks will not align with image space.

The test:

```python
test_patch_index_roundtrip
```

protects the indexing convention, but visualization is still useful.

---

### Bug 2: Wrong channel order

PyTorch images usually use:

```python
[C, H, W]
```

but plotting libraries often expect:

```python
[H, W, C]
```

When visualizing:

```python
image_for_plot = image.permute(1, 2, 0)
```

---

### Bug 3: Non-contiguous tensors

After `permute`, tensors may be non-contiguous. Using `reshape` is usually safer than `view` because `reshape` handles non-contiguous tensors by copying if needed.

This is why the implementation uses:

```python
patches.reshape(...)
```

rather than:

```python
patches.view(...)
```

---

### Bug 4: Image size not divisible by patch size

Patchification assumes non-overlapping patches.

This requires:

\[
H \mod P = 0
\]

and:

\[
W \mod P = 0
\]

The implementation raises a `ValueError` if this is not true.

---

### Bug 5: Gathering indices with wrong dtype

`torch.gather` requires integer indices, usually `torch.long`.

The implementation converts:

```python
indices = indices.long()
```

if needed.

Still, mask samplers should produce `torch.long` tensors directly.

---

## 2.2.16 Summary

Patchification converts images into a sequence of patch vectors.

For JEPA, this is more than a preprocessing step. Patch indices become the coordinate system for:

- context masks,
- target masks,
- positional embeddings,
- predictor queries,
- visualization,
- mask leakage checks.

In this section, we implemented:

- `image_to_patches`,
- `patches_to_image`,
- patch grid indexing utilities,
- `gather_patches`,
- `PatchEmbed`,
- optional `ConvPatchEmbed`,
- unit tests,
- a marimo visualization workflow.

The next section adds positional embeddings so that patch tokens carry spatial information.

---

## References and Further Reading

- Alexey Dosovitskiy et al., **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**, 2020.  
  <https://arxiv.org/abs/2010.11929>

- Kaiming He et al., **Masked Autoencoders Are Scalable Vision Learners**, 2021.  
  <https://arxiv.org/abs/2111.06377>

- Mahmoud Assran et al., **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.  
  <https://arxiv.org/abs/2301.08243>

- PyTorch, **torch.gather documentation**.  
  <https://pytorch.org/docs/stable/generated/torch.gather.html>

- marimo, **Documentation**.  
  <https://docs.marimo.io/>
