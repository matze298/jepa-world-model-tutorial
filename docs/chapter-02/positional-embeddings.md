# 2.3 Positional Embeddings

Patchification turns an image into a sequence of patch tokens.

But a sequence alone does not tell the model where each patch came from.

After patchification and linear projection, an image becomes a sequence of patch embeddings:

\[
p \in \mathbb{R}^{B \times N \times D}
\]

where \(B\) is batch size, \(N\) is the number of patches, and \(D\) is the embedding dimension. A transformer sees this as a set or sequence of vectors. Without positional information, patch token \(17\) and patch token \(83\) are just two vectors. The model has no intrinsic knowledge that one patch came from the top-left of the image and another came from the bottom-right.

For JEPA, this is especially important.

The predictor receives context representations and must predict the representation of specific target patches. It cannot do that well unless it knows where the context and target patches are located.

This section implements positional embeddings for patch tokens.

---

## 2.3.1 Why Positional Embeddings Matter

A transformer encoder is permutation-equivariant over its input tokens unless positional information is added.

If we feed the same patch tokens in a different order without positional embeddings, the model has no way to distinguish the spatial layout. But image understanding depends heavily on spatial structure.

For example:

```text
sky above grass
```

is different from:

```text
grass above sky
```

even if the same local patches are present.

For JEPA, spatial information matters in two places.

First, the encoder needs to know where context patches are located:

\[
z_{\mathcal{C}} = f_\theta(x_{\mathcal{C}}, \mathcal{C})
\]

Second, the predictor needs to know which target positions to predict:

\[
\hat{z}_{\mathcal{T}} = g_\theta(z_{\mathcal{C}}, \mathcal{C}, \mathcal{T})
\]

So the predictor is not merely answering:

> What is missing?

It is answering:

> What should be at these target positions?

That is why target positional embeddings are not optional.

---

## 2.3.2 Patch Position Convention

From the previous section, a patch at row \(r\) and column \(c\) has flattened index:

\[
i = rG + c
\]

where \(G\) is the grid width.

For a \(4 \times 4\) grid:

```text
 0   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14  15
```

A positional embedding assigns a vector to each patch index:

\[
e_i^{\mathrm{pos}} \in \mathbb{R}^{D}
\]

Patch tokens are then modified as:

\[
h_i = p_i + e_i^{\mathrm{pos}}
\]

where:

- \(p_i\) is the patch embedding,
- \(e_i^{\mathrm{pos}}\) is the position embedding,
- \(h_i\) is the final token passed to the transformer.

In code:

```python
tokens = patch_embed(images)
tokens = tokens + pos_embed
```

When using selected patch indices:

```python
selected_tokens = gather_patches(tokens, indices)
```

or equivalently:

```python
selected_patch_tokens = gather_patches(patch_tokens, indices)
selected_pos = position_embedding(indices)
selected_tokens = selected_patch_tokens + selected_pos
```

The second version is more explicit and is the one we will use in our minimal implementation.

---

## 2.3.3 Learned Positional Embeddings

The simplest option is a learned embedding table.

For \(N\) patches and embedding dimension \(D\):

\[
E \in \mathbb{R}^{N \times D}
\]

Each patch index \(i\) selects one row:

\[
e_i = E_i
\]

In PyTorch, this is exactly `nn.Embedding`.

Create:

```text
src/jepa_world_model/position.py
```

Add:

```python
from __future__ import annotations

import torch
import torch.nn as nn


class Learned2DPositionEmbedding(nn.Module):
    """
    Learned position embeddings for a 2D patch grid.

    The grid is flattened in row-major order.

    Input:
        indices [B, K]

    Output:
        embeddings [B, K, D]
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        embed_dim: int,
    ):
        super().__init__()

        if grid_height <= 0 or grid_width <= 0:
            raise ValueError(
                f"grid_height and grid_width must be positive. "
                f"Got grid_height={grid_height}, grid_width={grid_width}."
            )

        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_positions = grid_height * grid_width
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(
            num_embeddings=self.num_positions,
            embedding_dim=embed_dim,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(
            self.embedding.weight,
            std=0.02,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() != 2:
            raise ValueError(
                f"Expected indices with shape [B, K], got {indices.shape}."
            )

        if indices.dtype != torch.long:
            indices = indices.long()

        if indices.numel() > 0:
            min_index = int(indices.min().item())
            max_index = int(indices.max().item())

            if min_index < 0 or max_index >= self.num_positions:
                raise ValueError(
                    f"Position index out of range. "
                    f"Valid range is [0, {self.num_positions - 1}], "
                    f"got min={min_index}, max={max_index}."
                )

        return self.embedding(indices)
```

This module maps patch indices to learned vectors.

Example:

```python
pos = Learned2DPositionEmbedding(
    grid_height=12,
    grid_width=12,
    embed_dim=192,
)

indices = torch.tensor([
    [0, 1, 2],
    [10, 11, 12],
])

emb = pos(indices)

emb.shape
# [2, 3, 192]
```

Learned positional embeddings are simple and work well when training and evaluation use the same image resolution.

---

## 2.3.4 Full-Grid Positional Embeddings

Sometimes we want the full positional embedding table as a sequence:

```python
pos_tokens.shape
# [1, N, D]
```

Add this method to `Learned2DPositionEmbedding`:

```python
    def full_grid(self, batch_size: int = 1) -> torch.Tensor:
        """
        Return positional embeddings for all grid positions.

        Returns:
            [batch_size, N, D]
        """
        device = self.embedding.weight.device

        indices = torch.arange(
            self.num_positions,
            device=device,
            dtype=torch.long,
        )

        indices = indices.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        return self.forward(indices)
```

Then:

```python
pos_tokens = pos.full_grid(batch_size=images.size(0))

tokens = patch_tokens + pos_tokens
```

This is useful when the encoder processes all tokens.

For JEPA, we often gather selected patches first and then add selected positional embeddings:

```python
patch_tokens = patch_embed(images)
context_tokens = gather_patches(patch_tokens, context_indices)
context_pos = pos_embed(context_indices)

context_tokens = context_tokens + context_pos
```

---

## 2.3.5 Fixed Sinusoidal 2D Positional Embeddings

Learned positional embeddings are simple, but they depend on the training grid size.

A fixed sinusoidal embedding can be useful because it does not add learned parameters and can generalize more naturally across positions.

For a 2D grid, we build one sinusoidal embedding for the row coordinate and one for the column coordinate, then concatenate them.

For a position \((r, c)\):

\[
e_{r,c}
=
[\mathrm{sinusoid}(r), \mathrm{sinusoid}(c)]
\]

A standard 1D sinusoidal embedding uses frequencies:

\[
\omega_k = 1 / 10000^{2k / D}
\]

and:

\[
\mathrm{PE}(p, 2k) = \sin(p \omega_k)
\]

\[
\mathrm{PE}(p, 2k + 1) = \cos(p \omega_k)
\]

We will implement a simple version.

Add to `position.py`:

```python
def sinusoidal_1d_positions(
    positions: torch.Tensor,
    dim: int,
    temperature: float = 10_000.0,
) -> torch.Tensor:
    """
    Create sinusoidal embeddings for 1D positions.

    Args:
        positions:
            Tensor of shape [N].

        dim:
            Embedding dimension. Must be even.

        temperature:
            Frequency temperature.

    Returns:
        embeddings:
            Tensor of shape [N, dim].
    """
    if dim % 2 != 0:
        raise ValueError(
            f"sinusoidal_1d_positions requires even dim, got {dim}."
        )

    positions = positions.float()

    half_dim = dim // 2

    omega = torch.arange(
        half_dim,
        device=positions.device,
        dtype=torch.float32,
    )

    omega = 1.0 / (temperature ** (omega / half_dim))

    out = positions[:, None] * omega[None, :]

    emb = torch.cat(
        [torch.sin(out), torch.cos(out)],
        dim=1,
    )

    return emb
```

Now implement 2D sinusoidal embeddings:

```python
def sinusoidal_2d_grid(
    grid_height: int,
    grid_width: int,
    embed_dim: int,
    temperature: float = 10_000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create fixed 2D sinusoidal embeddings for a patch grid.

    Returns:
        embeddings [N, D], where N = grid_height * grid_width.
    """
    if embed_dim % 2 != 0:
        raise ValueError(
            f"2D sinusoidal embeddings require even embed_dim, got {embed_dim}."
        )

    row_dim = embed_dim // 2
    col_dim = embed_dim // 2

    if row_dim % 2 != 0 or col_dim % 2 != 0:
        raise ValueError(
            "For this simple implementation, embed_dim must be divisible by 4. "
            f"Got embed_dim={embed_dim}."
        )

    rows = torch.arange(
        grid_height,
        device=device,
        dtype=torch.float32,
    )

    cols = torch.arange(
        grid_width,
        device=device,
        dtype=torch.float32,
    )

    row_emb = sinusoidal_1d_positions(
        rows,
        dim=row_dim,
        temperature=temperature,
    )

    col_emb = sinusoidal_1d_positions(
        cols,
        dim=col_dim,
        temperature=temperature,
    )

    rr, cc = torch.meshgrid(
        torch.arange(grid_height, device=device),
        torch.arange(grid_width, device=device),
        indexing="ij",
    )

    emb = torch.cat(
        [
            row_emb[rr.reshape(-1)],
            col_emb[cc.reshape(-1)],
        ],
        dim=1,
    )

    return emb
```

This returns:

```python
emb.shape
# [grid_height * grid_width, embed_dim]
```

For a \(12 \times 12\) grid and \(D=192\):

```python
emb = sinusoidal_2d_grid(12, 12, 192)
emb.shape
# [144, 192]
```

---

## 2.3.6 Fixed Position Embedding Module

Wrap the sinusoidal grid in a module.

```python
class FixedSinCos2DPositionEmbedding(nn.Module):
    """
    Fixed 2D sinusoidal position embeddings.

    Input:
        indices [B, K]

    Output:
        embeddings [B, K, D]
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        embed_dim: int,
        temperature: float = 10_000.0,
    ):
        super().__init__()

        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_positions = grid_height * grid_width
        self.embed_dim = embed_dim

        emb = sinusoidal_2d_grid(
            grid_height=grid_height,
            grid_width=grid_width,
            embed_dim=embed_dim,
            temperature=temperature,
        )

        self.register_buffer(
            "embedding",
            emb,
            persistent=False,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() != 2:
            raise ValueError(
                f"Expected indices with shape [B, K], got {indices.shape}."
            )

        if indices.dtype != torch.long:
            indices = indices.long()

        if indices.numel() > 0:
            min_index = int(indices.min().item())
            max_index = int(indices.max().item())

            if min_index < 0 or max_index >= self.num_positions:
                raise ValueError(
                    f"Position index out of range. "
                    f"Valid range is [0, {self.num_positions - 1}], "
                    f"got min={min_index}, max={max_index}."
                )

        return self.embedding[indices]

    def full_grid(self, batch_size: int = 1) -> torch.Tensor:
        indices = torch.arange(
            self.num_positions,
            device=self.embedding.device,
            dtype=torch.long,
        )

        indices = indices.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        return self.forward(indices)
```

Now both learned and fixed positional embeddings have the same interface:

```python
pos_embed(indices)
pos_embed.full_grid(batch_size)
```

This lets the encoder switch between them easily.

---

## 2.3.7 Which Positional Embedding Should We Use?

For the minimal implementation, use **learned positional embeddings**.

They are:

- simple,
- easy to inspect,
- standard in ViT-style models,
- sufficient for fixed-resolution experiments.

Use fixed sinusoidal embeddings when:

- you want fewer learned parameters,
- you want simpler interpolation,
- you want deterministic position encodings,
- you want to experiment with resolution changes.

For Chapter 2, the default is:

```python
Learned2DPositionEmbedding
```

Later, we can add a config switch:

```python
position_embedding: str = "learned"
```

with options:

```text
learned
sincos
```

---

## 2.3.8 Adding Position Embeddings to Patch Tokens

Given patch tokens:

```python
patch_tokens.shape
# [B, N, D]
```

and selected indices:

```python
indices.shape
# [B, K]
```

we gather patch tokens and position embeddings separately:

```python
selected_patch_tokens = gather_patches(
    patch_tokens,
    indices,
)

selected_pos = pos_embed(indices)

selected_tokens = selected_patch_tokens + selected_pos
```

This is the core operation used by the encoder.

A helper function can make this explicit:

```python
from jepa_world_model.patchify import gather_patches


def gather_tokens_with_positions(
    patch_tokens: torch.Tensor,
    indices: torch.Tensor,
    position_embedding: nn.Module,
) -> torch.Tensor:
    """
    Gather patch tokens and add positional embeddings.

    Args:
        patch_tokens:
            Tensor of shape [B, N, D].

        indices:
            Tensor of shape [B, K].

        position_embedding:
            Module mapping indices [B, K] -> [B, K, D].

    Returns:
        selected tokens:
            Tensor of shape [B, K, D].
    """
    selected = gather_patches(
        patch_tokens,
        indices,
    )

    pos = position_embedding(indices)

    if selected.shape != pos.shape:
        raise ValueError(
            f"Token and position shapes differ: "
            f"selected={selected.shape}, pos={pos.shape}."
        )

    return selected + pos
```

This helper is optional, but it makes the encoder easier to read.

---

## 2.3.9 Positional Embeddings in the Predictor

The encoder uses context positions.

The predictor uses both context and target positions.

For image JEPA, the predictor receives:

```python
context_repr.shape
# [B, N_ctx, D]

context_indices.shape
# [B, N_ctx]

target_indices.shape
# [B, N_tgt]
```

The predictor projects context representations into predictor dimension:

```python
context_tokens = context_proj(context_repr)
```

Then it adds context positional embeddings:

```python
context_tokens = context_tokens + predictor_pos_embed(context_indices)
```

For targets, it creates learned query tokens:

```python
target_queries = learned_target_query.expand(B, N_tgt, -1)
```

Then it adds target positional embeddings:

```python
target_queries = target_queries + predictor_pos_embed(target_indices)
```

Then the predictor transformer processes:

```python
tokens = torch.cat(
    [context_tokens, target_queries],
    dim=1,
)
```

The output corresponding to target query positions becomes:

```python
pred_repr.shape
# [B, N_tgt, D]
```

This is why the predictor needs a positional embedding module of its own.

The encoder embedding dimension and predictor embedding dimension may differ:

```text
encoder_dim = D
predictor_dim = D_p
```

So the encoder and predictor usually use separate positional embeddings:

```python
encoder_pos_embed = Learned2DPositionEmbedding(..., embed_dim=D)
predictor_pos_embed = Learned2DPositionEmbedding(..., embed_dim=D_p)
```

---

## 2.3.10 Interpolating Learned Positional Embeddings

Learned positional embeddings are tied to a grid size.

If we train at \(96 \times 96\) with patch size \(8\), we have:

\[
G = 12
\]

and:

\[
N = 144
\]

If we later evaluate at \(128 \times 128\), then:

\[
G = 16
\]

and:

\[
N = 256
\]

The learned embedding table no longer has the right size.

A common solution is to interpolate the learned grid.

The learned table:

```python
embedding.weight.shape
# [N, D]
```

can be reshaped to:

```python
[1, D, G_h, G_w]
```

then resized with bicubic interpolation.

Add to `position.py`:

```python
import torch.nn.functional as F


def interpolate_position_embedding(
    pos_embedding: torch.Tensor,
    old_grid_size: tuple[int, int],
    new_grid_size: tuple[int, int],
) -> torch.Tensor:
    """
    Interpolate learned 2D positional embeddings.

    Args:
        pos_embedding:
            Tensor of shape [N_old, D].

        old_grid_size:
            (old_grid_h, old_grid_w).

        new_grid_size:
            (new_grid_h, new_grid_w).

    Returns:
        new_pos_embedding:
            Tensor of shape [N_new, D].
    """
    old_h, old_w = old_grid_size
    new_h, new_w = new_grid_size

    num_old, dim = pos_embedding.shape

    if num_old != old_h * old_w:
        raise ValueError(
            f"old_grid_size does not match pos_embedding length. "
            f"Got num_old={num_old}, old_grid_size={old_grid_size}."
        )

    pos = pos_embedding.reshape(
        old_h,
        old_w,
        dim,
    )

    pos = pos.permute(2, 0, 1).unsqueeze(0)

    pos = F.interpolate(
        pos,
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
    )

    pos = pos.squeeze(0).permute(1, 2, 0)

    return pos.reshape(new_h * new_w, dim)
```

We will not need this for the first minimal run, but it is useful to understand early.

---

## 2.3.11 Unit Tests

Create:

```text
tests/test_position.py
```

Add:

```python
import torch

from jepa_world_model.position import (
    FixedSinCos2DPositionEmbedding,
    Learned2DPositionEmbedding,
    gather_tokens_with_positions,
    interpolate_position_embedding,
    sinusoidal_2d_grid,
)


def test_learned_position_embedding_shape():
    pos = Learned2DPositionEmbedding(
        grid_height=8,
        grid_width=8,
        embed_dim=64,
    )

    indices = torch.tensor([
        [0, 1, 2],
        [4, 5, 6],
    ])

    out = pos(indices)

    assert out.shape == (2, 3, 64)


def test_learned_position_full_grid_shape():
    pos = Learned2DPositionEmbedding(
        grid_height=8,
        grid_width=8,
        embed_dim=64,
    )

    out = pos.full_grid(batch_size=3)

    assert out.shape == (3, 64, 64)


def test_sinusoidal_2d_grid_shape():
    emb = sinusoidal_2d_grid(
        grid_height=8,
        grid_width=8,
        embed_dim=64,
    )

    assert emb.shape == (64, 64)


def test_fixed_sincos_position_embedding_shape():
    pos = FixedSinCos2DPositionEmbedding(
        grid_height=8,
        grid_width=8,
        embed_dim=64,
    )

    indices = torch.tensor([
        [0, 1, 2],
        [4, 5, 6],
    ])

    out = pos(indices)

    assert out.shape == (2, 3, 64)


def test_gather_tokens_with_positions_shape():
    pos = Learned2DPositionEmbedding(
        grid_height=8,
        grid_width=8,
        embed_dim=64,
    )

    tokens = torch.randn(2, 64, 64)

    indices = torch.tensor([
        [0, 2, 4],
        [1, 3, 5],
    ])

    out = gather_tokens_with_positions(
        patch_tokens=tokens,
        indices=indices,
        position_embedding=pos,
    )

    assert out.shape == (2, 3, 64)


def test_interpolate_position_embedding_shape():
    pos = torch.randn(64, 32)

    new_pos = interpolate_position_embedding(
        pos_embedding=pos,
        old_grid_size=(8, 8),
        new_grid_size=(12, 12),
    )

    assert new_pos.shape == (144, 32)
```

Run:

```bash
pytest tests/test_position.py
```

---

## 2.3.12 marimo Visualization

Create or extend:

```text
notebooks/01_visualize_patches_and_masks.py
```

Add a visualization of position embeddings.

Example:

```python
import matplotlib.pyplot as plt
import torch

from jepa_world_model.position import sinusoidal_2d_grid
```

```python
grid_size = 12
embed_dim = 192

pos = sinusoidal_2d_grid(
    grid_height=grid_size,
    grid_width=grid_size,
    embed_dim=embed_dim,
)
```

Visualize one dimension:

```python
dim = 0

values = pos[:, dim].reshape(grid_size, grid_size)

plt.figure(figsize=(5, 5))
plt.imshow(values.cpu())
plt.colorbar()
plt.title(f"Sinusoidal position embedding dimension {dim}")
plt.show()
```

Visualize learned position embeddings after initialization:

```python
from jepa_world_model.position import Learned2DPositionEmbedding

learned = Learned2DPositionEmbedding(
    grid_height=grid_size,
    grid_width=grid_size,
    embed_dim=embed_dim,
)

values = learned.embedding.weight[:, 0].detach().reshape(
    grid_size,
    grid_size,
)

plt.figure(figsize=(5, 5))
plt.imshow(values.cpu())
plt.colorbar()
plt.title("Learned position embedding dimension 0")
plt.show()
```

At initialization, learned embeddings will look random. After training, they may develop spatial structure.

---

## 2.3.13 Common Bugs

### Bug 1: Position index does not match patch order

If patch indexing and position indexing use different conventions, masks and predictions will be spatially wrong.

Always use the same row-major convention:

\[
i = rG + c
\]

---

### Bug 2: Encoder and predictor dimensions differ

The encoder may use:

```python
encoder_dim = 192
```

while the predictor uses:

```python
predictor_dim = 128
```

Their positional embeddings must have different dimensions.

Do not reuse the encoder positional embedding inside the predictor unless the dimensions match.

---

### Bug 3: Positional embeddings added before gather vs after gather

Both of these can be valid:

```python
tokens = patch_tokens + full_pos
selected = gather_patches(tokens, indices)
```

and:

```python
selected = gather_patches(patch_tokens, indices)
selected = selected + pos_embed(indices)
```

But they must use the same indexing convention.

The second version is more explicit and easier to debug.

---

### Bug 4: Forgetting target positions in the predictor

If the predictor receives only:

```python
context_repr
```

but not:

```python
target_indices
```

then it does not know what target locations to predict.

Always condition the predictor on target positions.

---

### Bug 5: Learned positional embeddings and resolution changes

Learned position embeddings are grid-size dependent.

If the image resolution changes, the number of patch positions changes. Interpolate the positional embeddings or use fixed sinusoidal embeddings.

---

## 2.3.14 Summary

Positional embeddings give patch tokens spatial identity.

For JEPA, they are essential because the model must predict representations for specific target locations.

In this section, we implemented:

- learned 2D positional embeddings,
- fixed sinusoidal 2D positional embeddings,
- full-grid position access,
- token gathering with position addition,
- interpolation for learned embeddings,
- unit tests,
- marimo visualization.

The next section implements context and target mask sampling, where these patch positions become the basis for JEPA’s prediction task.

---

## References and Further Reading

- Ashish Vaswani et al., **Attention Is All You Need**, 2017.
  <https://arxiv.org/abs/1706.03762>

- Alexey Dosovitskiy et al., **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**, 2020.
  <https://arxiv.org/abs/2010.11929>

- Kaiming He et al., **Masked Autoencoders Are Scalable Vision Learners**, 2021.
  <https://arxiv.org/abs/2111.06377>

- Mahmoud Assran et al., **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.
  <https://arxiv.org/abs/2301.08243>
