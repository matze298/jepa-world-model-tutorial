# 2.5 Minimal ViT Encoder

We now have three ingredients:

1. image patchification,
2. positional embeddings,
3. context and target mask sampling.

The next step is the encoder.

In I-JEPA, the encoder maps selected image patches into latent representations. The same encoder architecture is used for both:

- the **online encoder**, which processes context patches and receives gradients,
- the **target encoder**, which processes target patches and is updated by EMA.

The encoder is a Vision Transformer-style module.

Its job is to compute:

\[
z_{\mathcal{I}}
=
f_\theta(x_{\mathcal{I}}, \mathcal{I})
\]

where:

- \(\mathcal{I}\) is a set of patch indices,
- \(x_{\mathcal{I}}\) are selected image patches,
- \(z_{\mathcal{I}}\) are output representations for those patches.

For the online branch:

\[
z_{\mathcal{C}}
=
f_\theta(x_{\mathcal{C}}, \mathcal{C})
\]

For the target branch:

\[
z_{\mathcal{T}}
=
f_{\bar{\theta}}(x_{\mathcal{T}}, \mathcal{T})
\]

The encoder should accept an image and a set of patch indices, then return one representation per selected patch.

---

## 2.5.1 Encoder Input and Output

The encoder receives:

```python
images.shape
# [B, C, H, W]

patch_indices.shape
# [B, K]
```

and returns:

```python
repr.shape
# [B, K, D]
```

where:

- \(B\) is batch size,
- \(C\) is number of image channels,
- \(H, W\) are image height and width,
- \(K\) is number of selected patches,
- \(D\) is encoder embedding dimension.

For context patches:

```python
context_repr = encoder(
    images=images,
    patch_indices=context_indices,
)

context_repr.shape
# [B, N_ctx, D]
```

For target patches:

```python
target_repr = encoder(
    images=images,
    patch_indices=target_indices,
)

target_repr.shape
# [B, N_tgt, D]
```

This shared interface is important. It lets us instantiate two copies:

```python
online_encoder = MinimalViTEncoder(...)
target_encoder = MinimalViTEncoder(...)
```

and use them identically.

---

## 2.5.2 Encoder Data Flow

The encoder performs four operations:

```text
image
  ↓
patch embedding
  ↓
gather selected patch tokens
  ↓
add selected positional embeddings
  ↓
transformer encoder
  ↓
patch representations
```

In code:

```python
patch_tokens = patch_embed(images)
selected_tokens = gather_patches(patch_tokens, patch_indices)
selected_pos = position_embedding(patch_indices)

tokens = selected_tokens + selected_pos
tokens = transformer(tokens)
tokens = norm(tokens)
```

The output is a representation for each selected patch.

---

## 2.5.3 Why the Encoder Processes Only Selected Patches

A common ViT encoder processes all patches:

\[
N = G^2
\]

For JEPA, the online encoder should only process context patches. If the online encoder processes target patches too, it may leak target information into the context representation.

So for the online branch, we do:

```python
context_tokens = gather_patches(
    patch_tokens,
    context_indices,
)
```

not:

```python
all_tokens = patch_tokens
```

The target encoder processes target patches, but that branch is stop-gradient-ed and used only to produce target representations.

This separation is central to JEPA.

---

## 2.5.4 Minimal Transformer Block

We will implement a small transformer encoder block.

A transformer block contains:

1. LayerNorm,
2. multi-head self-attention,
3. residual connection,
4. LayerNorm,
5. MLP,
6. residual connection.

Mathematically:

\[
x' = x + \mathrm{MSA}(\mathrm{LN}(x))
\]

\[
y = x' + \mathrm{MLP}(\mathrm{LN}(x'))
\]

This is the pre-norm transformer block.

Create or update:

```text
src/jepa_world_model/vit.py
```

Start with:

```python
from __future__ import annotations

import torch
import torch.nn as nn

from jepa_world_model.patchify import PatchEmbed, gather_patches
from jepa_world_model.position import Learned2DPositionEmbedding
```

Add the MLP:

```python
class MLP(nn.Module):
    """
    Transformer feed-forward network.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

Add the transformer block:

```python
class TransformerBlock(nn.Module):
    """
    Minimal pre-norm transformer encoder block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim must be divisible by num_heads. "
                f"Got dim={dim}, num_heads={num_heads}."
            )

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)

        attn_out, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=False,
        )

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))

        return x
```

This block uses PyTorch’s built-in `nn.MultiheadAttention`.

It is not the fastest possible implementation, but it is explicit and easy to debug.

---

## 2.5.5 Transformer Encoder Stack

Now define a stack of transformer blocks.

```python
class TransformerEncoder(nn.Module):
    """
    Stack of transformer encoder blocks.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}.")

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        return self.norm(x)
```

This stack maps:

```python
x.shape
# [B, K, D]
```

to:

```python
out.shape
# [B, K, D]
```

---

## 2.5.6 Minimal ViT Encoder

Now combine:

- patch embedding,
- positional embedding,
- selected-token gathering,
- transformer stack.

```python
class MinimalViTEncoder(nn.Module):
    """
    Minimal Vision Transformer encoder for JEPA.

    The encoder processes selected patch indices only.

    Input:
        images:
            [B, C, H, W]

        patch_indices:
            [B, K]

    Output:
        representations:
            [B, K, D]
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
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

        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.pos_embed = Learned2DPositionEmbedding(
            grid_height=self.grid_size,
            grid_width=self.grid_size,
            embed_dim=embed_dim,
        )

        self.encoder = TransformerEncoder(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(
        self,
        images: torch.Tensor,
        patch_indices: torch.Tensor,
    ) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError(
                f"Expected images [B, C, H, W], got {images.shape}."
            )

        if patch_indices.dim() != 2:
            raise ValueError(
                f"Expected patch_indices [B, K], got {patch_indices.shape}."
            )

        if images.size(0) != patch_indices.size(0):
            raise ValueError(
                f"Batch mismatch: images B={images.size(0)}, "
                f"patch_indices B={patch_indices.size(0)}."
            )

        patch_tokens = self.patch_embed(images)

        selected_tokens = gather_patches(
            patch_tokens,
            patch_indices,
        )

        selected_pos = self.pos_embed(patch_indices)

        tokens = selected_tokens + selected_pos

        representations = self.encoder(tokens)

        return representations
```

This is the core encoder used by the minimal I-JEPA model.

---

## 2.5.7 Encoder Shape Example

For STL-10-style images:

```python
images.shape
# [16, 3, 96, 96]
```

Config:

```python
image_size = 96
patch_size = 8
embed_dim = 192
```

Then:

```python
grid_size = 12
num_patches = 144
```

If the context mask contains 86 patches:

```python
context_indices.shape
# [16, 86]
```

then:

```python
context_repr = encoder(images, context_indices)

context_repr.shape
# [16, 86, 192]
```

If the target mask contains 36 patches:

```python
target_indices.shape
# [16, 36]
```

then:

```python
target_repr = encoder(images, target_indices)

target_repr.shape
# [16, 36, 192]
```

The predictor will later map:

```python
context_repr
# [B, N_ctx, D]
```

to:

```python
pred_target_repr
# [B, N_tgt, D]
```

---

## 2.5.8 No Class Token

Standard ViT classifiers often prepend a `[CLS]` token.

For this minimal JEPA encoder, we do **not** use a class token.

Why?

The encoder needs to produce patch-level target representations:

\[
z_{\mathcal{T}} \in \mathbb{R}^{B \times N_{\mathrm{tgt}} \times D}
\]

A class token would produce a global representation, but JEPA’s prediction target is patch-wise.

So the output remains one representation per selected patch.

Later, for evaluation, we can create a global image representation by pooling patch tokens:

```python
global_repr = patch_repr.mean(dim=1)
```

This is useful for k-NN or linear probing.

---

## 2.5.9 Online Encoder and Target Encoder

The online encoder and target encoder have the same architecture.

```python
online_encoder = MinimalViTEncoder(...)
target_encoder = MinimalViTEncoder(...)
```

At initialization, their weights should match:

```python
target_encoder.load_state_dict(
    online_encoder.state_dict()
)
```

Then the target encoder is frozen:

```python
for p in target_encoder.parameters():
    p.requires_grad = False
```

During training:

- online encoder receives gradients,
- target encoder does not,
- target encoder is updated by EMA.

This means the encoder class itself does not need to know whether it is online or target. That role is handled by the top-level JEPA model.

---

## 2.5.10 Attention Complexity

The encoder processes only selected patches.

If the context has \(N_{\mathrm{ctx}}\) patches, attention cost is roughly:

\[
O(N_{\mathrm{ctx}}^2)
\]

For the target encoder, cost is:

\[
O(N_{\mathrm{tgt}}^2)
\]

This is one reason patch selection matters computationally.

For example, if the full image has 144 patches but the context has 86 patches, the online encoder attention cost is reduced compared with processing all patches:

\[
86^2 < 144^2
\]

The minimal implementation does not optimize attention further. Later, research-grade engineering can add:

- fused attention,
- `torch.compile`,
- FlashAttention-compatible blocks,
- mixed precision,
- distributed training.

For now, correctness is the priority.

---

## 2.5.11 Encoder Initialization

The current implementation relies mostly on PyTorch defaults, plus the position embedding initialization.

For a minimal implementation, this is acceptable.

If we want more ViT-like initialization, we can add:

```python
def init_transformer_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(
            module.weight,
            std=0.02,
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
```

Then inside `MinimalViTEncoder.__init__`:

```python
self.apply(init_transformer_weights)
```

But be careful: this will also reinitialize submodules. If using custom initialization, apply it consistently to the encoder and predictor.

For Chapter 2, using PyTorch defaults is fine unless training becomes unstable.

---

## 2.5.12 Unit Tests

Create:

```text
tests/test_vit.py
```

Add:

```python
import torch

from jepa_world_model.vit import (
    MLP,
    MinimalViTEncoder,
    TransformerBlock,
    TransformerEncoder,
)


def test_mlp_shape():
    mlp = MLP(
        dim=64,
        hidden_dim=256,
    )

    x = torch.randn(2, 10, 64)
    y = mlp(x)

    assert y.shape == x.shape


def test_transformer_block_shape():
    block = TransformerBlock(
        dim=64,
        num_heads=4,
        mlp_ratio=4.0,
    )

    x = torch.randn(2, 10, 64)
    y = block(x)

    assert y.shape == x.shape


def test_transformer_encoder_shape():
    encoder = TransformerEncoder(
        dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
    )

    x = torch.randn(2, 10, 64)
    y = encoder(x)

    assert y.shape == x.shape


def test_minimal_vit_encoder_context_shape():
    encoder = MinimalViTEncoder(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )

    images = torch.randn(2, 3, 32, 32)

    patch_indices = torch.tensor([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ])

    out = encoder(
        images=images,
        patch_indices=patch_indices,
    )

    assert out.shape == (2, 4, 64)


def test_minimal_vit_encoder_target_shape():
    encoder = MinimalViTEncoder(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )

    images = torch.randn(2, 3, 32, 32)

    patch_indices = torch.tensor([
        [10, 11, 12],
        [20, 21, 22],
    ])

    out = encoder(
        images=images,
        patch_indices=patch_indices,
    )

    assert out.shape == (2, 3, 64)


def test_minimal_vit_encoder_rejects_bad_batch_size():
    encoder = MinimalViTEncoder(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )

    images = torch.randn(2, 3, 32, 32)
    patch_indices = torch.tensor([[0, 1, 2]])

    try:
        encoder(images, patch_indices)
    except ValueError:
        return

    raise AssertionError("Expected ValueError for batch mismatch.")
```

Run:

```bash
uv run pytest tests/test_vit.py
```

---

## 2.5.13 Integration Test with Mask Sampler

Add one integration test to make sure masks and encoder work together.

```python
from jepa_world_model.masks import BlockMaskConfig, sample_mask_batch


def test_encoder_with_sampled_masks():
    encoder = MinimalViTEncoder(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )

    images = torch.randn(2, 3, 32, 32)

    mask_config = BlockMaskConfig(
        grid_height=8,
        grid_width=8,
        num_target_blocks=2,
        target_block_height=2,
        target_block_width=2,
        context_ratio=0.5,
    )

    context_indices, target_indices = sample_mask_batch(
        config=mask_config,
        batch_size=2,
        device=torch.device("cpu"),
    )

    context_repr = encoder(
        images=images,
        patch_indices=context_indices,
    )

    target_repr = encoder(
        images=images,
        patch_indices=target_indices,
    )

    assert context_repr.shape == (
        2,
        mask_config.num_context_patches,
        64,
    )

    assert target_repr.shape == (
        2,
        mask_config.num_target_patches,
        64,
    )
```

This test confirms that the encoder can consume masks sampled by the mask module.

---

## 2.5.14 marimo Debug Notebook

Create or extend:

```text
notebooks/02_debug_encoder.py
```

Open:

```bash
uv run marimo edit notebooks/02_debug_encoder.py
```

Example cells:

```python
import torch

from jepa_world_model.masks import BlockMaskConfig, sample_mask_batch
from jepa_world_model.vit import MinimalViTEncoder
```

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = MinimalViTEncoder(
    image_size=32,
    patch_size=4,
    in_channels=3,
    embed_dim=64,
    depth=2,
    num_heads=4,
).to(device)
```

```python
images = torch.randn(4, 3, 32, 32, device=device)

mask_config = BlockMaskConfig(
    grid_height=8,
    grid_width=8,
    num_target_blocks=2,
    target_block_height=2,
    target_block_width=2,
    context_ratio=0.5,
)

context_indices, target_indices = sample_mask_batch(
    config=mask_config,
    batch_size=images.size(0),
    device=device,
)
```

```python
context_repr = encoder(images, context_indices)
target_repr = encoder(images, target_indices)

context_repr.shape, target_repr.shape
```

```python
context_repr.mean().item(), context_repr.std().item()
```

This notebook helps verify that the encoder runs on the intended device and produces nontrivial representations.

---

## 2.5.15 Gradient Check

The encoder must be differentiable.

A simple check:

```python
images = torch.randn(
    2,
    3,
    32,
    32,
    requires_grad=True,
)

indices = torch.tensor([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
])

out = encoder(images, indices)
loss = out.pow(2).mean()
loss.backward()

assert images.grad is not None
```

For the online encoder, parameters should receive gradients:

```python
has_grad = any(
    p.grad is not None
    for p in encoder.parameters()
)

assert has_grad
```

The target encoder will be frozen at the model level, not inside `MinimalViTEncoder`.

---

## 2.5.16 Common Bugs

### Bug 1: Adding positional embeddings before selecting patches but using wrong indices

Both approaches can work:

```python
tokens = patch_tokens + full_pos
selected = gather_patches(tokens, indices)
```

and:

```python
selected = gather_patches(patch_tokens, indices)
selected = selected + pos_embed(indices)
```

The second is more explicit and easier to debug.

---

### Bug 2: Forgetting that target and context lengths differ

The encoder can return different token counts depending on `patch_indices`.

This is expected:

```python
context_repr.shape
# [B, N_ctx, D]

target_repr.shape
# [B, N_tgt, D]
```

The predictor will handle this difference.

---

### Bug 3: Using a class token

A class token is not needed for patch-level JEPA targets.

Do not add one in the minimal encoder.

---

### Bug 4: Accidentally processing all patches in the online encoder

The online encoder must receive only context patches.

Do not use:

```python
encoder(images, all_indices)
```

for the online branch during JEPA pretraining.

Use:

```python
encoder(images, context_indices)
```

---

### Bug 5: Device mismatch for indices

Patch indices must live on the same device as patch tokens.

During training:

```python
context_indices, target_indices = sample_mask_batch(
    config=mask_config,
    batch_size=images.size(0),
    device=images.device,
)
```

---

## 2.5.17 Summary

This section implemented the minimal ViT encoder used by both the online and target branches.

The encoder:

1. patchifies the image,
2. embeds patches,
3. gathers selected patch tokens,
4. adds selected positional embeddings,
5. processes tokens with transformer blocks,
6. returns one representation per selected patch.

The encoder interface is:

```python
repr = encoder(
    images=images,
    patch_indices=indices,
)
```

with:

```python
repr.shape
# [B, K, D]
```

The next section implements the EMA target encoder mechanism: how the target encoder is initialized, frozen, and updated from the online encoder.

---

## References and Further Reading

- Ashish Vaswani et al., **Attention Is All You Need**, 2017.  
  <https://arxiv.org/abs/1706.03762>

- Alexey Dosovitskiy et al., **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**, 2020.  
  <https://arxiv.org/abs/2010.11929>

- Mahmoud Assran et al., **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.  
  <https://arxiv.org/abs/2301.08243>

- PyTorch, **MultiheadAttention**.  
  <https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>