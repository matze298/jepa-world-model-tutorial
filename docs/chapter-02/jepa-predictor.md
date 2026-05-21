# 2.7 JEPA Predictor

The predictor is the module that turns context representations into predicted target representations.

The online encoder produces:

\[
z_{\mathcal{C}}
=
f_\theta(x_{\mathcal{C}}, \mathcal{C})
\]

The target encoder produces:

\[
z_{\mathcal{T}}
=
f_{\bar{\theta}}(x_{\mathcal{T}}, \mathcal{T})
\]

The predictor must learn:

\[
\hat{z}_{\mathcal{T}}
=
g_\theta(z_{\mathcal{C}}, \mathcal{C}, \mathcal{T})
\]

where:

- \(\mathcal{C}\) are context patch indices,
- \(\mathcal{T}\) are target patch indices,
- \(z_{\mathcal{C}}\) are context representations,
- \(\hat{z}_{\mathcal{T}}\) are predicted target representations.

The predictor is trained by the JEPA loss:

\[
\mathcal{L}
=
\left\|
\hat{z}_{\mathcal{T}}
-
\mathrm{sg}(z_{\mathcal{T}})
\right\|^2
\]

This section implements a minimal transformer predictor.

---

## 2.7.1 What the Predictor Receives

The predictor receives:

```python
context_repr.shape
# [B, N_ctx, D]

context_indices.shape
# [B, N_ctx]

target_indices.shape
# [B, N_tgt]
```

and returns:

```python
pred_repr.shape
# [B, N_tgt, D]
```

where:

- \(B\) is batch size,
- \(N_{\mathrm{ctx}}\) is the number of context patches,
- \(N_{\mathrm{tgt}}\) is the number of target patches,
- \(D\) is the encoder representation dimension.

The predictor may use a different internal dimension:

\[
D_p
\]

So internally:

```python
context_repr
# [B, N_ctx, D]

context_tokens
# [B, N_ctx, D_p]

target_queries
# [B, N_tgt, D_p]

predictor_output
# [B, N_tgt, D_p]

pred_repr
# [B, N_tgt, D]
```

The final output is projected back to the encoder dimension so it can be compared with the target encoder output.

---

## 2.7.2 Why the Predictor Needs Target Positions

The predictor should not merely predict “some missing representation”.

It must predict the representation at specific target locations.

For example, given a visible dog head and background, the representation for a target patch below the head may differ from the representation for a target patch above the head.

So the predictor must condition on target positions:

\[
\hat{z}_{\mathcal{T}}
=
g_\theta(z_{\mathcal{C}}, \mathcal{C}, \mathcal{T})
\]

not simply:

\[
\hat{z}_{\mathcal{T}}
=
g_\theta(z_{\mathcal{C}})
\]

In code, target positions enter through learned positional embeddings:

```python
target_queries = target_query + predictor_pos_embed(target_indices)
```

This tells the predictor which target tokens to produce.

---

## 2.7.3 Predictor Data Flow

The minimal predictor performs the following operations:

```text
context representations
  ↓
linear projection to predictor dimension
  ↓
add context positional embeddings

target indices
  ↓
learned target query tokens
  ↓
add target positional embeddings

context tokens + target query tokens
  ↓
transformer encoder
  ↓
select target outputs
  ↓
project back to encoder dimension
  ↓
predicted target representations
```

In code:

```python
context_tokens = context_proj(context_repr)
context_tokens = context_tokens + pos_embed(context_indices)

target_queries = target_query.expand(B, N_tgt, D_p)
target_queries = target_queries + pos_embed(target_indices)

tokens = torch.cat([context_tokens, target_queries], dim=1)
tokens = transformer(tokens)

target_tokens = tokens[:, -N_tgt:]
pred_repr = output_proj(target_tokens)
```

---

## 2.7.4 Why Use a Separate Predictor Dimension?

The encoder representation dimension is \(D\). The predictor internal dimension is \(D_p\).

Usually:

\[
D_p \leq D
\]

For example:

```text
encoder_dim = 192
predictor_dim = 128
```

The predictor is intentionally smaller than the encoder. This creates an asymmetric bottleneck.

The encoder learns general representations. The predictor learns to solve the pretext task. After pretraining, the predictor is usually discarded.

This separation is useful because the predictor can specialize in predicting target representations without forcing the encoder itself to become overly specialized for the prediction head.

---

## 2.7.5 Implementing `predictor.py`

Create:

```text
src/jepa_world_model/predictor.py
```

Start with imports:

```python
from __future__ import annotations

import torch
import torch.nn as nn

from jepa_world_model.position import Learned2DPositionEmbedding
from jepa_world_model.vit import TransformerEncoder
```

Now implement the predictor.

```python
class JEPAPredictor(nn.Module):
    """
    Transformer predictor for minimal I-JEPA.

    It maps context representations and target positions to predicted
    target representations.

    Input:
        context_repr:
            [B, N_ctx, encoder_dim]

        context_indices:
            [B, N_ctx]

        target_indices:
            [B, N_tgt]

    Output:
        pred_repr:
            [B, N_tgt, encoder_dim]
    """

    def __init__(
        self,
        grid_size: int,
        encoder_dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}.")

        if encoder_dim <= 0:
            raise ValueError(f"encoder_dim must be positive, got {encoder_dim}.")

        if predictor_dim <= 0:
            raise ValueError(
                f"predictor_dim must be positive, got {predictor_dim}."
            )

        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim

        self.context_proj = nn.Linear(
            encoder_dim,
            predictor_dim,
        )

        self.pos_embed = Learned2DPositionEmbedding(
            grid_height=grid_size,
            grid_width=grid_size,
            embed_dim=predictor_dim,
        )

        self.target_query = nn.Parameter(
            torch.zeros(1, 1, predictor_dim)
        )

        nn.init.trunc_normal_(
            self.target_query,
            std=0.02,
        )

        self.transformer = TransformerEncoder(
            dim=predictor_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.output_proj = nn.Linear(
            predictor_dim,
            encoder_dim,
        )

    def forward(
        self,
        context_repr: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        if context_repr.dim() != 3:
            raise ValueError(
                f"Expected context_repr [B, N_ctx, D], "
                f"got {context_repr.shape}."
            )

        if context_indices.dim() != 2:
            raise ValueError(
                f"Expected context_indices [B, N_ctx], "
                f"got {context_indices.shape}."
            )

        if target_indices.dim() != 2:
            raise ValueError(
                f"Expected target_indices [B, N_tgt], "
                f"got {target_indices.shape}."
            )

        batch_size, num_context, dim = context_repr.shape

        if dim != self.encoder_dim:
            raise ValueError(
                f"context_repr dim mismatch. Expected {self.encoder_dim}, got {dim}."
            )

        if context_indices.shape != (batch_size, num_context):
            raise ValueError(
                f"context_indices shape must be [B, N_ctx]. "
                f"Expected {(batch_size, num_context)}, "
                f"got {context_indices.shape}."
            )

        if target_indices.size(0) != batch_size:
            raise ValueError(
                f"target_indices batch mismatch. Expected B={batch_size}, "
                f"got B={target_indices.size(0)}."
            )

        num_targets = target_indices.size(1)

        context_tokens = self.context_proj(context_repr)
        context_tokens = context_tokens + self.pos_embed(context_indices)

        target_queries = self.target_query.expand(
            batch_size,
            num_targets,
            -1,
        )

        target_queries = target_queries + self.pos_embed(target_indices)

        tokens = torch.cat(
            [context_tokens, target_queries],
            dim=1,
        )

        tokens = self.transformer(tokens)

        target_tokens = tokens[:, -num_targets:]

        pred_repr = self.output_proj(target_tokens)

        return pred_repr
```

This predictor is now compatible with the encoder implemented in the previous section.

---

## 2.7.6 Shape Example

Suppose:

```python
B = 16
N_ctx = 86
N_tgt = 36
encoder_dim = 192
predictor_dim = 128
```

Then:

```python
context_repr.shape
# [16, 86, 192]

context_indices.shape
# [16, 86]

target_indices.shape
# [16, 36]
```

After projection:

```python
context_tokens.shape
# [16, 86, 128]
```

Target queries:

```python
target_queries.shape
# [16, 36, 128]
```

Concatenated tokens:

```python
tokens.shape
# [16, 122, 128]
```

Predictor transformer output:

```python
tokens.shape
# [16, 122, 128]
```

Target outputs:

```python
target_tokens.shape
# [16, 36, 128]
```

Projected prediction:

```python
pred_repr.shape
# [16, 36, 192]
```

This matches the target encoder representation:

```python
target_repr.shape
# [16, 36, 192]
```

So the loss can compare:

```python
loss = loss_fn(pred_repr, target_repr)
```

---

## 2.7.7 Why Concatenate Context Tokens and Target Queries?

The predictor transformer needs target queries to attend to context tokens.

By concatenating:

```python
[context_tokens, target_queries]
```

and applying self-attention, each target query can attend to all context tokens and to other target queries.

This lets the predictor use:

- context content,
- context positions,
- target positions,
- relationships between target patches.

A target query starts as:

\[
q_i = q_{\mathrm{learned}} + e^{\mathrm{pos}}_i
\]

After transformer attention, it becomes a context-conditioned target prediction.

This is analogous to a decoder-style query, but implemented with a simple transformer encoder over concatenated tokens.

---

## 2.7.8 Should Target Queries Attend to Each Other?

In this minimal predictor, yes.

All tokens attend to all tokens:

```text
context ↔ context
context ↔ target
target ↔ target
```

This allows target predictions to coordinate with each other.

For example, if the target block covers part of an object, neighboring target patches should have compatible representations.

A stricter design could use cross-attention:

```text
target queries attend to context tokens
```

without target-to-target self-attention, or with separate self-attention. That is a valid extension.

For the minimal implementation, concatenated self-attention is simpler and works well enough.

---

## 2.7.9 Predictor Is Trained, Target Encoder Is Not

The predictor parameters are optimized by gradient descent.

The target encoder is not.

The optimizer should include:

```python
list(model.online_encoder.parameters())
+ list(model.predictor.parameters())
```

It should not include:

```python
model.target_encoder.parameters()
```

Correct:

```python
optimizer = torch.optim.AdamW(
    [
        *model.online_encoder.parameters(),
        *model.predictor.parameters(),
    ],
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
)
```

Avoid:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.learning_rate,
)
```

because this includes the target encoder, even though its parameters are frozen. Being explicit avoids mistakes.

---

## 2.7.10 Predictor Initialization

The learned target query is initialized with a small truncated normal:

```python
nn.init.trunc_normal_(self.target_query, std=0.02)
```

The linear layers use PyTorch defaults in this minimal implementation.

If we want a ViT-style initialization later, we can add:

```python
def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
```

and apply it to the predictor.

For Chapter 2, PyTorch defaults are sufficient.

---

## 2.7.11 Predictor Complexity

The predictor processes:

\[
N_{\mathrm{ctx}} + N_{\mathrm{tgt}}
\]

tokens.

Attention cost is roughly:

\[
O((N_{\mathrm{ctx}} + N_{\mathrm{tgt}})^2)
\]

For the example:

```text
N_ctx = 86
N_tgt = 36
total = 122
```

This is manageable.

The predictor is usually smaller than the encoder:

```text
encoder_dim = 192
predictor_dim = 128
encoder_depth = 6
predictor_depth = 3
```

This keeps the predictor relatively cheap.

---

## 2.7.12 Unit Tests

Create:

```text
tests/test_predictor.py
```

Add:

```python
import torch

from jepa_world_model.predictor import JEPAPredictor


def test_predictor_shape():
    predictor = JEPAPredictor(
        grid_size=8,
        encoder_dim=64,
        predictor_dim=32,
        depth=2,
        num_heads=4,
    )

    context_repr = torch.randn(2, 20, 64)

    context_indices = torch.randint(
        low=0,
        high=64,
        size=(2, 20),
    )

    target_indices = torch.randint(
        low=0,
        high=64,
        size=(2, 8),
    )

    pred = predictor(
        context_repr=context_repr,
        context_indices=context_indices,
        target_indices=target_indices,
    )

    assert pred.shape == (2, 8, 64)


def test_predictor_rejects_bad_context_dim():
    predictor = JEPAPredictor(
        grid_size=8,
        encoder_dim=64,
        predictor_dim=32,
        depth=2,
        num_heads=4,
    )

    context_repr = torch.randn(2, 20, 32)

    context_indices = torch.randint(
        low=0,
        high=64,
        size=(2, 20),
    )

    target_indices = torch.randint(
        low=0,
        high=64,
        size=(2, 8),
    )

    try:
        predictor(
            context_repr=context_repr,
            context_indices=context_indices,
            target_indices=target_indices,
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError for context dimension mismatch.")


def test_predictor_rejects_context_index_shape_mismatch():
    predictor = JEPAPredictor(
        grid_size=8,
        encoder_dim=64,
        predictor_dim=32,
        depth=2,
        num_heads=4,
    )

    context_repr = torch.randn(2, 20, 64)

    context_indices = torch.randint(
        low=0,
        high=64,
        size=(2, 19),
    )

    target_indices = torch.randint(
        low=0,
        high=64,
        size=(2, 8),
    )

    try:
        predictor(
            context_repr=context_repr,
            context_indices=context_indices,
            target_indices=target_indices,
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError for context index mismatch.")


def test_predictor_receives_gradients():
    predictor = JEPAPredictor(
        grid_size=8,
        encoder_dim=64,
        predictor_dim=32,
        depth=2,
        num_heads=4,
    )

    context_repr = torch.randn(
        2,
        20,
        64,
        requires_grad=True,
    )

    context_indices = torch.randint(
        low=0,
        high=64,
        size=(2, 20),
    )

    target_indices = torch.randint(
        low=0,
        high=64,
        size=(2, 8),
    )

    pred = predictor(
        context_repr=context_repr,
        context_indices=context_indices,
        target_indices=target_indices,
    )

    loss = pred.pow(2).mean()
    loss.backward()

    has_grad = any(
        p.grad is not None
        for p in predictor.parameters()
    )

    assert has_grad
    assert context_repr.grad is not None
```

Run:

```bash
uv run pytest tests/test_predictor.py
```

---

## 2.7.13 Integration Test with Encoder and Masks

Add an integration test:

```python
from jepa_world_model.masks import BlockMaskConfig, sample_mask_batch
from jepa_world_model.vit import MinimalViTEncoder


def test_predictor_with_encoder_and_masks():
    encoder = MinimalViTEncoder(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )

    predictor = JEPAPredictor(
        grid_size=8,
        encoder_dim=64,
        predictor_dim=32,
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

    pred_repr = predictor(
        context_repr=context_repr,
        context_indices=context_indices,
        target_indices=target_indices,
    )

    assert pred_repr.shape == (
        2,
        mask_config.num_target_patches,
        64,
    )
```

This verifies that:

```text
mask sampler → encoder → predictor
```

works as a chain.

---

## 2.7.14 marimo Debug Notebook

Create:

```text
notebooks/04_debug_predictor.py
```

Open:

```bash
uv run marimo edit notebooks/04_debug_predictor.py
```

Example cells:

```python
import torch

from jepa_world_model.masks import BlockMaskConfig, sample_mask_batch
from jepa_world_model.predictor import JEPAPredictor
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

predictor = JEPAPredictor(
    grid_size=8,
    encoder_dim=64,
    predictor_dim=32,
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
context_repr = encoder(
    images=images,
    patch_indices=context_indices,
)

pred_repr = predictor(
    context_repr=context_repr,
    context_indices=context_indices,
    target_indices=target_indices,
)

context_repr.shape, pred_repr.shape
```

```python
pred_repr.mean().item(), pred_repr.std().item()
```

This verifies that the predictor runs end-to-end on the chosen device.

---

## 2.7.15 Common Bugs

### Bug 1: Forgetting target positional embeddings

If target queries do not receive target positional embeddings, the predictor does not know which target locations to predict.

Correct:

```python
target_queries = target_queries + self.pos_embed(target_indices)
```

---

### Bug 2: Reusing encoder positional embeddings with wrong dimension

The encoder uses `encoder_dim`.

The predictor uses `predictor_dim`.

If these differ, the predictor needs its own positional embedding.

---

### Bug 3: Returning all predictor tokens

The predictor transformer returns context and target outputs.

We only want the target outputs:

```python
target_tokens = tokens[:, -num_targets:]
```

Then:

```python
pred_repr = output_proj(target_tokens)
```

Do not compare all tokens to the target encoder output.

---

### Bug 4: Shape mismatch between prediction and target

The predictor output must be:

```python
[B, N_tgt, encoder_dim]
```

The target encoder output is:

```python
[B, N_tgt, encoder_dim]
```

The loss requires exact shape equality.

---

### Bug 5: Predictor too strong or too weak

If the predictor is too small, it may fail to predict useful target representations.

If it is too large, it may absorb too much of the task and reduce pressure on the encoder.

For the minimal implementation, a reasonable starting point is:

```text
encoder_dim = 192
predictor_dim = 128
encoder_depth = 6
predictor_depth = 3
```

---

## 2.7.16 Summary

The JEPA predictor maps context representations and target positions into predicted target representations.

It uses:

- a projection from encoder dimension to predictor dimension,
- context positional embeddings,
- learned target query tokens,
- target positional embeddings,
- a transformer stack,
- an output projection back to encoder dimension.

The predictor interface is:

```python
pred_repr = predictor(
    context_repr=context_repr,
    context_indices=context_indices,
    target_indices=target_indices,
)
```

with:

```python
pred_repr.shape
# [B, N_tgt, encoder_dim]
```

The next section implements losses and diagnostics so that we can compare predicted target representations to target encoder representations and monitor representation health.

---

## References and Further Reading

- Mahmoud Assran et al., **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.
  <https://arxiv.org/abs/2301.08243>

- Facebook Research, **Official I-JEPA Codebase**.
  <https://github.com/facebookresearch/ijepa>

- Ashish Vaswani et al., **Attention Is All You Need**, 2017.
  <https://arxiv.org/abs/1706.03762>

- Alexey Dosovitskiy et al., **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**, 2020.
  <https://arxiv.org/abs/2010.11929>
