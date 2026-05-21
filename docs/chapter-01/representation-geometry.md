# 1.4 Representation Geometry

JEPA is not only an architecture. It is also a way of shaping representation space.

The objective:

\[
\mathcal{L}_{\mathrm{JEPA}}
=
\left\|
g_\theta(f_\theta(x_{\mathrm{ctx}}), m_{\mathrm{tgt}})
-
\mathrm{sg}(f_{\bar{\theta}}(x_{\mathrm{tgt}}))
\right\|^2
\]

does more than train a predictor. It defines a geometry over latent variables.

If two target regions are semantically similar, we would like their representations to be close. If they differ in important ways, we would like their representations to be distinguishable. If two observations differ only in nuisance details, we may want their representations to be approximately invariant. If they differ by spatial position, pose, time, or action consequence, we may want the representation to change in a structured way.

This section studies the geometry that JEPA-style objectives try to create.

The goal is to understand what it means for a representation to be useful, what can go wrong, and what diagnostics we should implement from the beginning.

---

## 1.4.1 What Is Representation Geometry?

A representation is a mapping:

\[
f_\theta : \mathcal{X} \rightarrow \mathcal{Z}
\]

from observation space \(\mathcal{X}\) to latent space \(\mathcal{Z}\).

For images:

\[
x \in \mathbb{R}^{3 \times H \times W}
\]

and:

\[
z = f_\theta(x) \in \mathbb{R}^{D}
\]

or, for patch-level encoders:

\[
z = f_\theta(x) \in \mathbb{R}^{N \times D}
\]

where \(N\) is the number of patches and \(D\) is the embedding dimension.

The geometry of \(\mathcal{Z}\) is determined by distances, angles, norms, covariance structure, neighborhoods, and directions. When we say that a representation is useful, we are usually making claims about this geometry.

For example:

- similar images should be close,
- different semantic classes should be separable,
- nuisance variation should be suppressed,
- meaningful factors should correspond to recoverable directions,
- future states should be predictable from past states,
- actions should induce structured movement in latent space.

A representation is not useful merely because it compresses the input. It is useful because the geometry of the latent space makes downstream prediction, classification, retrieval, planning, or control easier.

---

## 1.4.2 Similarity and Distance

Most self-supervised objectives impose some form of similarity constraint.

In contrastive learning:

\[
\mathrm{sim}(f(v_1), f(v_2)) \uparrow
\]

for positive pairs, and:

\[
\mathrm{sim}(f(v_i), f(v_j)) \downarrow
\]

for negatives.

In BYOL-style learning:

\[
q_\theta(f_\theta(v_1))
\approx
f_{\bar{\theta}}(v_2)
\]

for two views of the same input.

In JEPA:

\[
g_\theta(f_\theta(x_{\mathrm{ctx}}), m_{\mathrm{tgt}})
\approx
f_{\bar{\theta}}(x_{\mathrm{tgt}})
\]

for context-target pairs from the same observation.

This makes the choice of distance function important.

The most common candidates are:

### Euclidean distance

\[
d(z_1,z_2)
=
\|z_1 - z_2\|_2
\]

### Squared Euclidean distance

\[
d(z_1,z_2)
=
\|z_1 - z_2\|_2^2
\]

### Cosine distance

\[
d(z_1,z_2)
=
1
-
\frac{z_1^\top z_2}{\|z_1\|\|z_2\|}
\]

In code:

```python
import torch
import torch.nn.functional as F


def pairwise_metrics(
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    z1, z2:
        [batch, dim] or [batch, tokens, dim]
    """
    if z1.dim() == 3:
        z1 = z1.flatten(0, 1)
        z2 = z2.flatten(0, 1)

    mse = F.mse_loss(z1, z2)

    cosine = F.cosine_similarity(
        z1.float(),
        z2.float(),
        dim=-1,
    ).mean()

    euclidean = (z1 - z2).norm(dim=-1).mean()

    return {
        "mse": mse,
        "cosine": cosine,
        "euclidean": euclidean,
    }
```

Different metrics emphasize different geometry.

MSE preserves scale. Cosine similarity ignores scale and compares direction. Smooth L1 is less sensitive to outliers. A good implementation should make this choice explicit rather than burying it inside the training loop.

---

## 1.4.3 Invariance

An invariant representation ignores transformations that should not matter.

Let \(T\) be a transformation of the input. A representation is invariant to \(T\) if:

\[
f(Tx) \approx f(x)
\]

For images, useful invariances may include:

- small crops,
- color jitter,
- mild blur,
- lighting changes,
- background texture variation.

For cycling telemetry, useful invariances may include:

- sensor noise,
- small GPS fluctuations,
- harmless cadence jitter,
- small timestamp misalignment.

Invariance is one of the central goals of self-supervised learning.

Contrastive and BYOL-style methods often induce invariance by applying augmentations. If two augmented views are forced to have similar representations, then the representation becomes invariant to those augmentations.

JEPA induces a different kind of invariance. It does not merely align two global augmented views. It asks the representation of a target to be predictable from context. This creates pressure to represent what is predictable and stable across partial observations.

If a target feature is arbitrary noise, it is difficult to predict from context. If a target feature reflects object structure, scene layout, or physiological state, it is more predictable.

This suggests an intuitive principle:

> JEPA encourages representations of predictable structure and discourages representations of unpredictable nuisance detail.

This is not guaranteed automatically, but it is the intended bias.

A simple invariance diagnostic can measure how much representations change under augmentations:

```python
@torch.no_grad()
def invariance_score(
    encoder,
    x: torch.Tensor,
    augment,
) -> dict[str, float]:
    """
    Measures representation similarity under augmentation.
    """
    view_1 = augment(x)
    view_2 = augment(x)

    z1 = encoder(view_1)
    z2 = encoder(view_2)

    if z1.dim() == 3:
        z1 = z1.mean(dim=1)
        z2 = z2.mean(dim=1)

    cosine = F.cosine_similarity(
        z1.float(),
        z2.float(),
        dim=-1,
    ).mean()

    mse = F.mse_loss(z1.float(), z2.float())

    return {
        "invariance/cosine": cosine.item(),
        "invariance/mse": mse.item(),
    }
```

This diagnostic is not a full evaluation. It only tells us how stable the representation is under a chosen transformation. But it is useful for catching gross instability.

---

## 1.4.4 Equivariance

Not all transformations should be ignored.

Some transformations should produce structured changes in representation space.

A representation is equivariant to a transformation \(T\) if there exists a corresponding transformation \(A_T\) in latent space such that:

\[
f(Tx) \approx A_T f(x)
\]

For example:

- if an object moves right in an image, its spatial representation should move right,
- if time advances in a video, the latent state should evolve,
- if a cyclist increases power, the latent physiological state should change in a predictable direction.

Invariance says:

\[
f(Tx) \approx f(x)
\]

Equivariance says:

\[
f(Tx) \approx A_T f(x)
\]

World models need equivariance.

If actions do not change the representation in a structured way, planning becomes difficult. A latent dynamics model needs actions to induce predictable latent transitions:

\[
z_{t+1}
\approx
F(z_t, a_t)
\]

For image JEPA, spatial information is especially important. We do not want every target patch representation to be completely position-invariant. The model must know where a target is in order to predict its representation.

This is why positional embeddings and target indices matter.

A predictor that receives target positions can learn:

\[
\hat{z}_{\mathcal{T}}
=
g(z_{\mathcal{C}}, \mathcal{T})
\]

rather than:

\[
\hat{z}_{\mathcal{T}}
=
g(z_{\mathcal{C}})
\]

The target position \(\mathcal{T}\) tells the predictor which latent structure to infer.

In code:

```python
target_queries = target_query + position_embedding(target_indices)
```

This line is small, but conceptually important. It gives the predictor spatial conditioning.

---

## 1.4.5 Latent Manifolds

A common intuition is that natural data lies near a lower-dimensional manifold.

Images are high-dimensional. A \(224 \times 224\) RGB image has:

\[
3 \times 224 \times 224 = 150{,}528
\]

pixel values.

But natural images do not fill this space uniformly. Most random pixel arrays are not meaningful images. Real images occupy a structured subset.

A representation model attempts to map observations onto a latent manifold where useful factors are easier to model.

\[
x \in \mathcal{X}
\quad \rightarrow \quad
z \in \mathcal{Z}
\]

For world modeling, we want a latent manifold where temporal transitions are simpler than observation-space transitions.

Instead of:

\[
x_{t+1} = G(x_t, a_t)
\]

we want:

\[
z_{t+1} = F(z_t, a_t)
\]

where \(F\) is smoother, lower-dimensional, and more predictable.

For cycling, the raw telemetry may include noisy measurements, but the latent state may evolve smoothly:

```text
fresh → warming up → steady effort → accumulating fatigue → recovery
```

A good latent space should make such trajectories coherent.

A simple way to inspect this is to embed representations with PCA or UMAP.

```python
@torch.no_grad()
def collect_representations(
    encoder,
    dataloader,
    device: torch.device,
    max_batches: int = 20,
) -> torch.Tensor:
    reps = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        x = batch[0].to(device)
        z = encoder(x)

        if z.dim() == 3:
            z = z.mean(dim=1)

        reps.append(z.cpu())

    return torch.cat(reps, dim=0)
```

Then one can visualize with PCA:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_pca(z: torch.Tensor, labels=None) -> None:
    z_np = z.numpy()

    coords = PCA(n_components=2).fit_transform(z_np)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        s=8,
        alpha=0.7,
    )
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Representation PCA")
    plt.show()
```

These plots are only qualitative. But they help reveal whether representations cluster, collapse, or organize according to meaningful factors.

---

## 1.4.6 Predictable Information as an Inductive Bias

JEPA does not reconstruct the input. It predicts target representations from context.

This creates an important inductive bias:

> The representation should encode information that is predictable from other parts of the input.

Suppose the target contains a random noise pattern independent of the context. A predictor cannot infer that noise from the context. If the target encoder encodes the noise strongly, the prediction loss will be difficult to minimize.

Suppose instead the target contains part of an object whose structure is implied by the visible context. The predictor can learn to infer that structure.

So the learning system is biased toward encoding predictable regularities.

This can be described informally as:

\[
z_{\mathrm{tgt}}
\approx
\text{predictable structure of } x_{\mathrm{tgt}}
\]

rather than:

\[
z_{\mathrm{tgt}}
\approx
\text{all details of } x_{\mathrm{tgt}}
\]

This is one reason representation prediction may produce more semantic features than pixel reconstruction.

The same principle applies temporally.

A future sensor glitch is not predictable from past physiological state. A general rise in heart rate under sustained power is predictable. A good latent dynamics model should focus more on the second than the first.

---

## 1.4.7 The Information Bottleneck View

The information bottleneck perspective asks for a representation \(z\) that preserves useful information while discarding irrelevant information.

Classically, one might want to maximize information about a target \(y\) while minimizing information about the input \(x\):

\[
\max I(z; y) - \beta I(z; x)
\]

In self-supervised learning, there may be no label \(y\). Instead, the target is another view, region, or future state.

For JEPA, the target is:

\[
z_{\mathrm{tgt}} = f_{\bar{\theta}}(x_{\mathrm{tgt}})
\]

The context representation should contain enough information to predict the target representation:

\[
I(z_{\mathrm{ctx}}; z_{\mathrm{tgt}})
\]

but not necessarily all information about the raw context.

The bottleneck is implicit rather than explicit. It comes from:

- partial context,
- target abstraction,
- finite embedding dimension,
- predictor architecture,
- masking strategy,
- optimization dynamics.

A simplified intuition is:

```text
Context contains many details.
Target contains many details.
Only some target details are predictable from context.
JEPA rewards the predictable part.
```

This is why mask design and target representation quality matter. If the task is too easy, the bottleneck does not force abstraction. If the task is too hard, the model may learn weak averages. If the target representation is poor, the model predicts poor features.

---

## 1.4.8 Collapse

Collapse is the most dangerous geometric failure mode.

A representation collapses if many inputs map to the same or nearly the same vector:

\[
f(x) \approx c
\]

for all \(x\).

A fully collapsed representation has zero variance across samples:

\[
\mathrm{Var}(f(x)) = 0
\]

In JEPA, collapse is especially dangerous because the loss can become small.

If:

\[
f_\theta(x_{\mathrm{ctx}}) = c
\]

and:

\[
f_{\bar{\theta}}(x_{\mathrm{tgt}}) = c
\]

then the predictor can output \(c\), and:

\[
\mathcal{L}_{\mathrm{JEPA}} \approx 0
\]

But the representation is useless.

This means the training loss alone cannot be trusted.

We need geometry diagnostics.

A basic collapse diagnostic computes variance per dimension:

```python
@torch.no_grad()
def feature_variance(
    z: torch.Tensor,
) -> torch.Tensor:
    """
    z:
        [batch, dim] or [batch, tokens, dim]

    returns:
        [dim]
    """
    if z.dim() == 3:
        z = z.flatten(0, 1)

    return z.float().var(dim=0)
```

Then log summary statistics:

```python
@torch.no_grad()
def collapse_metrics(
    z: torch.Tensor,
    eps: float = 1e-4,
) -> dict[str, float]:
    var = feature_variance(z)
    std = torch.sqrt(var + 1e-8)

    dead_fraction = (std < eps).float().mean()

    return {
        "std/mean": std.mean().item(),
        "std/min": std.min().item(),
        "std/max": std.max().item(),
        "std/dead_fraction": dead_fraction.item(),
    }
```

During training:

```python
with torch.no_grad():
    target_metrics = collapse_metrics(target_repr)
    pred_metrics = collapse_metrics(pred_repr)
```

A healthy representation should have nontrivial variance across dimensions.

---

## 1.4.9 Dimensional Collapse

Full collapse is obvious. Dimensional collapse is subtler.

The representation may not map every input to the same vector, but it may use only a small number of dimensions.

For example, a 768-dimensional embedding might effectively use only 20 dimensions. The remaining dimensions have near-zero variance.

This can hurt downstream performance and reduce the capacity of the latent state.

We can estimate dimensional usage by looking at the covariance spectrum.

Given centered representations:

\[
Z \in \mathbb{R}^{B \times D}
\]

the covariance matrix is:

\[
C =
\frac{1}{B-1}
Z^\top Z
\]

The eigenvalues of \(C\) indicate how variance is distributed across dimensions.

If most eigenvalues are near zero, the representation is low-rank.

In code:

```python
@torch.no_grad()
def covariance_spectrum(
    z: torch.Tensor,
) -> torch.Tensor:
    """
    Returns eigenvalues of the feature covariance matrix.
    """
    if z.dim() == 3:
        z = z.flatten(0, 1)

    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)

    cov = (z.T @ z) / max(z.size(0) - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov)

    return eigvals.flip(dims=[0])
```

An effective-rank diagnostic is useful:

```python
@torch.no_grad()
def effective_rank(
    z: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    eigvals = covariance_spectrum(z)
    eigvals = torch.clamp(eigvals, min=0)

    probs = eigvals / (eigvals.sum() + eps)
    entropy = -(probs * torch.log(probs + eps)).sum()

    return torch.exp(entropy).item()
```

The effective rank is high when variance is spread across many dimensions and low when the representation uses only a few directions.

This diagnostic is especially useful when the loss decreases but downstream quality is poor.

---

## 1.4.10 Redundancy and Whitening

A representation can also be redundant.

Two dimensions may encode nearly the same information. Redundancy is not always bad, but excessive redundancy reduces effective capacity.

The covariance matrix helps diagnose this.

If the representation is whitened, then:

\[
C \approx I
\]

meaning:

- each dimension has unit variance,
- different dimensions are uncorrelated.

Methods such as Barlow Twins and VICReg explicitly encourage decorrelation or covariance control. JEPA does not necessarily require explicit whitening, but monitoring covariance is still valuable.

A simple covariance off-diagonal metric:

```python
def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m

    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@torch.no_grad()
def covariance_metrics(
    z: torch.Tensor,
) -> dict[str, float]:
    if z.dim() == 3:
        z = z.flatten(0, 1)

    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)

    cov = (z.T @ z) / max(z.size(0) - 1, 1)

    diag = torch.diagonal(cov)
    off_diag = off_diagonal(cov)

    return {
        "cov/diag_mean": diag.mean().item(),
        "cov/diag_min": diag.min().item(),
        "cov/diag_max": diag.max().item(),
        "cov/off_diag_abs_mean": off_diag.abs().mean().item(),
        "cov/off_diag_sq_mean": off_diag.pow(2).mean().item(),
    }
```

We will not necessarily optimize these metrics directly in the minimal JEPA implementation. But we should log them during experiments.

---

## 1.4.11 Norm Geometry

Representation norms also matter.

A model may encode information in:

- the direction of \(z\),
- the magnitude \(\|z\|\),
- both.

Cosine losses discard norm information. MSE losses preserve it. Normalization layers also affect what information can be represented by scale.

A useful diagnostic:

```python
@torch.no_grad()
def norm_metrics(
    z: torch.Tensor,
) -> dict[str, float]:
    if z.dim() == 3:
        z = z.flatten(0, 1)

    norms = z.float().norm(dim=-1)

    return {
        "norm/mean": norms.mean().item(),
        "norm/std": norms.std().item(),
        "norm/min": norms.min().item(),
        "norm/max": norms.max().item(),
    }
```

If norms explode, training may become unstable. If norms shrink toward zero, the representation may collapse. If norms carry too much information, cosine-only evaluation may miss important structure.

A robust training setup should log representation norms for:

- online context representations,
- target representations,
- predictor outputs.

---

## 1.4.12 Neighborhood Geometry

Another way to inspect representation quality is nearest-neighbor retrieval.

If representations are semantically meaningful, nearby embeddings should correspond to similar inputs.

For images:

- images of similar objects should retrieve each other,
- related poses or scenes may cluster,
- augmentations of the same image should be near.

For cycling:

- similar workout states should retrieve each other,
- similar climb efforts may cluster,
- fatigue states may form trajectories.

A simple nearest-neighbor function:

```python
@torch.no_grad()
def nearest_neighbors(
    queries: torch.Tensor,
    database: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """
    queries:
        [num_queries, dim]

    database:
        [num_database, dim]

    returns:
        indices [num_queries, k]
    """
    queries = F.normalize(queries.float(), dim=-1)
    database = F.normalize(database.float(), dim=-1)

    similarities = queries @ database.T
    indices = similarities.topk(k=k, dim=-1).indices

    return indices
```

For images, we can visualize retrieved examples. For telemetry, we can inspect retrieved ride segments.

Nearest-neighbor retrieval is a qualitative but powerful diagnostic. It often reveals representation problems before linear probes do.

---

## 1.4.13 Linear Probing

A common way to evaluate representation geometry is linear probing.

Freeze the encoder and train a linear classifier or regressor on top:

\[
\hat{y} = W z + b
\]

If a simple linear model performs well, then the representation makes the downstream target easily accessible.

A minimal linear probe for image classification:

```python
class LinearProbe(torch.nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.classifier = torch.nn.Linear(dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)
```

Training step:

```python
def train_linear_probe_step(
    encoder,
    probe,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    with torch.no_grad():
        z = encoder(images)

        if z.dim() == 3:
            z = z.mean(dim=1)

    logits = probe(z)
    loss = F.cross_entropy(logits, labels)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.detach()
```

For cycling, a probe might predict:

- ride intensity zone,
- future heart-rate drift,
- interval completion,
- fatigue category,
- recovery need.

Linear probing does not fully measure representation quality, but it is simple and informative.

A representation that supports linear prediction of useful variables has a geometry aligned with those variables.

---

## 1.4.14 Trajectory Geometry

For world models, representation geometry should also respect time.

A sequence of observations:

\[
x_1, x_2, \dots, x_T
\]

maps to a sequence of representations:

\[
z_1, z_2, \dots, z_T
\]

A useful latent trajectory should be smoother and more structured than the raw observation trajectory.

For cycling, a ride segment might trace a path through latent space:

```text
easy riding → sustained tempo → threshold effort → fatigue accumulation → recovery
```

A temporal smoothness diagnostic:

```python
@torch.no_grad()
def temporal_smoothness(
    z: torch.Tensor,
) -> dict[str, float]:
    """
    z:
        [batch, time, dim]
    """
    delta = z[:, 1:] - z[:, :-1]
    step_norm = delta.float().norm(dim=-1)

    return {
        "temporal/step_norm_mean": step_norm.mean().item(),
        "temporal/step_norm_std": step_norm.std().item(),
        "temporal/step_norm_max": step_norm.max().item(),
    }
```

For action-conditioned settings, latent changes should correlate with actions.

A simple action-sensitivity diagnostic:

```python
@torch.no_grad()
def action_sensitivity(
    dynamics,
    z: torch.Tensor,
    actions_a: torch.Tensor,
    actions_b: torch.Tensor,
) -> dict[str, float]:
    """
    Compares predicted next states under two different actions.
    """
    z_next_a = dynamics(z, actions_a)
    z_next_b = dynamics(z, actions_b)

    diff = (z_next_a - z_next_b).float().norm(dim=-1)

    return {
        "action_sensitivity/mean": diff.mean().item(),
        "action_sensitivity/std": diff.std().item(),
    }
```

If different actions produce nearly identical latent transitions, the model may be ignoring actions. If tiny action changes produce enormous latent jumps, the dynamics may be unstable.

---

## 1.4.15 Representation Geometry for Planning

Planning requires more than good static representations.

A planning-friendly latent space should have several properties:

1. **Predictability**  
   Future latents should be predictable from current latents and actions.

2. **Smoothness**  
   Small changes in state or action should not produce arbitrary jumps.

3. **Task relevance**  
   The latent should preserve variables relevant to objectives.

4. **Abstraction**  
   Nuisance detail should be suppressed.

5. **Compositionality**  
   Multi-step rollouts should remain meaningful.

6. **Uncertainty awareness**  
   The model should express uncertainty when futures are ambiguous.

A latent rollout is useful only if distances and directions in latent space correspond to meaningful differences.

For example, in a cycling pacing model, a latent trajectory with increasing fatigue should be distinguishable from one with stable effort. A planning objective can then evaluate candidate action sequences.

```python
def plan_with_random_shooting(
    dynamics,
    objective,
    z0: torch.Tensor,
    action_sampler,
    num_candidates: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simple random-shooting planner in latent space.

    Returns:
        best_actions:
            [batch, horizon, action_dim]

        best_cost:
            [batch]
    """
    batch_size = z0.size(0)

    candidate_actions = action_sampler(
        batch_size=batch_size,
        num_candidates=num_candidates,
        horizon=horizon,
        device=z0.device,
    )

    z0_expanded = z0[:, None].expand(
        batch_size,
        num_candidates,
        *z0.shape[1:],
    )

    z = z0_expanded.reshape(batch_size * num_candidates, -1)
    actions = candidate_actions.reshape(
        batch_size * num_candidates,
        horizon,
        -1,
    )

    trajectory = rollout_latent_dynamics(
        dynamics=dynamics,
        z0=z,
        actions=actions,
    )

    costs = objective(trajectory)
    costs = costs.view(batch_size, num_candidates)

    best_idx = costs.argmin(dim=1)

    best_actions = candidate_actions[
        torch.arange(batch_size, device=z0.device),
        best_idx,
    ]

    best_cost = costs[
        torch.arange(batch_size, device=z0.device),
        best_idx,
    ]

    return best_actions, best_cost
```

This planner assumes the latent space is meaningful. If the representation geometry is bad, planning will be bad even if the dynamics loss is low.

That is why representation geometry is not an abstract side topic. It directly determines whether a world model can be used for decision-making.

---

## 1.4.16 A Unified Diagnostic Function

For JEPA experiments, we want a reusable diagnostic function.

```python
@torch.no_grad()
def representation_geometry_report(
    z: torch.Tensor,
    prefix: str = "repr",
) -> dict[str, float]:
    """
    Computes basic representation geometry diagnostics.

    z:
        [batch, dim] or [batch, tokens, dim]
    """
    if z.dim() == 3:
        z_flat = z.flatten(0, 1)
    else:
        z_flat = z

    z_flat = z_flat.float()

    # Norms
    norms = z_flat.norm(dim=-1)

    # Standard deviation per dimension
    std = z_flat.std(dim=0)

    # Covariance
    centered = z_flat - z_flat.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / max(z_flat.size(0) - 1, 1)
    diag = torch.diagonal(cov)
    off = off_diagonal(cov)

    # Effective rank
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
    probs = eigvals / (eigvals.sum() + 1e-12)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    eff_rank = torch.exp(entropy)

    return {
        f"{prefix}/norm_mean": norms.mean().item(),
        f"{prefix}/norm_std": norms.std().item(),
        f"{prefix}/std_mean": std.mean().item(),
        f"{prefix}/std_min": std.min().item(),
        f"{prefix}/std_max": std.max().item(),
        f"{prefix}/dead_dim_fraction": (std < 1e-4).float().mean().item(),
        f"{prefix}/cov_diag_mean": diag.mean().item(),
        f"{prefix}/cov_offdiag_abs_mean": off.abs().mean().item(),
        f"{prefix}/effective_rank": eff_rank.item(),
    }
```

In a JEPA training step:

```python
with torch.no_grad():
    logs.update(
        representation_geometry_report(
            target_repr,
            prefix="target",
        )
    )

    logs.update(
        representation_geometry_report(
            pred_repr,
            prefix="pred",
        )
    )
```

This gives us a basic health check for the geometry of learned representations.

---

## 1.4.17 What Good Geometry Looks Like

There is no single perfect geometry. It depends on the domain and task.

But for JEPA-style world models, a good representation should usually have:

- non-collapsed variance,
- reasonable effective rank,
- stable norms,
- meaningful nearest neighbors,
- predictable target representations,
- useful downstream probe performance,
- smooth temporal trajectories,
- action-sensitive latent transitions,
- robustness to nuisance variation.

A bad representation may show:

- near-zero variance,
- very low effective rank,
- exploding norms,
- arbitrary nearest neighbors,
- low loss but poor probes,
- no temporal coherence,
- action-insensitive dynamics,
- strong sensitivity to noise.

The key lesson is:

> Representation quality is geometric.

The loss is only one measurement of that geometry.

---

## 1.4.18 Summary

JEPA shapes a latent space by training a predictor to infer target representations from context representations.

This creates pressure toward representations that encode predictable structure. But the objective does not automatically guarantee useful semantics. The geometry can fail through collapse, dimensional collapse, redundancy, poor invariance, missing equivariance, or weak temporal organization.

For this reason, every serious JEPA implementation should include representation diagnostics from the beginning.

We will monitor:

- feature variance,
- effective rank,
- covariance structure,
- representation norms,
- prediction-target cosine similarity,
- nearest-neighbor structure,
- linear probe performance,
- temporal smoothness,
- action sensitivity.

The next chapter will move from the high-level foundation to implementation. We will start building the minimal I-JEPA system: patch embeddings, masks, encoders, predictors, losses, and training loops.

---

## References and Further Reading

- Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas, **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.  
  <https://arxiv.org/abs/2301.08243>

- Jean-Bastien Grill et al., **Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning**, 2020.  
  <https://arxiv.org/abs/2006.07733>

- Xinlei Chen, Kaiming He, **Exploring Simple Siamese Representation Learning**, 2020.  
  <https://arxiv.org/abs/2011.10566>

- Jure Zbontar et al., **Barlow Twins: Self-Supervised Learning via Redundancy Reduction**, 2021.  
  <https://arxiv.org/abs/2103.03230>

- Adrien Bardes, Jean Ponce, Yann LeCun, **VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning**, 2021.  
  <https://arxiv.org/abs/2105.04906>

- Randall Balestriero et al., **A Cookbook of Self-Supervised Learning**, 2023.  
  <https://arxiv.org/abs/2304.12210>

- Yann LeCun, **A Path Towards Autonomous Machine Intelligence**, 2022.  
  <https://openreview.net/forum?id=BZ5a1r-kVsf>