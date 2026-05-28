# 1.2 Why Pixel Prediction Is Problematic

Pixel prediction is one of the most natural self-supervised learning objectives.

It is simple, dense, and easy to evaluate. Given an observed context, ask the model to predict the missing pixels. If the prediction is close to the true pixels, the loss is low. If it is far away, the loss is high.

For an image, a masked pixel-prediction objective can be written as:

\[
\mathcal{L}_{\mathrm{pixel}}
=
\left\|
\hat{x}_{\mathcal{M}}
-
x_{\mathcal{M}}
\right\|_2^2
\]

where:

- \(\mathcal{M}\) is the set of masked patches,
- \(x_{\mathcal{M}}\) are the true masked pixels,
- \(\hat{x}_{\mathcal{M}}\) are the predicted masked pixels.

This objective is the foundation of masked reconstruction methods such as **Masked Autoencoders**. MAE masks random image patches and trains a decoder to reconstruct the missing pixels. Its asymmetric encoder-decoder design lets the encoder process only visible patches, while a lightweight decoder reconstructs the original image from latent representations and mask tokens. See [He et al., 2021 — Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).

MAE is an important and successful method. Pixel reconstruction is not wrong.

But if our long-term goal is to build JEPA-style world models, pixel prediction exposes several problems:

1. raw observations are too detailed,
2. future observations are often multimodal,
3. pixel losses overweight irrelevant variation,
4. reconstruction can encourage shortcut learning,
5. observation prediction does not necessarily produce action-useful state,
6. pixel-space world models can waste capacity modeling nuisance factors.

This section explains these issues carefully.

The purpose is not to dismiss pixel prediction. The purpose is to understand why JEPA changes the target from pixels to representations.

---

## 1.2.1 Pixel Prediction Solves a Harder Problem Than We Usually Need

A pixel-prediction model tries to recover the observation itself.

For an image, the target is:

\[
x \in \mathbb{R}^{3 \times H \times W}
\]

For a video, the target might be:

\[
x_{t+1:t+K}
\in
\mathbb{R}^{K \times 3 \times H_{\mathrm{img}} \times W_{\mathrm{img}}}
\]

For a sensor sequence, the target might be:

\[
x_{t+1:t+K}
\in
\mathbb{R}^{K \times D}
\]

where \(K\) is the prediction horizon and \(D\) is the number of observed channels.

The problem is that the raw observation contains much more information than the model may need.

An image contains:

- object identity,
- object pose,
- object boundaries,
- depth cues,
- lighting,
- texture,
- shadows,
- background clutter,
- camera noise,
- compression artifacts.

Some of these factors are useful. Others are nuisance variables.

A future video frame contains even more nuisance detail. To predict the exact future image, the model must account for:

- precise object motion,
- exact camera motion,
- specular reflections,
- cloth deformation,
- lighting flicker,
- stochastic background motion,
- sensor noise.

But a model that supports reasoning or control may only need a compressed state.

For example, a robot deciding how to grasp a cup may need to know:

- where the cup is,
- what shape it has,
- whether it is reachable,
- whether it is likely to fall,
- what action would move it.

It probably does not need to predict the exact texture of the table one second from now.

Likewise, a cycling world model may need to know:

- current fatigue state,
- heart-rate response,
- recent exertion load,
- sustainable power,
- terrain context,
- risk of overpacing.

It does not need to predict every sample of future sensor noise.

This distinction can be written as:

\[
x_t
=
(s_t, n_t)
\]

where:

- \(s_t\) is task-relevant state,
- \(n_t\) is nuisance detail.

Pixel prediction tries to model both:

\[
p(x_{t+1} \mid x_{\leq t})
=
p(s_{t+1}, n_{t+1} \mid x_{\leq t})
\]

But a world model for decision-making may mostly need:

\[
p(s_{t+1} \mid s_{\leq t}, a_{\leq t})
\]

JEPA-style methods try to learn a representation \(z_t\) that is closer to \(s_t\) than to the full observation \(x_t\).

---

## 1.2.2 Pixel Prediction Penalizes Plausible Alternatives

Pixel losses assume that the correct answer is the observed target.

For a deterministic reconstruction loss:

\[
\mathcal{L}
=
\left\|
\hat{x}
-
x
\right\|_2^2
\]

the model is penalized for any prediction that differs from the observed pixels.

But many prediction problems are inherently multimodal.

Suppose a masked image region contains grass. There may be many plausible textures that would be semantically correct. But an MSE loss rewards only the exact observed texture. A slightly different, equally plausible grass texture receives a penalty.

For future video, the problem is even stronger. If a person is walking, several future poses may be plausible. If a cyclist approaches a climb, several pacing trajectories may be plausible. If a ball is bouncing, small uncertainty in velocity can produce large pixel-level differences after a few frames.

A unimodal pixel loss tends to average over plausible futures.

If the true conditional distribution is:

\[
p(x_{t+1} \mid x_{\leq t})
\]

and this distribution has multiple modes, a simple squared-error predictor learns something like a conditional mean:

\[
\hat{x}_{t+1}
\approx
\mathbb{E}[x_{t+1} \mid x_{\leq t}]
\]

For images or videos, this conditional mean may be blurry or physically implausible.

This is a classic problem in observation-space prediction.

A representation-space target can reduce the burden. If multiple pixel-level futures correspond to the same semantic state, their representations may be closer together than their raw pixels.

We can express this as:

\[
x^{(1)} \neq x^{(2)}
\]

but:

\[
f(x^{(1)}) \approx f(x^{(2)})
\]

if both observations share the same relevant structure.

This is the hope behind latent prediction.

Instead of asking the model to choose an exact pixel future, we ask it to predict a representation that is invariant to irrelevant differences.

---

## 1.2.3 Pixel Losses Overweight Low-Level Detail

A pixel loss treats every pixel dimension as part of the target.

For an RGB image:

\[
x \in \mathbb{R}^{3HW}
\]

the MSE loss is:

\[
\mathcal{L}_{\mathrm{MSE}}
=
\frac{1}{3HW}
\sum_{c,h,w}
(\hat{x}_{c,h,w} - x_{c,h,w})^2
\]

This gives enormous weight to low-level appearance.

Consider two prediction errors:

1. the model predicts the correct object but slightly wrong texture,
2. the model predicts the wrong object but similar color distribution.

A pixel loss may not distinguish these in the way a semantic task would.

For example, predicting a dog-shaped region with slightly incorrect fur texture might receive a larger pixel loss than predicting a background-like patch with similar average color. The pixel objective has no intrinsic notion of objectness, affordance, or semantic continuity.

This issue is not limited to images.

For cycling telemetry, a raw time-series MSE may overweight high-frequency fluctuations:

\[
\mathcal{L}_{\mathrm{series}}
=
\frac{1}{HD}
\sum_{t=1}^{H}
\sum_{d=1}^{D}
(\hat{x}_{t,d} - x_{t,d})^2
\]

If heart rate has sensor jitter, cadence has dropouts, or speed fluctuates due to GPS noise, the model may spend capacity reducing errors that are not central to physiological state.

A latent target allows us to define prediction at a more useful level:

\[
z = f(x)
\]

and:

\[
\mathcal{L}_{\mathrm{latent}}
=
\left\|
\hat{z}
-
z
\right\|^2
\]

The encoder \(f\) can, in principle, learn to suppress irrelevant variation.

This changes the learning signal from:

> Match every observed dimension.

to:

> Match the representation of the target.

That is the essential JEPA move.

---

## 1.2.4 Reconstruction Can Encourage Shortcut Learning

Masked reconstruction seems to require understanding. But depending on the masking scheme, the task can sometimes be solved with local interpolation.

If the model sees neighboring pixels around a small missing patch, it may reconstruct the target from texture continuation rather than semantic reasoning.

For example:

```text
visible grass pixels → predict missing grass pixels
visible sky pixels   → predict missing sky pixels
visible wall texture → predict missing wall texture
```

This can train useful local features, but it may not force the model to learn higher-level structure.

Mask design matters.

If masks are small and randomly scattered, the model may rely heavily on local continuity. If masks are large and structured, the model has to infer missing content from broader context.

This is why I-JEPA emphasizes the masking strategy. I-JEPA predicts representations of target blocks from context blocks in the same image, and the authors argue that target blocks should be sufficiently large while context should be spatially distributed and informative. See [Assran et al., 2023 — I-JEPA](https://arxiv.org/abs/2301.08243).

The difference can be illustrated as:

```text
Small random mask:
    "Fill in the missing texture."

Large semantic block:
    "Infer what object or structure belongs here."
```

In code, a small random patch mask might look like this:

```python
def sample_random_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_masked = int(mask_ratio * num_patches)

    noise = torch.rand(batch_size, num_patches, device=device)
    perm = noise.argsort(dim=1)

    masked = perm[:, :num_masked]
    visible = perm[:, num_masked:]

    return visible, masked
```

A block mask forces spatial structure:

```python
def sample_rectangular_block(
    grid_size: int,
    block_h: int,
    block_w: int,
    device: torch.device,
) -> torch.Tensor:
    max_row = grid_size - block_h
    max_col = grid_size - block_w

    row = torch.randint(0, max_row + 1, (), device=device)
    col = torch.randint(0, max_col + 1, (), device=device)

    rows = torch.arange(row, row + block_h, device=device)
    cols = torch.arange(col, col + block_w, device=device)

    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    return (rr * grid_size + cc).flatten()
```

The block mask creates a more meaningful prediction task because the missing region is spatially coherent.

But even a block mask is not enough by itself. If the target is still pixels, the model may still spend capacity on low-level reconstruction. JEPA combines structured masking with representation prediction.

---

## 1.2.5 Observation Prediction Is Not the Same as State Prediction

A world model should predict state.

But raw observations are not necessarily state.

An observation is what the sensor sees:

\[
x_t
\]

A state is the information needed to predict and act:

\[
s_t
\]

In a fully observed Markov decision process, the state satisfies:

\[
p(s_{t+1} \mid s_{\leq t}, a_{\leq t})
=
p(s_{t+1} \mid s_t, a_t)
\]

But real observations are often partial, noisy, and redundant.

For example, in cycling:

\[
x_t =
[
\mathrm{power}_t,
\mathrm{heart\ rate}_t,
\mathrm{cadence}_t,
\mathrm{speed}_t,
\mathrm{gradient}_t
]
\]

This is not the full physiological state.

The hidden state may include:

- glycogen availability,
- accumulated muscular fatigue,
- thermal stress,
- hydration status,
- aerobic strain,
- recovery state,
- motivation or perceived exertion.

Some of these are not directly observed.

A model trained only to predict the next raw telemetry vector may learn short-term signal dynamics without discovering a useful latent state. For instance, it may learn that heart rate is smooth and power is autocorrelated, but fail to infer deeper fatigue accumulation.

A latent predictive model instead tries to learn:

\[
z_t = f(x_{\leq t})
\]

where \(z_t\) summarizes the history in a way that is useful for predicting future latent targets:

\[
\hat{z}_{t+k}
=
F(z_t, a_{t:t+k})
\]

The purpose of \(z_t\) is not merely compression. It is predictive state abstraction.

This is why representation learning and world modeling are connected.

A useful representation is not just one that reconstructs the input. It is one that supports prediction and action.

---

## 1.2.6 Pixel-Space World Models Can Waste Capacity

World models can be trained in observation space.

For example:

\[
\hat{x}_{t+1}
=
G_\theta(x_{\leq t}, a_t)
\]

This is attractive because the target is available. No labels are needed.

But the model may spend enormous capacity on observation details.

For video prediction, the model has to model:

- background appearance,
- object texture,
- lighting,
- camera motion,
- occlusion boundaries,
- stochastic motion.

For many control tasks, this is excessive.

A latent world model instead learns:

\[
z_t = f_\theta(x_{\leq t})
\]

\[
\hat{z}_{t+1}
=
F_\phi(z_t, a_t)
\]

and optionally decodes only when needed:

\[
\hat{x}_{t+1}
=
d_\psi(\hat{z}_{t+1})
\]

The decoder becomes optional. The latent state is the main object.

This distinction matters because a world model used for planning may roll out many imagined futures. If every rollout requires predicting high-dimensional observations, planning becomes expensive and fragile.

In latent space, a rollout can be much cheaper:

```python
def rollout_latent_dynamics(
    dynamics,
    z0: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """
    z0:
        [batch, latent_dim]

    actions:
        [batch, horizon, action_dim]

    returns:
        latent trajectory [batch, horizon + 1, latent_dim]
    """
    z = z0
    trajectory = [z]

    for t in range(actions.size(1)):
        z = dynamics(z, actions[:, t])
        trajectory.append(z)

    return torch.stack(trajectory, dim=1)
```

Planning can then evaluate trajectories in latent space:

```python
def evaluate_pacing_plan(
    latent_trajectory: torch.Tensor,
    objective_head,
) -> torch.Tensor:
    """
    Example:
        predict fatigue cost or failure probability from latent states.
    """
    cost = objective_head(latent_trajectory)
    return cost.sum(dim=1)
```

This is the direction we want for the cycling application.

We do not need to reconstruct every second of future telemetry perfectly. We need to evaluate the consequences of candidate pacing strategies.

That suggests a latent predictive model, not necessarily a pixel- or sensor-space reconstruction model.

---

## 1.2.7 Representation Prediction Changes the Bottleneck

A bottleneck determines what information must pass through the model.

In an autoencoder, the bottleneck is usually the latent vector:

\[
z = f(x)
\]

If the decoder must reconstruct the full input, then \(z\) must preserve enough information for reconstruction.

This can be useful, but it can also force \(z\) to carry nuisance detail.

In JEPA, the bottleneck is different.

The context encoder sees only part of the input:

\[
z_{\mathrm{ctx}} = f_\theta(x_{\mathrm{ctx}})
\]

The predictor must infer the target representation:

\[
\hat{z}_{\mathrm{tgt}}
=
g_\theta(z_{\mathrm{ctx}}, m)
\]

The target is itself a representation:

\[
z_{\mathrm{tgt}}
=
f_{\bar{\theta}}(x_{\mathrm{tgt}})
\]

So the model is bottlenecked twice:

1. the context is partial,
2. the target is abstract.

This encourages the model to learn features that are predictable from context and useful across views.

A useful target representation should not encode arbitrary noise, because arbitrary noise is not predictable from context. If the target encoder encodes unpredictable nuisance details, the predictor cannot learn them reliably. Over training, the system is biased toward predictable structure.

This is one of the most important intuitions behind JEPA.

The model learns what is predictable in representation space.

---

## 1.2.8 The JEPA Alternative

JEPA replaces pixel targets with representation targets.

Instead of:

```python
target = target_pixels
prediction = decoder(context_tokens)
loss = mse(prediction, target)
```

we use:

```python
with torch.no_grad():
    target = target_encoder(images, target_indices)

context = online_encoder(images, context_indices)
prediction = predictor(context, target_indices)

loss = latent_loss(prediction, target)
```

The conceptual difference is:

| Method | Context | Target | Loss |
|---|---|---|---|
| MAE | visible patches | masked pixels | pixel reconstruction |
| I-JEPA | context patches | target patch representations | latent prediction |
| Temporal JEPA | past window | future representation | latent prediction |
| Action world model | past + actions | future latent state | latent dynamics loss |

A minimal JEPA-like loss is:

```python
def latent_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(
        pred,
        target.detach(),
    )
```

A normalized cosine version is also possible:

```python
def cosine_latent_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    pred = torch.nn.functional.normalize(pred, dim=-1)
    target = torch.nn.functional.normalize(target.detach(), dim=-1)

    return 1.0 - (pred * target).sum(dim=-1).mean()
```

The choice of loss matters.

MSE or Smooth L1 preserves magnitude information. Cosine loss focuses on direction. Normalization may help stability but can remove potentially useful scale. Later chapters will compare these choices experimentally.

The key point is that JEPA moves the target into representation space.

---

## 1.2.9 A Concrete Example: Masked Image Prediction

Let us compare MAE and I-JEPA on the same image.

Assume an image is divided into \(N\) patches:

\[
x = [x_1, x_2, \dots, x_N]
\]

We sample context indices:

\[
\mathcal{C}
\]

and target indices:

\[
\mathcal{T}
\]

### MAE-style objective

The encoder processes visible patches:

\[
h_\mathcal{C}
=
f_\theta(x_\mathcal{C})
\]

The decoder predicts target pixels:

\[
\hat{x}_\mathcal{T}
=
d_\phi(h_\mathcal{C}, \mathcal{T})
\]

The loss is:

\[
\mathcal{L}_{\mathrm{MAE}}
=
\left\|
\hat{x}_\mathcal{T}
-
x_\mathcal{T}
\right\|^2
\]

Code skeleton:

```python
visible_tokens = encoder(images, context_indices)
pred_pixels = decoder(visible_tokens, target_indices)
target_pixels = patchify_and_gather(images, target_indices)

loss = F.mse_loss(pred_pixels, target_pixels)
```

### I-JEPA-style objective

The online encoder processes context patches:

\[
h_\mathcal{C}
=
f_\theta(x_\mathcal{C})
\]

The target encoder processes target patches:

\[
z_\mathcal{T}
=
f_{\bar{\theta}}(x_\mathcal{T})
\]

The predictor predicts target representations:

\[
\hat{z}_\mathcal{T}
=
g_\theta(h_\mathcal{C}, \mathcal{T})
\]

The loss is:

\[
\mathcal{L}_{\mathrm{IJEPA}}
=
\left\|
\hat{z}_\mathcal{T}
-
\mathrm{sg}(z_\mathcal{T})
\right\|^2
\]

Code skeleton:

```python
context_repr = online_encoder(images, context_indices)

with torch.no_grad():
    target_repr = target_encoder(images, target_indices)

pred_repr = predictor(
    context_repr=context_repr,
    context_indices=context_indices,
    target_indices=target_indices,
)

loss = F.smooth_l1_loss(pred_repr, target_repr)
```

The difference is the target.

This difference changes what the model is rewarded for learning.

---

## 1.2.10 A Concrete Example: Cycling Telemetry

Now consider a cycling sequence.

Let:

\[
x_t =
[
P_t,
HR_t,
C_t,
V_t,
G_t
]
\]

where:

- \(P_t\) is power,
- \(HR_t\) is heart rate,
- \(C_t\) is cadence,
- \(V_t\) is speed,
- \(G_t\) is gradient.

A raw prediction model might learn:

\[
\hat{x}_{t+1:t+K}
=
G_\theta(x_{t-L:t}, a_{t:t+K})
\]

with loss:

\[
\mathcal{L}_{\mathrm{raw}}
=
\left\|
\hat{x}_{t+1:t+K}
-
x_{t+1:t+K}
\right\|^2
\]

This objective may be useful, especially for short-horizon forecasting. But it can overemphasize exact signal matching.

A JEPA-style temporal objective instead learns:

\[
z_{\mathrm{past}}
=
f_\theta(x_{t-L:t})
\]

\[
z_{\mathrm{future}}
=
f_{\bar{\theta}}(x_{t+1:t+K})
\]

\[
\hat{z}_{\mathrm{future}}
=
F_\phi(z_{\mathrm{past}}, a_{t:t+K})
\]

with loss:

\[
\mathcal{L}_{\mathrm{latent}}
=
\left\|
\hat{z}_{\mathrm{future}}
-
\mathrm{sg}(z_{\mathrm{future}})
\right\|^2
\]

Code skeleton:

```python
past_window = batch["past_observations"]
future_window = batch["future_observations"]
future_actions = batch["future_actions"]

past_z = context_encoder(past_window)

with torch.no_grad():
    future_z = target_encoder(future_window)

pred_future_z = dynamics_model(
    past_z=past_z,
    future_actions=future_actions,
)

loss = F.mse_loss(pred_future_z, future_z)
```

The raw model asks:

> Can I predict the future telemetry trace?

The latent model asks:

> Can I predict the future physiological/performance state representation?

For coaching or planning, the second question may be more useful.

---

## 1.2.11 What Pixel Prediction Still Does Well

Pixel prediction is not obsolete.

Reconstruction objectives have several strengths:

- they are simple,
- they are stable,
- they provide dense supervision,
- they are easy to debug,
- they can learn strong visual features,
- they can be combined with latent objectives.

MAE is a strong demonstration that masked pixel reconstruction can scale well and produce useful representations. The lesson is not that pixel prediction fails. The lesson is that pixel prediction optimizes a different target than JEPA.

For some applications, reconstructing observations is exactly what we want.

Examples:

- image restoration,
- denoising,
- super-resolution,
- inpainting,
- generative modeling,
- compression,
- simulation where visual fidelity matters.

For world models used in planning, however, full observation reconstruction may be unnecessary or even counterproductive.

A useful compromise is a hybrid model:

\[
\mathcal{L}
=
\lambda_{\mathrm{latent}}
\mathcal{L}_{\mathrm{latent}}
+
\lambda_{\mathrm{recon}}
\mathcal{L}_{\mathrm{recon}}
\]

In code:

```python
loss = (
    lambda_latent * latent_prediction_loss(pred_z, target_z)
    + lambda_recon * reconstruction_loss(pred_x, target_x)
)
```

This can be useful when we want both:

- abstract predictive representations,
- enough reconstruction pressure to preserve important details.

Later chapters will focus first on pure latent prediction. Hybrid objectives will be treated as an extension.

---

## 1.2.12 Diagnostic Consequences

Because JEPA does not reconstruct pixels, we need different diagnostics.

For MAE, a quick visual check is natural:

```text
Does the reconstructed image look correct?
```

For JEPA, the model predicts representations. We cannot directly look at \(\hat{z}\) and know whether it is good.

Therefore we need representation diagnostics.

Useful diagnostics include:

- feature variance,
- feature norm,
- cosine similarity between prediction and target,
- collapse indicators,
- nearest-neighbor retrieval,
- linear probing,
- downstream transfer,
- attention or mask visualizations,
- prediction error by target location.

A minimal diagnostic function might compute basic representation statistics:

```python
@torch.no_grad()
def latent_diagnostics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """
    pred, target:
        [batch, tokens, dim]
    """
    pred_flat = pred.float().flatten(0, 1)
    target_flat = target.float().flatten(0, 1)

    pred_norm = pred_flat.norm(dim=-1).mean()
    target_norm = target_flat.norm(dim=-1).mean()

    pred_std = pred_flat.std(dim=0).mean()
    target_std = target_flat.std(dim=0).mean()

    cosine = F.cosine_similarity(
        pred_flat,
        target_flat,
        dim=-1,
    ).mean()

    return {
        "pred/norm": pred_norm.item(),
        "target/norm": target_norm.item(),
        "pred/std": pred_std.item(),
        "target/std": target_std.item(),
        "pred_target/cosine": cosine.item(),
    }
```

These diagnostics are not optional. In JEPA-style learning, the model can appear to train while learning a weak or collapsed representation.

The loss alone is not enough.

---

## 1.2.13 Summary

Pixel prediction is a powerful self-supervised objective, but it solves a very detailed problem. It asks the model to predict raw observations, including texture, lighting, noise, and other nuisance variation.

For semantic representation learning and world modeling, this can be inefficient. The model may spend capacity predicting details that do not matter for downstream reasoning or action. Future observations are also often multimodal, making exact pixel or signal prediction unnecessarily difficult.

JEPA changes the target.

Instead of predicting:

\[
x_{\mathrm{tgt}}
\]

it predicts:

\[
z_{\mathrm{tgt}} = f_{\bar{\theta}}(x_{\mathrm{tgt}})
\]

This shifts the problem from observation reconstruction to latent prediction.

The rest of the tutorial builds on this idea. We will now move from intuition to a more precise mathematical formulation of the JEPA objective.

---

## References and Further Reading

- Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick, **Masked Autoencoders Are Scalable Vision Learners**, 2021.
  <https://arxiv.org/abs/2111.06377>

- Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas, **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.
  <https://arxiv.org/abs/2301.08243>

- Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, Nicolas Ballas, **Revisiting Feature Prediction for Learning Visual Representations from Video**, 2024.
  <https://arxiv.org/abs/2404.08471>

- Yann LeCun, **A Path Towards Autonomous Machine Intelligence**, 2022.
  <https://openreview.net/forum?id=BZ5a1r-kVsf>

- Danijar Hafner et al., **Learning Latent Dynamics for Planning from Pixels**, 2019.
  <https://arxiv.org/abs/1811.04551>

- Danijar Hafner et al., **Dream to Control: Learning Behaviors by Latent Imagination**, 2020.
  <https://arxiv.org/abs/1912.01603>
