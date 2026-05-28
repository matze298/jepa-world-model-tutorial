# 1.0 Introduction

## Building JEPA-Style World Models from First Principles

Modern deep learning has made extraordinary progress by learning useful representations from data. Supervised learning maps inputs to labels. Generative modeling learns to reconstruct, predict, or sample observations. Reinforcement learning learns policies through interaction and reward. Each paradigm has produced powerful systems, but each also exposes a central difficulty: intelligent systems need to represent the world at the right level of abstraction.

A model that predicts every pixel of a future video frame is solving a brutally detailed problem. It must account for lighting, texture, camera noise, background clutter, and countless irrelevant microscopic variations. But an intelligent agent usually does not need all of that detail. A robot trying to grasp a cup does not need to predict the exact future RGB value of every pixel. A cyclist planning a climb does not need to predict the exact sensor noise in their heart-rate trace. A driver approaching an intersection does not need to reconstruct every photon reflected from the road surface.

What matters is not the full observation.

What matters is the latent structure that supports understanding, prediction, and action.

This is the core motivation behind **Joint-Embedding Predictive Architectures**, or **JEPAs**. A JEPA-style model does not try to reconstruct raw observations directly. Instead, it learns to predict representations of observations. The prediction target is not a pixel array, waveform, or token sequence, but an embedding: a learned abstraction of the part of the world we want the model to infer.

In its simplest image-based form, a JEPA receives a visible context region of an image and predicts the representation of one or more hidden target regions. This is the idea behind **I-JEPA**, the Image-based Joint-Embedding Predictive Architecture introduced by Assran et al. The key design choice is that the model predicts target-region representations rather than target-region pixels. This makes the task less about low-level reconstruction and more about inferring semantic structure. See [Assran et al., 2023 — Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243).

This tutorial develops that idea into a complete path from representation learning to world modeling.

We will begin with images because images make the central ingredients easy to isolate:

- visible context,
- hidden targets,
- latent representations,
- masking,
- predictor networks,
- target encoders,
- collapse diagnostics.

Then we will move from images to time. A world model is not merely a representation model. A world model predicts how the world evolves. In latent form, this means predicting future representations from past representations. Later, we will also condition those predictions on actions, giving us a practical bridge from self-supervised learning to planning.

The final goal is to build a JEPA-style world model that can be applied to a real-world sequential problem. Our running applied example will be **cycling intelligence**: learning latent dynamics from ride telemetry, predicting future physiological and performance states, and eventually using those predictions for pacing or training-plan decisions.

---

## 1.0 What We Are Trying to Build

At the highest level, we want a system that learns a useful latent representation of the world and predicts how that representation changes.

A standard representation model learns:

\[
z = f_\theta(x)
\]

where:

- \(x\) is an observation,
- \(f_\theta\) is an encoder,
- \(z\) is a latent representation.

For an image, \(x\) might be an RGB tensor:

\[
x \in \mathbb{R}^{3 \times H \times W}
\]

and \(z\) might be a vector or sequence of patch-level embeddings:

\[
z \in \mathbb{R}^{N \times D}
\]

where:

- \(N\) is the number of tokens or patches,
- \(D\) is the representation dimension.

A predictive representation model goes one step further. It learns to predict one representation from another.

For an image, we can split the image into a visible context and a hidden target:

\[
x = (x_{\mathrm{ctx}}, x_{\mathrm{tgt}})
\]

The model observes \(x_{\mathrm{ctx}}\) and predicts a representation of \(x_{\mathrm{tgt}}\):

\[
\hat{z}_{\mathrm{tgt}}
=
g_\theta(f_\theta(x_{\mathrm{ctx}}))
\]

The target representation is produced by an encoder:

\[
z_{\mathrm{tgt}}
=
f_{\bar{\theta}}(x_{\mathrm{tgt}})
\]

The loss compares predicted target representations to encoded target representations:

\[
\mathcal{L}
=
\left\|
\hat{z}_{\mathrm{tgt}}
-
z_{\mathrm{tgt}}
\right\|^2
\]

In a JEPA-style architecture, the target encoder is usually a slowly updated copy of the online encoder. We write its parameters as \(\bar{\theta}\), while the online parameters are \(\theta\). The target representation is stop-gradient-ed so that the target branch is not directly optimized by the loss:

\[
\mathcal{L}
=
\left\|
g_\theta(f_\theta(x_{\mathrm{ctx}}), m)
-
\mathrm{sg}(f_{\bar{\theta}}(x_{\mathrm{tgt}}))
\right\|_2^2
\]

Here:

- \(f_\theta\) is the online context encoder,
- \(f_{\bar{\theta}}\) is the target encoder,
- \(g_\theta\) is the predictor,
- \(m\) contains information about the target mask or target position,
- \(\mathrm{sg}(\cdot)\) denotes stop-gradient.

The target encoder is updated by exponential moving average:

\[
\bar{\theta}
\leftarrow
\tau \bar{\theta}
+
(1-\tau)\theta
\]

where \(\tau\) is close to one, for example \(0.99\), \(0.996\), or higher.

In PyTorch, this update is simple:

```python
@torch.no_grad()
def update_target_encoder(
    online_encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    tau: float,
) -> None:
    for online_param, target_param in zip(
        online_encoder.parameters(),
        target_encoder.parameters(),
    ):
        target_param.data.mul_(tau).add_(
            online_param.data,
            alpha=1.0 - tau,
        )
```

This tutorial will make every part of this objective concrete.

We will not treat the equation as a decorative research artifact. We will implement the encoders, masks, predictor, loss, EMA update, training loop, diagnostics, and evaluation tools.

---

## 1.0.1 From Pixel Prediction to Representation Prediction

A common way to train a self-supervised model is to corrupt the input and reconstruct the original observation. For images, a masked reconstruction objective can be written as:

\[
\mathcal{L}_{\mathrm{pixel}}
=
\left\|
\hat{x}_{\mathrm{tgt}}
-
x_{\mathrm{tgt}}
\right\|_2^2
\]

This is the basic idea behind many masked autoencoding approaches.

For example, **Masked Autoencoders** train a model by masking image patches and reconstructing the missing pixels. MAE uses an asymmetric encoder-decoder design: the encoder processes only visible patches, while a lightweight decoder reconstructs the missing pixels. See [He et al., 2021 — Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).

Pixel reconstruction is attractive because it provides a dense and unambiguous training signal. Every missing pixel becomes a supervised target.

But the objective also has a weakness. It rewards the model for predicting details that may not matter.

Suppose the hidden region contains part of a dog. A pixel reconstruction model is penalized if it predicts the wrong fur texture, the wrong exact lighting pattern, or a slightly different background. But for semantic understanding, those details may be irrelevant. What matters is that the missing region corresponds to an animal body, a continuation of a visible object, or a semantically meaningful structure.

We can write the observation schematically as:

\[
x
=
s
+
a
+
\epsilon
\]

where:

- \(s\) is semantic structure,
- \(a\) is appearance variation,
- \(\epsilon\) is noise or irrelevant detail.

A pixel-level objective forces the model to care about all three. A representation-level objective attempts to move the prediction target closer to \(s\), or at least toward features that are more stable and useful.

Instead of predicting:

\[
x_{\mathrm{tgt}}
\]

we predict:

\[
z_{\mathrm{tgt}} = f_{\bar{\theta}}(x_{\mathrm{tgt}})
\]

The model is no longer asked to answer:

> What are the exact missing pixels?

It is asked:

> What abstract representation should the missing region have?

This is the key move.

JEPA does not claim that pixels are useless. Pixel reconstruction can be a powerful training signal. But for world modeling, the aim is not to predict every observable detail. The aim is to predict the latent variables that matter for future understanding and action.

---

## 1.0.2 Why This Matters for World Models

A world model is a model of how the world evolves.

In observation space, a world model might try to learn:

\[
p(x_{t+1} \mid x_{\leq t}, a_{\leq t})
\]

That is, given past observations and actions, predict the next observation.

This can be useful, but it can also be wasteful. The next observation may contain enormous detail unrelated to the decision problem. A video frame contains shadows, textures, camera noise, irrelevant background motion, and other details that may not matter for control.

A latent world model instead learns something like:

\[
p(z_{t+1} \mid z_{\leq t}, a_{\leq t})
\]

or, in deterministic form:

\[
\hat{z}_{t+1}
=
F_\phi(z_{\leq t}, a_{\leq t})
\]

The model predicts future latent states rather than future observations.

This is where JEPA becomes interesting. JEPA gives us a self-supervised recipe for learning predictive latent representations. The image version predicts hidden spatial representations. The video version predicts hidden spatiotemporal representations. An action-conditioned version can predict future latent states under actions.

The progression is:

\[
x_{\mathrm{ctx}}
\rightarrow
z_{\mathrm{tgt}}
\]

for image JEPA,

\[
x_{\leq t}
\rightarrow
z_{t+k}
\]

for video or temporal JEPA,

and:

\[
(x_{\leq t}, a_{t:t+k})
\rightarrow
z_{t+k}
\]

for action-conditioned world modeling.

This progression mirrors the broader JEPA research program. Yann LeCun’s proposal for autonomous machine intelligence argues for predictive world models that operate in abstract representation space and can support planning under uncertainty. See [LeCun, 2022 — A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf).

Recent video JEPA work follows this direction. **V-JEPA** extends feature prediction to video by training models to predict masked spatiotemporal regions in representation space, without relying on reconstruction, negative examples, text, or pretrained image encoders. See [Bardes et al., 2024 — Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471). **V-JEPA 2** further connects self-supervised video learning with understanding, prediction, and planning in the physical world. See [Assran et al., 2025 — V-JEPA 2](https://arxiv.org/abs/2506.09985).

The research direction is clear: move from reconstructing observations to predicting abstractions, and from predicting abstractions to planning with them.

That is the path of this tutorial.

---

## 1.0.3 The Running Example: Cycling Intelligence

To keep the tutorial grounded, we will eventually apply the ideas to a real-world sequential problem: cycling performance modeling.

This is not just a toy example. Cycling data has many properties that make it a useful world-modeling domain:

- it is temporal,
- it is partially observable,
- it contains actions,
- it contains delayed consequences,
- it contains physiological state,
- it contains environmental context,
- it supports planning and counterfactual reasoning.

A single ride can be represented as a multivariate time series:

\[
x_t
=
[
\mathrm{power}_t,
\mathrm{heart\ rate}_t,
\mathrm{cadence}_t,
\mathrm{speed}_t,
\mathrm{gradient}_t,
\mathrm{temperature}_t,
\dots
]
\]

A model might encode a window of recent observations:

\[
z_t
=
f_\theta(x_{t-L:t})
\]

and predict a future latent state:

\[
\hat{z}_{t+k}
=
F_\phi(z_t, a_{t:t+k})
\]

where \(a_t\) could include planned power, cadence, pacing decisions, or route segment information.

Eventually, this allows questions like:

- What happens if the rider holds 310 W for the next five minutes?
- How likely is heart-rate drift under this pacing strategy?
- Which effort profile minimizes predicted fatigue while maintaining target speed?
- How should intervals be prescribed given recent training history?
- Can we simulate future performance degradation under different pacing plans?

This gives us a concrete path from JEPA-style representation prediction to a usable latent dynamics model.

The important point is that we do not need to predict the exact future sensor trace in full detail. We need a representation that captures the relevant physiological and performance state.

For example, the raw target may be:

\[
x_{t:t+k}
\]

but the useful prediction target may be:

\[
z_{t:t+k}
\]

where \(z\) encodes something closer to fatigue, sustainability, cardiac response, or effort state.

This is why JEPA-style thinking is attractive for applied world modeling.

---

## 1.0.4 Implementation Philosophy

This tutorial is written for readers who want both the research story and the engineering details.

The guiding rule is:

> Every mathematical object should eventually become code.

When we introduce the online encoder:

\[
f_\theta
\]

we will implement it.

When we introduce the target encoder:

\[
f_{\bar{\theta}}
\]

we will implement its initialization, stop-gradient behavior, and EMA update.

When we introduce context and target masks:

\[
m_{\mathrm{ctx}}, m_{\mathrm{tgt}}
\]

we will implement mask samplers and visualize their outputs.

When we introduce the predictor:

\[
g_\theta
\]

we will implement its token interface, positional encoding strategy, and transformer blocks.

A simplified JEPA training step looks like this:

```python
def train_step(
    model,
    images,
    context_indices,
    target_indices,
    optimizer,
    ema_tau: float,
):
    pred_repr, target_repr = model(
        images=images,
        context_indices=context_indices,
        target_indices=target_indices,
    )

    loss = torch.nn.functional.smooth_l1_loss(
        pred_repr,
        target_repr.detach(),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    update_target_encoder(
        online_encoder=model.online_encoder,
        target_encoder=model.target_encoder,
        tau=ema_tau,
    )

    return loss
```

The corresponding model interface might look like this:

```python
class MinimalIJEPA(torch.nn.Module):
    def __init__(
        self,
        online_encoder: torch.nn.Module,
        target_encoder: torch.nn.Module,
        predictor: torch.nn.Module,
    ):
        super().__init__()
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

At this point, many details are unresolved:

- What exactly is an encoder?
- How are images patchified?
- How are context and target indices sampled?
- How does the predictor know target positions?
- Should target representations be normalized?
- Should the loss be MSE, cosine distance, or Smooth L1?
- How do we prevent collapse?
- How do we evaluate the representation?
- How do we know whether the model is learning semantics rather than shortcuts?

These questions are not implementation footnotes. They are the substance of the tutorial.

Each chapter will refine the code until it becomes a working research-quality implementation.

---

## 1.0.5 What Makes This Tutorial Different

There are official repositories and research papers for I-JEPA, V-JEPA, and related methods. Those are essential references. But this tutorial has a different purpose.

The goal is not simply to run existing code.

The goal is to understand the method deeply enough to rebuild it, modify it, debug it, and extend it.

That requires working through several layers:

1. **Conceptual layer**
   Why predict representations instead of pixels?

2. **Mathematical layer**
   What objective is being optimized, and what assumptions does it encode?

3. **Architectural layer**
   What are the encoder, target encoder, predictor, and mask sampler?

4. **Optimization layer**
   How do EMA, stop-gradient, normalization, and predictor asymmetry affect training?

5. **Diagnostic layer**
   How do we detect collapse, shortcut learning, leakage, or useless representations?

6. **Systems layer**
   How do we train efficiently and reproducibly?

7. **World-model layer**
   How do we move from static prediction to temporal and action-conditioned dynamics?

8. **Application layer**
   How do we use latent predictions for real planning problems?

This is why the tutorial begins with I-JEPA but does not end there.

I-JEPA is the cleanest entry point.

The final destination is a practical world model.

---

## 1.0.6 Local Documentation Setup

This tutorial is designed to be written and hosted locally as a Markdown documentation site.

The recommended setup is:

```text
MkDocs Material + Markdown + PyTorch code + optional notebooks
```

A useful project structure is:

```text
jepa-world-model-tutorial/
├── mkdocs.yml
├── docs/
│   ├── index.md
│   ├── chapter-01/
│   │   ├── introduction.md
│   │   └── ssl-landscape.md
│   ├── chapter-02/
│   │   └── jepa-objective.md
│   ├── references.md
│   └── assets/
│       ├── figures/
│       └── code/
├── src/
│   └── jepa_world_model/
├── notebooks/
├── configs/
├── experiments/
├── pyproject.toml
└── README.md
```

Run locally with:

```bash
mkdocs serve
```

Then open:

```text
http://127.0.0.1:8000
```

This structure keeps the tutorial readable as a website while allowing the implementation to grow into a proper codebase.

---

## 1.0.7 Roadmap

The tutorial follows a staged progression.

First, we study the self-supervised learning landscape and place JEPA in context.

Then we derive the JEPA objective carefully.

Next, we implement a minimal image JEPA from scratch.

After that, we scale the implementation into a research-grade codebase.

Then we move from image prediction to temporal prediction.

Finally, we extend the model to action-conditioned latent dynamics and apply it to cycling intelligence.

The core path is:

```text
Representation learning
        ↓
Masked representation prediction
        ↓
I-JEPA implementation
        ↓
Video / temporal JEPA
        ↓
Action-conditioned latent dynamics
        ↓
Real-world planning application
```

The mathematical path is:

\[
z = f(x)
\]

\[
\hat{z}_{\mathrm{tgt}}
=
g(z_{\mathrm{ctx}})
\]

\[
\hat{z}_{t+k}
=
F(z_{\leq t})
\]

\[
\hat{z}_{t+k}
=
F(z_{\leq t}, a_{t:t+k})
\]

The implementation path is:

```text
patch embeddings
        ↓
mask samplers
        ↓
online encoder
        ↓
target encoder
        ↓
predictor
        ↓
latent loss
        ↓
EMA update
        ↓
diagnostics
        ↓
temporal extension
        ↓
action-conditioned rollout
```

By the end, the reader should be able to:

- explain why JEPA predicts representations instead of pixels,
- implement the core I-JEPA training loop,
- debug collapse and mask leakage,
- evaluate learned representations,
- extend image JEPA ideas to temporal data,
- build an action-conditioned latent dynamics model,
- apply the model to a real-world sequential decision problem.

---

## 1.0.8 Summary

This chapter introduced the central idea of the tutorial: world models should predict useful abstractions, not necessarily raw observations.

JEPA provides a clean framework for this idea. It trains a model to infer the representation of hidden or future parts of the world from visible or past parts. In images, this means predicting target-region embeddings from context-region embeddings. In videos, it means predicting masked spatiotemporal representations. In action-conditioned settings, it means predicting future latent states under possible actions.

The rest of the tutorial will build this idea step by step.

We begin with the broader self-supervised learning landscape.

---

## References and Further Reading

- Yann LeCun, **A Path Towards Autonomous Machine Intelligence**, 2022.
  <https://openreview.net/forum?id=BZ5a1r-kVsf>

- Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas, **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.
  <https://arxiv.org/abs/2301.08243>

- Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick, **Masked Autoencoders Are Scalable Vision Learners**, 2021.
  <https://arxiv.org/abs/2111.06377>

- Adrien Bardes et al., **Revisiting Feature Prediction for Learning Visual Representations from Video**, 2024.
  <https://arxiv.org/abs/2404.08471>

- Mahmoud Assran et al., **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**, 2025.
  <https://arxiv.org/abs/2506.09985>

- Meta AI, **V-JEPA: The next step toward advanced machine intelligence**, 2024.
  <https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/>

- Facebook Research, **Official I-JEPA Codebase**.
  <https://github.com/facebookresearch/ijepa>
