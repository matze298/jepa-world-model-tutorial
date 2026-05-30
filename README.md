# JEPA-Style World Models

This repository is a tutorial and implementation project for building JEPA-style world models from first principles.

The tutorial itself is the documentation, and it lives in `docs/`. As the tutorial progresses, it builds the implementation in `src/` alongside it.

> Disclaimer: this tutorial is a work in progress and is primarily written as a self-teaching project. It is AI-generated, and I am not an expert. If you need rigorous correctness, please verify the material against the original sources linked in [References](docs/references.md), especially the [I-JEPA paper](https://arxiv.org/abs/2301.08243) and the [official I-JEPA codebase](https://github.com/facebookresearch/ijepa).

## Local Setup

Make sure `uv` is installed first. Then bootstrap the development environment, sync the locked dependencies, and install the Git hooks:

```bash
./setup.py
```

## Serve the docs

```bash
mkdocs serve
```

Then open:

```text
http://127.0.0.1:8000
```
