## CEDM: Training Diffusion Models with Conditional Entropy for Efficiency Training and Sampling<br><sub>Official PyTorch implementation of the CEDM paper</sub>


Steps: 

1. Implmement EDM  EDM Framework in PyTorch Lightning
2. Add our entropic sampler for efficient training 


# Modular EDM Framework in PyTorch Lightning

This repository is intended to build a modular and research-friendly implementation of Elucidated Diffusion Models (EDM) using PyTorch Lightning. The goal is to provide a clear and extensible framework for studying the design space of diffusion models, starting from controlled experiments on low-dimensional 2D datasets.

## Objective

The initial phase of this project focuses on implementing EDM for 2D toy datasets to validate and study the impact of:

- Preconditioning strategies,
- Score network architectures (e.g., MLP vs. UNet),
- Sampler configurations (e.g., ancestral, DDIM, Heun),
- Entropy-based training schemes.

This setup enables efficient experimentation and visualization prior to scaling to high-dimensional image data.

## Implementation Plan

The implementation is expected to follow a modular structure:

### `networks/`
- `edm_score_net.py`: Score-based models with EDM-compatible input/output normalization.
- Support for multiple architectures.

### `preconditioning/`
- EDM-specific input and output scaling functions.
- σ-dependent rescaling and embedding logic.

### `samplers/`
- `heun_sampler.py`, `ddim.py`, `ancestral.py`: Samplers with unified interfaces.
- Support both EDM and DDPM-style schedules.

### `sde/`
- `edm_sde.py`: EDM-specific continuous σ schedule with σ_min, σ_max, and EDM-specific sampling noise levels.

### `datasets/`
- `toy_data.py`: Generate 2D toy datasets including:
  - Two moons,
  - Swiss roll,
  - Gaussian mixtures.

### `training/`
- `lightning_module.py`: Wrap model, loss, optimizer, and score function.
- Integrated with PyTorch Lightning's trainer and hooks.

### `evaluation/`
- Plotting utilities for:
  - Sample trajectories,
  - Entropy and score norm evolution,
  - Wasserstein/MMD distances (optional).

## Experimental Procedure

Each experiment will consist of a configuration specifying:
- Dataset and model architecture,
- Sampler and noise schedule,
- Preconditioning type,
- Entropy-based vs. uniform time sampling.

The training loop should support:
- Warmup phase with standard uniform training,
- Optional conditional entropy estimation,
- CDF-based time sampling.

## Suggested Workflow

1. Implement EDM schedule and MLP score network for 2D.
2. Validate forward noise process and sampler (e.g., Heun).
3. Add entropy estimation and sampling from estimated CDF.
4. Compare sample quality and training dynamics with standard methods.
5. Visualize entropy profiles, loss curves, and generated samples.

## Requirements

- Python 3.8+
- PyTorch >= 2.0
- PyTorch Lightning >= 2.0
- NumPy, Matplotlib, SciPy

## Getting Started

To begin, start with a simple MLP score network trained with EDM sampling on a 2D dataset. Confirm sample quality before extending the architecture or integrating entropy-aware scheduling.

---

For any questions or contributions, please follow standard practices and document each module clearly.
