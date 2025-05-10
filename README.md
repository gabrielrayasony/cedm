# CEDM: Conditional Entropy Diffusion Models for Efficient Training and Sampling

Steps: 

1. Implmement EDM  EDM Framework in PyTorch Lightning
2. Add our entropic sampler for efficient training 



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

### EDM-2D SwissRoll
```bash
python main.py network=mlp dataset=swiss_roll train.batch_size=512
```

### CEDM-2D SwissRoll
```bash
python main.py network=mlp dataset=swiss_roll train.batch_size=512 train.entropic_sampler=True
```


### EDM CIFAR10
```bash
python main.py network=ddpmpp dataset=cifar10 train.batch_size=128
```

---

For any questions or contributions, please follow standard practices and document each module clearly.
