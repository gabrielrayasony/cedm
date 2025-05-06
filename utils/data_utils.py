from torch import Tensor
from datamodules.image_datasets import CIFAR10DataModule
from datamodules.toy_datasets import ToyDataModule


def get_datamodule(dataset_name: str, num_samples: int = 15000, batch_size: int = 2000, num_workers: int = 4):
    if dataset_name == "cifar10":
        return CIFAR10DataModule(batch_size, num_workers)
    elif dataset_name == "checkerboard":
        return ToyDataModule(dataset_name, num_samples=num_samples, noise=0.0, batch_size=batch_size, num_workers=num_workers)
    elif dataset_name == "gaussian_mixture":
        return ToyDataModule(dataset_name, num_samples=num_samples, noise=0.0, batch_size=batch_size, num_workers=num_workers, min_max_normalize=True)
    elif dataset_name == "spirals":
        return ToyDataModule(dataset_name, num_samples=num_samples, noise=0.0, batch_size=batch_size, num_workers=num_workers) 
    elif dataset_name == "moons":
        return ToyDataModule(dataset_name, num_samples=num_samples, noise=0.01, batch_size=batch_size, num_workers=num_workers, min_max_normalize=True)
    elif dataset_name == "swiss_roll":
        return ToyDataModule(dataset_name, num_samples=num_samples, noise=0.5, batch_size=batch_size, num_workers=num_workers, min_max_normalize=True)
    elif dataset_name == "circles":
        return ToyDataModule(dataset_name, num_samples=num_samples, noise=0.0, batch_size=batch_size, num_workers=num_workers)
    elif dataset_name == "text-points":
        return ToyDataModule(dataset_name, num_samples=num_samples, noise=0.0, batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    
def expand_dims(x: Tensor, target_ndim: int) -> Tensor:
    """Expands the dimensions of a tensor to match a target number of dimensions.

    Args:
        x: Input tensor of shape [N].
        target_ndim: Target number of dimensions.

    Returns:
        Tensor of shape [N, 1, ..., 1] with target_ndim dimensions and the same values as x.
    """
    return x.reshape(x.shape + (1,) * (target_ndim - x.ndim))

