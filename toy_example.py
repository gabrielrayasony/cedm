"""Toy example for training and sampling from an EDM model on 2D data.

This script demonstrates:
1. Loading and visualizing toy data
2. Training an EDM model
3. Sampling from the trained model
4. Visualizing the results
"""

import torch
import matplotlib.pyplot as plt
from datamodules.toy_datasets import ToyDataModule
from utils.plots import plot_2d_data
from networks.toy_network import MLP
from models.edm import EDM
from models.lightning.edm_lightning import EDMLightning
from training.losses import get_loss_fn, DiffusionLoss
from inference import (
    KarrasDiffEq,
    KarrasHeun2Solver,
    KarrasNoiseSchedule,
    sample_trajectory_batch
)
import lightning as L
from typing import Dict, Any, Tuple, Optional
import logging
from utils.model_utils import get_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_default_config() -> Dict[str, Any]:
    """Get default configuration for the EDM model and training.
    
    Returns:
        Dictionary containing model, training, and sampling configuration
    """
    return {
        'model': {
            'type': 'mlp',
            'hidden_dim': 64,
            'num_hidden_layers': 6,
            'time_embedding_dim': 8
        },
        'loss': {
            'type': 'edm',
            'P_mean': -1.2,
            'P_std': 1.2
        },
        'sampler': {
            'type': 'log_normal',
            'P_mean': -1.2,
            'P_std': 1.2
        },
        'precond': {
            'type': 'edm',
            'sigma_min': 0.02,
            'sigma_max': 100,
            'rho': 7.0,
            'sigma_data': 0.5
        },
        'train': {
            'learning_rate': 1e-3,
            'max_epochs': 1500,
            'batch_size': 512,
            'num_workers': 0,
            'sampling_interval': 50  # Generate samples every 50 epochs
        },
        'sampling': {
            'batch_size': 500,
            'n_steps': 50
        }
    }

def setup_data(config: Dict[str, Any]) -> Tuple[ToyDataModule, torch.Tensor]:
    """Set up the toy dataset and get a sample batch.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (data module, sample batch)
    """
    logger.info("Setting up toy dataset...")
    
    # Create data module
    toy_dm = ToyDataModule(
        dataset_name="swiss_roll",
        num_samples=10000,
        noise=0.05,
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        min_max_normalize=True
    )
    
    # Prepare data
    toy_dm.prepare_data()
    toy_dm.setup()
    
    # Get sample batch
    dataloader = toy_dm.train_dataloader()
    x, _ = next(iter(dataloader))
    
    # Plot toy data
    plot_2d_data(x.numpy(), name="toy_data", figsize=(3, 3))
    
    return toy_dm, x

def plot_training_samples(model: EDM, config: Dict[str, Any], device: str, epoch: int) -> None:
    """Plot samples during training.
    
    Args:
        model: EDM model
        config: Configuration dictionary
        device: Device to use for sampling
        epoch: Current epoch number
    """
    logger.info(f"Generating samples at epoch {epoch}...")
    
    # Set model to eval mode
    model.eval()
    
    # Create ODE and solver
    ode = KarrasDiffEq(model)
    solver = KarrasHeun2Solver()
    
    # Create noise schedule
    noise_schedule = KarrasNoiseSchedule(
        sigma_data=config['precond']['sigma_data'],
        sigma_min=config['precond']['sigma_min'],
        sigma_max=config['precond']['sigma_max'],
        rho=config['precond']['rho']
    )
    
    # Generate samples
    samples = sample_trajectory_batch(
        input_shape=(2,),
        ode=ode,
        solver=solver,
        noise_schedule=noise_schedule,
        batch_size=config['sampling']['batch_size'],
        n_steps=config['sampling']['n_steps'],
        device=device
    )
    
    # Plot samples
    plt.figure(figsize=(5, 5))
    final_samples = samples[-1].detach().cpu().numpy()
    plt.scatter(final_samples[:, 0], final_samples[:, 1])
    plt.title(f"Generated Samples (Epoch {epoch})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig(f"samples_epoch_{epoch}.png")
    plt.close()
    
    # Set model back to training mode
    model.train()

class EDMLightningWithSampling(EDMLightning):
    """Extension of EDMLightning that generates samples during training."""
    
    def __init__(
        self,
        model: EDM,
        loss_fn: DiffusionLoss,
        config: Optional[Dict[str, Any]] = None,
        sampling_interval: int = 50
    ):
        """Initialize the Lightning module with sampling.
        
        Args:
            model: EDM model
            loss_fn: Loss function
            config: Configuration dictionary
            sampling_interval: Number of epochs between sample generations
        """
        super().__init__(model=model, loss_fn=loss_fn, config=config)
        self.sampling_interval = sampling_interval
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        super().on_train_epoch_end()
        
        # Generate samples every sampling_interval epochs
        if (self.current_epoch + 1) % self.sampling_interval == 0:
            plot_training_samples(
                self.model,
                self.config,
                self.device,
                self.current_epoch + 1
            )

def setup_model(config: Dict[str, Any]) -> Tuple[EDM, EDMLightning]:
    """Set up the EDM model and Lightning module.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (EDM model, Lightning module)
    """
    logger.info("Setting up EDM model...")
    
    # Create EDM model directly from config
    model = get_model(config=config)
    
    # Create loss function
    loss_fn = get_loss_fn(config['loss'])
    
    # Create Lightning module with sampling
    lightning_model = EDMLightningWithSampling(
        model=model,
        loss_fn=loss_fn,
        config=config,
        sampling_interval=config['train'].get('sampling_interval', 50)
    )
    
    return model, lightning_model

def train_model(model: EDMLightning, data_module: ToyDataModule, config: Dict[str, Any]) -> None:
    """Train the model using PyTorch Lightning.
    
    Args:
        model: Lightning module
        data_module: Data module
        config: Configuration dictionary
    """
    logger.info("Training model...")
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=config['train']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto"
    )
    
    # Train model
    trainer.fit(model, data_module)

def sample_from_model(model: EDM, config: Dict[str, Any], device: str) -> torch.Tensor:
    """Sample from the trained model.
    
    Args:
        model: Trained EDM model
        config: Configuration dictionary
        device: Device to use for sampling
        
    Returns:
        Generated samples
    """
    logger.info("Sampling from model...")
    
    # Set model to eval mode
    denoiser = model.eval()
    
    # Create ODE and solver
    ode = KarrasDiffEq(denoiser)
    solver = KarrasHeun2Solver()
    
    # Create noise schedule
    noise_schedule = KarrasNoiseSchedule(
        sigma_data=config['precond']['sigma_data'],
        sigma_min=config['precond']['sigma_min'],
        sigma_max=config['precond']['sigma_max'],
        rho=config['precond']['rho']
    )
    
    # Generate samples
    samples = sample_trajectory_batch(
        input_shape=(2,),
        ode=ode,
        solver=solver,
        noise_schedule=noise_schedule,
        batch_size=config['sampling']['batch_size'],
        n_steps=config['sampling']['n_steps'],
        device=device
    )
    
    return samples

def plot_samples(samples: torch.Tensor, config: Dict[str, Any]) -> None:
    """Plot the generated samples.
    
    Args:
        samples: Generated samples
        config: Configuration dictionary
    """
    logger.info("Plotting samples...")
    
    plt.figure(figsize=(5, 5))
    final_samples = samples[-1].detach().cpu().numpy()
    plt.scatter(final_samples[:, 0], final_samples[:, 1])
    plt.title("Generated 2D Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def main():
    """Main function to run the toy example."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Get configuration
    config = get_default_config()
    
    try:
        # Set up data
        toy_dm, x = setup_data(config)
        
        # Set up model
        model, lightning_model = setup_model(config)
        
        # Train model
        train_model(lightning_model, toy_dm, config)
        
        # Sample from model
        samples = sample_from_model(model, config, device)
        
        # Plot samples
        plot_samples(samples, config)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()