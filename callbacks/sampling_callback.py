"""Callback for generating and visualizing samples during training."""

import torch
import lightning as L
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
import logging
from inference import (
    KarrasDiffEq,
    KarrasHeun2Solver,
    KarrasNoiseSchedule,
    sample_trajectory_batch
)

logger = logging.getLogger(__name__)

class SamplingCallback(L.Callback):
    """Callback for generating and visualizing samples during training.
    
    This callback generates samples from the model at specified intervals
    and saves them as plots.
    """
    
    def __init__(
        self,
        sampling_config: Dict[str, Any],
        viz_config: Dict[str, Any],
        save_dir: str,
        sampling_interval: int = 50
    ):
        """Initialize the sampling callback.
        
        Args:
            sampling_config: Configuration for sampling parameters
            viz_config: Configuration for visualization parameters
            save_dir: Directory to save samples
            sampling_interval: Number of epochs between sample generations
        """
        super().__init__()
        self.sampling_config = sampling_config
        self.viz_config = viz_config
        self.save_dir = Path(save_dir)
        self.sampling_interval = sampling_interval
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Generate and save samples at the end of each training epoch if interval is reached.
        
        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        if (trainer.current_epoch + 1) % self.sampling_interval != 0:
            return
            
        logger.info(f"Generating samples at epoch {trainer.current_epoch + 1}")
        
        # Set model to eval mode
        pl_module.eval()
        
        # Create ODE and solver
        ode = KarrasDiffEq(pl_module.model)
        solver = KarrasHeun2Solver()
        
        # Create noise schedule
        noise_schedule = KarrasNoiseSchedule(
            sigma_data=pl_module.config.model[pl_module.config.model.type].sigma_data,
            sigma_min=pl_module.config.model[pl_module.config.model.type].sigma_min,
            sigma_max=pl_module.config.model[pl_module.config.model.type].sigma_max,
            rho=pl_module.config.model[pl_module.config.model.type].get('rho', 7.0)
        )
        
        # Generate samples
        with torch.no_grad():
            samples = sample_trajectory_batch(
                input_shape=(2,),  # For 2D data
                ode=ode,
                solver=solver,
                noise_schedule=noise_schedule,
                batch_size=self.sampling_config.batch_size,
                n_steps=self.sampling_config.n_steps,
                device=pl_module.device
            )
        
        # Plot and save samples
        self._plot_samples(samples, trainer.current_epoch + 1)
        
        # Set model back to training mode
        pl_module.train()
        
    def _plot_samples(self, samples: torch.Tensor, epoch: int) -> None:
        """Plot and save samples.
        
        Args:
            samples: Generated samples
            epoch: Current epoch number
        """
        # Get final samples
        final_samples = samples[-1].detach().cpu().numpy()
        
        # Create plot
        plt.figure(figsize=tuple(self.viz_config.figsize))
        plt.scatter(final_samples[:, 0], final_samples[:, 1])
        plt.title(f"Generated Samples (Epoch {epoch})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        
        # Save plot
        save_path = self.save_dir / f"samples_epoch_{epoch}.png"
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved samples to {save_path}") 