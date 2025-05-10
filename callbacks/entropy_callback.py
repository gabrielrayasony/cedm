"""Callback for entropy-based sampling in EDM training."""

import lightning as L
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Optional, Dict
from samplers.entropy_analysis import EntropyAnalyzer
from samplers.edm_entropic_sampler import EDMEntropicSampler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class EntropyCallback(L.Callback):
    """Callback for entropy-based sampling in EDM training.
    
    This callback:
    1. Computes and tracks conditional entropy during training
    2. Can switch to entropy-based sampling at a specified epoch
    3. Visualizes entropy evolution over time
    """
    
    def __init__(
        self,
        interval: int = 5,
        num_timesteps: int = 100,
        save_dir: Optional[Path] = None,
        switch_to_entropic: bool = False,
        switch_epoch: Optional[int] = None,
        entropic_config: Optional[Dict] = None
    ):
        """Initialize the entropy callback."""
        self.interval = interval                        # How often to compute entropy
        self.num_timesteps = num_timesteps              # Number of timesteps to compute entropy for
        self.save_dir = save_dir                        # Directory to save entropy history
        self.switch_to_entropic = switch_to_entropic    # Whether to switch to entropic sampling
        self.switch_epoch = switch_epoch                # Epoch to switch to entropic sampling
        self.entropic_config = entropic_config or {}    # Entropic configuration
        
        # Initialize entropy analyzer containing entropy related
        self.entropy_analyzer = EntropyAnalyzer(
            sigma_min=entropic_config.get('sigma_min', 0.002),
            sigma_max=entropic_config.get('sigma_max', 80.0),
            num_steps=num_timesteps
        )
        
        # Initialize entropy related attributes
        self.entropy = None # conditional entropy
        self.cdf = None # cumulative distribution function
        self.inverse_cdf = None # inverse cumulative distribution function

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
        
        self.timesteps = self._create_timesteps()

    def _create_timesteps(self) -> torch.Tensor:
        """Create timesteps for entropy computation. """
        return torch.linspace(1e-3, 1 - 1e-3, self.num_timesteps)
    
    def _compute_cdf(self, entropy: torch.Tensor) -> torch.Tensor:
        """Compute the CDF h(σ) by normalizing the entropy."""
        logger.debug(f"Computing CDF from entropy: shape={entropy.shape}, requires_grad={entropy.requires_grad}")
        
        # Ensure entropy is non-negative
        entropy = torch.clamp(entropy, min=0)
        logger.debug(f"Clamped entropy: min={entropy.min().item():.4f}, max={entropy.max().item():.4f}")
        
        # Normalize to [0, 1] by subtracting min and dividing by range
        entropy_min = entropy.min()
        entropy_range = entropy.max() - entropy_min
        cdf = (entropy - entropy_min) / entropy_range
        logger.debug(f"Normalized CDF: min={cdf.min().item():.4f}, max={cdf.max().item():.4f}")
        
        # Ensure monotonicity
        cdf = torch.cummax(cdf, dim=0)[0]
        logger.debug(f"Monotonic CDF: min={cdf.min().item():.4f}, max={cdf.max().item():.4f}")
        
        # Ensure first value is 0 and last value is 1
        cdf[0] = 0.0
        cdf[-1] = 1.0
        logger.debug(f"Final CDF: min={cdf.min().item():.4f}, max={cdf.max().item():.4f}")
        
        return cdf
    
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Compute entropy estimate at the end of each interval."""
        if (trainer.current_epoch + 1) == self.interval:
            logger.info(f"Computing entropy estimate at epoch {trainer.current_epoch + 1}")
            
            # Set model to eval mode
            pl_module.eval()
            
            # Get a batch of data
            batch = next(iter(trainer.datamodule.train_dataloader()))
            x, _ = batch
            
            # Create timesteps using the new method
            timesteps = self.timesteps.to(pl_module.device)

            # Compute entropy derivatives
            entropy_derivatives = self.entropy_analyzer.get_entropy_derivative(pl_module, x, timesteps)
            logger.debug(f"Entropy derivatives: min={entropy_derivatives.min().item():.4f}, max={entropy_derivatives.max().item():.4f}")
            
            # Compute conditional entropy
            self.entropy = self.entropy_analyzer.compute_conditional_entropy(entropy_derivatives, timesteps)
            logger.debug(f"Conditional entropy: min={self.entropy.min().item():.4f}, max={self.entropy.max().item():.4f}")
            
            # Compute CDF
            self.cdf = self._compute_cdf(self.entropy)
            logger.debug(f"CDF: min={self.cdf.min().item():.4f}, max={self.cdf.max().item():.4f}")
            
            # Plot and save entropy history
            if self.save_dir is not None:
                self._plot_entropy_history(trainer.current_epoch + 1)
            
            # Check if we should switch to entropic sampler
            if self.switch_to_entropic and trainer.current_epoch + 1 >= self.switch_epoch:
                self._switch_to_entropic_sampler(pl_module)
        
            # Set model back to training mode
            pl_module.train()
    
    def _switch_to_entropic_sampler(self, pl_module: L.LightningModule) -> None:
        """Switch the model's noise sampler to entropic sampler."""
        if not hasattr(pl_module, 'noise_sampler'):
            logger.warning("Model has no noise_sampler attribute")
            return
            
        if isinstance(pl_module.noise_sampler, EDMEntropicSampler):
            logger.info("Already using entropic sampler")
            return
            
        logger.info("Switching to entropic noise sampler")
        
        # Create entropic sampler using current sampler as base
        entropic_sampler = EDMEntropicSampler(
            base_sampler=pl_module.noise_sampler,
            cdf=self.cdf,
            timesteps=self.timesteps,
            device=pl_module.device
        )
        
        # Update model's noise sampler
        pl_module.noise_sampler = entropic_sampler
        
        # Log the switch
        pl_module.log('noise_sampler_switch', 1.0, on_step=False, on_epoch=True)
    
    def _plot_entropy_history(self, epoch: int) -> None:
        """Plot entropy history."""
        if len(self.entropy) == 0 or len(self.cdf) == 0:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot entropy
        ax.plot(self.entropy.cpu().numpy(), label='Entropy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Entropy')
        ax.set_title('Entropy History')
        ax.legend()
        
        # Save plot
        plt.savefig(self.save_dir / f'entropy_{epoch}.png')
        plt.close()
        
        # Plot CDF
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.cdf.cpu().numpy(), label='CDF')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('CDF')
        ax.set_title('CDF History')
        ax.legend()
        
        # Save plot
        plt.savefig(self.save_dir / f'cdf_{epoch}.png')
        plt.close()