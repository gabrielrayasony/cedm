"""Lightning module for training EDM models."""

import torch
import lightning as L
from typing import Optional, Dict, Any, Tuple 
from utils.data_utils import expand_dims
from training.noise_samplers.factory import get_matching_sampler_and_loss
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class EDMLightning(L.LightningModule):
    """Lightning module for training EDM models.
    
    This module handles the training loop, loss computation, and optimization
    for EDM models. It takes the EDM model and configuration as inputs.
    
    Args:
        model: EDM model containing the network and preconditioning
        config: Training configuration dictionary
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any]
    ):
        """Initialize the EDM Lightning module.
        
        Args:
            model: EDM model instance containing the network and preconditioning
            config: Training configuration dictionary
        """
        super().__init__()
        
        self.denoiser = model
        self.config = config
        
        # Get matching noise sampler and loss function
        self.noise_sampler, self.loss_fn = get_matching_sampler_and_loss(config)
        
        # Store training history
        self.train_loss = []
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, x, sigma, class_labels=None, augment_labels=None):
        """Forward pass through the model."""
        return self.denoiser(x, sigma, class_labels, augment_labels)
    
    @torch.no_grad()
    def compute_score(self, x, sigma, class_labels=None, augment_labels=None):
        """Compute the score function from the denoising model.
        
        For EDM, the score function is computed as:
        score(x, sigma) = (D(x, sigma) - x) / sigma^2
        
        where D(x, sigma) is the denoising model output.
        
        Args:
            x: Input tensor
            sigma: Noise level tensor
            labels: Optional conditioning labels
            class_labels: Optional class conditioning labels
            
        Returns:
            Score function tensor
        """
        # Get denoised output from model
        denoised = self.denoiser(x, sigma, class_labels, augment_labels)
        
        # Compute score according to EDM formulation
        score = (denoised - x) / (expand_dims(sigma, x.ndim) ** 2)
        
        return score
    
    def training_step(self, batch: tuple, batch_idx: int):
        x, _ = batch
        
        # Sample noise levels
        sigmas = self.noise_sampler(x.shape[0], device=x.device)

        # Compute loss
        loss = self.loss_fn(self.denoiser, x, sigmas, class_labels=None)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_loss.append(loss.item())
            
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int):
        x, _ = batch
        
        # Sample noise levels
        sigmas = self.noise_sampler(x.shape[0], device=x.device)
        
        # Compute loss with pre-sampled sigmas
        loss = self.loss_fn(self.denoiser, x, sigmas, None)
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Get optimizer configuration
        optimizer_config = self.config.get('optimizer', {})
        lr = optimizer_config.get('learning_rate', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 0.0)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Get scheduler configuration
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('T_max', 1000),
                eta_min=scheduler_config.get('eta_min', 0.0)
            )
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        else:
            return optimizer
            
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss' if scheduler_type == 'reduce_on_plateau' else None
            }
        }
    
    def on_train_batch_end(self, outputs: torch.Tensor, batch: Tuple[torch.Tensor, Optional[torch.Tensor]], batch_idx: int) -> None:
        """Called at the end of each training batch."""
        # Log learning rate
        if self.trainer.is_global_zero:
            self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'])
            
    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch."""
        # Log model parameters
        if self.trainer.is_global_zero:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.log(f'params/{name}', param.norm().item())


# x = torch.randn(18, 2).to(device) #  Dummy data     
# sigma = torch.rand(18).to(device) #  Dummy noise

# m = model.model.to(device)
# m.forward(x, sigma)