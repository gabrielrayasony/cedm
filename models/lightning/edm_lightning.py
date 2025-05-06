"""Lightning module for training EDM models."""

import torch
import lightning as L
from typing import Optional, Dict, Any, Union, Tuple
from models.edm import EDM
from training.losses import DiffusionLoss
import torch.nn as nn

class EDMLightning(L.LightningModule):
    """Lightning module for training EDM models.
    
    This module handles the training loop, loss computation, and optimization
    for EDM models. It takes the EDM model (which contains the network and
    preconditioning) and the loss function as inputs.
    
    Args:
        model: EDM model containing the network and preconditioning
        loss_fn: Loss function for training (handles noise sampling internally)
        config: Training configuration dictionary
    """
    
    def __init__(
        self,
        model: EDM,
        loss_fn: DiffusionLoss,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the EDM Lightning module.
        
        Args:
            model: EDM model instance containing the network and preconditioning
            loss_fn: Loss function for training (handles noise sampling internally)
            config: Training configuration dictionary
        """
        super().__init__()
        
        self.model = model
        self.loss_fn = loss_fn
        self.config = config or {}
        self.train_loss = []
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        
    def forward(self, x: torch.Tensor, sigma: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x, sigma, labels)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Tuple of (x, labels) where labels is optional
            batch_idx: Index of the current batch
            
        Returns:
            Loss value
        """
        x, _ = batch
        
        # Get model prediction and compute loss
        # The loss function will handle noise sampling internally
        loss = self.loss_fn(self.model, x, class_labels=None)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_loss.append(loss.item())
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step.
        
        Args:
            batch: Tuple of (x, labels) where labels is optional
            batch_idx: Index of the current batch
        """
        x, _ = batch
        
        # Get model prediction and compute loss
        # The loss function will handle noise sampling internally
        loss = self.loss_fn(self.model, x, None)
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
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