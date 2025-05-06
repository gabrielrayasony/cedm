"""EDM score network implementation with comprehensive configuration options."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from networks.preconditioning import BasePrecond


class EDM(nn.Module):
    """Score network for EDM with comprehensive configuration options.
    
    This network implements the EDM framework with configurable:
    - Base model architecture
    - Preconditioning strategy 
    
    Can be initialized in two ways:
    1. Component injection (for advanced users):
        model = EDM(
            base_model=my_network,
            precond=my_precond
        )
        
    2. Configuration-based (for simpler use):
        model = EDM(
            base_model=base_model,
            precond=precond,
        )
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        precond: BasePrecond,
    ):
        """Initialize EDM model.
        
        Args:
            base_model: Base neural network for score prediction
            precond: Preconditioning scheme
            config: Configuration dictionary for component initialization
        """
        super().__init__()
        
        # Store components and configuration
        self.base_model = base_model
        self.precond = precond
            
    def forward(self, x: torch.Tensor, sigma: torch.Tensor, labels: Optional[torch.Tensor] = None, class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            sigma: Noise level tensor
            labels: Optional conditioning labels (alternative to class_labels)
            class_labels: Optional conditioning labels (alternative to labels)
            
        Returns:
            Predicted score/denoised output
        """
        # Use either labels or class_labels, preferring class_labels if both are provided
        conditioning = class_labels if class_labels is not None else labels
        
        # Get preconditioning coefficients
        c_skip, c_out, c_in, c_noise = self.precond(x, sigma)
        
        # Apply input preconditioning
        x_scaled = x * c_in
        
        # Get model prediction
        F_x = self.base_model(x_scaled, c_noise, conditioning)
        
        # Apply output preconditioning
        D_x = c_skip * x + c_out * F_x
        
        return D_x