"""Loss functions for diffusion models."""

import torch
import math
from typing import Optional, Union, Dict, Any
from utils.data_utils import expand_dims


class DiffusionLoss:
    """Base class for diffusion model losses."""
    
    def __init__(self, **kwargs):
        """Initialize the loss function."""
        pass
       
    def get_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Get the loss weight for a given noise level.
        
        Args:
            sigma: Noise levels of shape [batch_size]
            
        Returns:
            Weights of shape [batch_size]
        """
        raise NotImplementedError
        
    def __call__(self, net: torch.nn.Module, images: torch.Tensor, sigmas: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the loss for a batch of images.
        
        Args:
            net: The network to evaluate
            images: Input images of shape [batch_size, ...]
            sigmas: Noise levels of shape [batch_size]
            class_labels: Optional class labels of shape [batch_size]
            
        Returns:
            Loss value
        """
        # Add noise to images
        noise = torch.randn_like(images) * expand_dims(sigmas, images.ndim)
        noisy_images = images + noise
        
        # Get network prediction
        pred = net(noisy_images, sigmas, class_labels)
        
        # Compute weighted loss
        weight = expand_dims(self.get_weight(sigmas), images.ndim)
        loss = weight * ((pred - images) ** 2)
        
        return loss.mean()


class EDMLoss(DiffusionLoss):
    """Loss function from the EDM paper.
    
    This implements the improved loss function proposed in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    
    def __init__(self, sigma_data: float = 0.5):
        """Initialize the EDM loss.
        
        Args:
            sigma_data: Data noise level
        """
        super().__init__(sigma_data=sigma_data)
        self.sigma_data = sigma_data
       
    def get_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Get the EDM loss weight for a given noise level."""
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
    
    
class VPLoss(DiffusionLoss):
    """Loss function for the variance preserving (VP) formulation.
    
    This implements the loss from the paper "Score-Based Generative Modeling
    through Stochastic Differential Equations".
    """
    
    def __init__(self):
        """Initialize the VP loss."""
        super().__init__()
    
    def get_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Get the VP loss weight for a given noise level."""
        return 1 / sigma ** 2
    
    def sigma_to_timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma values to timesteps for VP.
        
        For VP, we solve the quadratic equation for t.
        """
        # Convert to the form: at^2 + bt + c = 0
        a = 0.5 * self.beta_d
        b = self.beta_min
        c = -torch.log(sigma ** 2 + 1)
        
        # Solve quadratic equation
        t = (-b + torch.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        # Normalize to [0, 1]
        t = (t - 1) / (self.epsilon_t - 1)
        return t.clamp(0, 1)


class VELoss(DiffusionLoss):
    """Loss function for the variance exploding (VE) formulation.
    
    This implements the loss from the paper "Score-Based Generative Modeling
    through Stochastic Differential Equations".
    """
    
    def __init__(self):
        """Initialize the VE loss."""
        super().__init__()
    
    def get_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Get the VE loss weight for a given noise level."""
        return 1 / sigma ** 2
    
    def sigma_to_timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma values to timesteps for VE."""
        return torch.log(sigma / self.sigma_min) / torch.log(self.sigma_max / self.sigma_min)
    
    def timestep_to_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Convert timesteps to sigma values for VE."""
        return self.sigma_min * ((self.sigma_max / self.sigma_min) ** t)


def get_loss_fn(config: Dict[str, Any]) -> DiffusionLoss:
    """Create a loss function based on preconditioning type.
    
    Args:
        config: Dictionary containing configuration parameters. Must include:
            - precond: Dictionary with preconditioning configuration
                - type: One of ['edm', 'vp', 've']
                - Additional parameters specific to each type
    
    Returns:
        An instance of the specified loss function
        
    Raises:
        ValueError: If loss type doesn't match preconditioning type
    """
    # Get preconditioning type
    precond_type = config.precond.name
    
    # Ensure loss type matches preconditioning type
    if config.loss.name.lower() != precond_type:
        raise ValueError(
            f"Loss type {config.loss.name} does not match preconditioning type {precond_type}. "
            f"Loss type must match preconditioning type."
        )
    
    # Create loss function based on preconditioning type
    if precond_type == 'edm':
        return EDMLoss(
            sigma_data=config['precond'].get('sigma_data', 0.5)
        )
    elif precond_type == 'vp':
        return VPLoss()
    elif precond_type == 've':
        return VELoss()
    else:
        raise ValueError(f"Invalid preconditioning type: {precond_type}. Must be one of ['edm', 'vp', 've']")


# Test loss function 
# loss_fn = get_loss_fn(config['loss'])
# x = x.to(device)
# loss_fn(net.to(device), x, None)