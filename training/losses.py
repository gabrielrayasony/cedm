"""Loss functions for diffusion models."""

import torch
import math
from typing import Optional, Union, Dict, Any
from utils.data_utils import expand_dims
from training.noise_samplers.schedules import NoiseSampler, LogNormalNoiseSampler, LinearNoiseSampler, LogUniformNoiseSampler


class DiffusionLoss:
    """Base class for diffusion model losses."""
    
    def __init__(self, **kwargs):
        pass
       
    def get_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Get the loss weight for a given noise level.
        
        Args:
            sigma: Noise levels of shape [batch_size]
            
        Returns:
            Weights of shape [batch_size]
        """
        raise NotImplementedError
        
    def __call__(self, net: torch.nn.Module, images: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the loss for a batch of images.
        
        Args:
            net: The network to evaluate
            images: Input images of shape [batch_size, ...]
            class_labels: Optional class labels of shape [batch_size]
            
        Returns:
            Loss value
        """
        # Sample noise levels
        sigma = self.noise_sampler(images.shape[0], device=images.device)
        sigma = expand_dims(sigma, images.ndim)

        # Add noise to images
        noise = torch.randn_like(images) * sigma
        noisy_images = images + noise
        # print("DEBUG: ", noisy_images.shape, images.shape, class_labels)
        # Get network prediction
        pred = net(noisy_images, sigma, class_labels)
        
        # Compute weighted loss
        weight = self.get_weight(sigma)
        loss = weight * ((pred - images) ** 2)
        
        return loss.mean()


class EDMLoss(DiffusionLoss):
    """Loss function from the EDM paper.
    
    This implements the improved loss function proposed in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    
    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5):
        """Initialize the EDM loss.
        
        Args:
            P_mean: Mean of the log-normal distribution
            P_std: Standard deviation of the log-normal distribution
            sigma_data: Data noise level
        """
        # Store parameters in config dictionary for parent class
        config = {
            'P_mean': P_mean,
            'P_std': P_std,
            'sigma_data': sigma_data
        }
        super().__init__(**config)
        
        # Store parameters as instance variables
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.noise_sampler = LogNormalNoiseSampler(P_mean=P_mean, P_std=P_std)
       
    def get_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Get the EDM loss weight for a given noise level."""
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
    
    
class VPLoss(DiffusionLoss):
    """Loss function for the variance preserving (VP) formulation.
    
    This implements the loss from the paper "Score-Based Generative Modeling
    through Stochastic Differential Equations".
    """
    
    def __init__(self, beta_d: float = 19.9, beta_min: float = 0.1, epsilon_t: float = 1e-5):
        """Initialize the VP loss.
        
        Args:
            beta_d: Maximum noise level
            beta_min: Minimum noise level
            epsilon_t: Small constant to prevent numerical issues
        """
        super().__init__(beta_d=beta_d, beta_min=beta_min, epsilon_t=epsilon_t)
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.noise_sampler = LinearNoiseSampler()
    
    def sample_noise_levels(self, batch_size: int, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Sample noise levels according to VP schedule."""
        rnd_uniform = torch.rand(batch_size, device=device)
        t = 1 + rnd_uniform * (self.epsilon_t - 1)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
    
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
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Convert timesteps to sigma values for VP."""
        # Denormalize from [0, 1] to [1, epsilon_t]
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
    
    def signal(self, t: torch.Tensor) -> torch.Tensor:
        """VP Signal level as per the original paper.""" 
        return 1/((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp()).sqrt()


class VELoss(DiffusionLoss):
    """Loss function for the variance exploding (VE) formulation.
    
    This implements the loss from the paper "Score-Based Generative Modeling
    through Stochastic Differential Equations".
    """
    
    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 100):
        """Initialize the VE loss.
        
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
        """
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.noise_sampler = LogUniformNoiseSampler()
    
    def sample_noise_levels(self, batch_size: int, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Sample noise levels according to VE schedule."""
        rnd_uniform = torch.rand(batch_size, device=device)
        return self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
    
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
            P_mean=config['precond'].get('P_mean', -1.2),
            P_std=config['precond'].get('P_std', 1.2),
            sigma_data=config['precond'].get('sigma_data', 0.5)
        )
    elif precond_type == 'vp':
        return VPLoss(
            beta_d=config['precond'].get('beta_d', 19.9),
            beta_min=config['precond'].get('beta_min', 0.1),
            epsilon_t=config['precond'].get('epsilon_t', 1e-5)
        )
    elif precond_type == 've':
        return VELoss(
            sigma_min=config['precond'].get('sigma_min', 0.02),
            sigma_max=config['precond'].get('sigma_max', 100)
        )
    else:
        raise ValueError(f"Invalid preconditioning type: {precond_type}. Must be one of ['edm', 'vp', 've']")


# Test loss function 
# loss_fn = get_loss_fn(config['loss'])
# x = x.to(device)
# loss_fn(net.to(device), x, None)