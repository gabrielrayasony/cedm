"""Noise schedule implementations."""

import torch
import math
import numpy as np
from typing import Optional, Union, Tuple
from abc import ABC, abstractmethod

#------------------------------------------------------------------------------------------------
# Base class for noise samplers
#------------------------------------------------------------------------------------------------

class NoiseSampler(ABC):
    """Base class for noise level samplers.
    
    A noise sampler determines how noise levels (sigma values) are sampled during training.
    Each sampler can be configured with different sampling strategies and parameters.
    """
    
    def __init__(self,):
        pass
        
    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Given timesteps t in [0, 1], return corresponding noise levels σ(t)."""
        raise NotImplementedError
    
    @abstractmethod
    def signal(self, t: torch.Tensor) -> torch.Tensor:
        """Signal level as per the original paper."""
        raise NotImplementedError
        
    @abstractmethod
    def __call__(self, batch_size: int, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Sample a batch of noise levels (sigma values). Used for training.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to place the samples on
            
        Returns:
            Tensor of shape [batch_size] containing the sampled sigma values
        """
        pass
    

#------------------------------------------------------------------------------------------------
# Linear noise sampler
#------------------------------------------------------------------------------------------------

class LinearNoiseSampler(NoiseSampler):
    """Samples noise levels according to a linear schedule as per the original paper Solh D. et al and DDPM.

    The noise schedule is defined as:
    """
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, epsilon_t: float = 1e-5):
        """Initialize the linear noise sampler.
        
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_d = beta_max - beta_min
        self.epsilon_t = epsilon_t

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Convert timesteps to beta values."""
        t = torch.as_tensor(t).clamp(min=self.epsilon_t)
        return self.beta_min + t * (self.beta_d)
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Convert timesteps to sigma values for VP."""
        # Denormalize from [0, 1] to [1, epsilon_t]
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
    
    def signal(self, t: torch.Tensor) -> torch.Tensor:
        """VP Signal level as per the original paper.""" 
        return 1/((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp()).sqrt()
    
    def logsnr(self, t): 
        return torch.log(self.signal(t) ** 2) - torch.log(1 - self.signal(t) ** 2)
    
    def __call__(self, batch_size: int, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Sample noise levels according to a linear schedule.""" 
        rnd_uniform = torch.rand(batch_size, device=device)
        return self.sigma(1 + rnd_uniform * (self.epsilon_t - 1)) 
    

#------------------------------------------------------------------------------------------------
# Uniform noise sampler
#------------------------------------------------------------------------------------------------

class LogUniformNoiseSampler(NoiseSampler):
    """Samples noise levels uniformly in log space.
    
    This sampler is equivalent to sampling uniformly from the log of the noise levels,
    which is often a good default choice for training diffusion models.
    """
    
    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0):
        """Initialize the uniform noise sampler.
        
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
        """
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)
           
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        log_sigmas = t * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min
        return torch.exp(log_sigmas)
    
    def signal(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t)
        return torch.ones_like(t)
    
    def logsnr(self, t): 
        return - torch.log(self.sigma(t) ** 2)

    def __call__(self, batch_size: int, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Sample noise levels uniformly in log space."""
        rnd_uniform = torch.rand(batch_size, device=device)
        log_sigmas = rnd_uniform * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min
        return torch.exp(log_sigmas)


class LogNormalNoiseSampler(NoiseSampler):
    """Samples noise levels from a log-normal distribution."""
    
    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2):
        """Initialize the log-normal noise sampler.
        
        Args:
            mean: Mean of the log-normal distribution
            std: Standard deviation of the log-normal distribution
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_min = 0.002
        self.sigma_max = 80.0
        self.rho = 7.0

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map uniform t ∈ [0, 1] to log-normal sigma(t)
        where log(sigma) ~ N(P_mean, P_std^2)
        """
        normal_t = torch.erfinv(2 * t - 1) * np.sqrt(2)  # z ∼ N(0, 1)
        return (normal_t * self.P_std + self.P_mean).exp()
    
    # def sigma(self, t: torch.Tensor) -> torch.Tensor:
    #     """
    #     Map uniform t ∈ [0, 1] to log-normal sigma(t)
    #     where log(sigma) ~ N(P_mean, P_std^2)
    #     """
    #     sigmas = (self.sigma_min ** (1 / self.rho) +  t * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho
    #     return sigmas
    
    def signal(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t)
        return torch.ones_like(t)
    
    def logsnr(self, t): 
        return - 2 * torch.log(self.sigma(t))
    
    def __call__(self, batch_size: int, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Sample noise levels from a log-normal distribution."""
        rnd_normal = torch.randn(batch_size, device=device) # z ~ N(0, 1)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        return sigma
    
class CosineNoiseSampler(NoiseSampler):
    """Samples noise levels according to a cosine schedule."""
    
    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0, eps: float = 1e-3):
        """Initialize the cosine noise sampler.
        
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            eps: Small constant to prevent numerical issues
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps = eps
    
    def __call__(self, batch_size: int, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Sample noise levels according to a cosine schedule."""
        t = torch.rand(batch_size, device=device)
        alpha = torch.cos((t + self.eps) / (1 + self.eps) * math.pi / 2) ** 2
        return self.sigma_min + alpha * (self.sigma_max - self.sigma_min)

