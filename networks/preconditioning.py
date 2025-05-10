"""Preconditioning schemes for diffusion models.

Partially based on the implementation from https://github.com/alshedivat/diffusion-playground/blob/main/diffusion/denoisers.py

"""
import abc
import torch
import torch.nn as nn
from typing import Union, Tuple
from utils.data_utils import expand_dims

# -----------------------------------------------------------------------------
# Denoising models.  
# -----------------------------------------------------------------------------
# Classes deined below are used to wrap trainable models, precondition inputs
# and outputs, and define a consistent API used at training and inference time.
# Preconditioning helps improve the dynamics of training (Karras et al., 2022).
# -----------------------------------------------------------------------------


class Denoiser(nn.Module):
    """Abstract base class for denoising models.
       This essentially implements the forward pass and preconditioning."""

    @abc.abstractmethod
    def _c_in(self, sigma, sigma_data=None) -> torch.Tensor:
        """Preconditioning coefficient for the input."""
        pass

    @abc.abstractmethod
    def _c_out(self, sigma, sigma_data=None) -> torch.Tensor:
        """Preconditioning coefficient for the output."""
        pass
    
    @abc.abstractmethod
    def _c_skip(self, sigma, sigma_data=None) -> torch.Tensor:
        """Preconditioning coefficient for the skip connection."""
        pass
    
    def round_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Round sigma to supported values."""
        return torch.clamp(sigma, self.sigma_min, self.sigma_max)

    @abc.abstractmethod
    def forward(self, input, sigma, **kwargs):
        """Computes the denoised output for a given input and noise level.
        
        Args:
            x: Input tensor
            sigma: Noise level tensor
            **kwargs: Additional arguments
        Returns:
            Denoiser Prediction using the Preconditioning Coefficients c_skip, c_out, c_in, c_noise
        """ 
        

class EDMDenoiser(Denoiser):
    """EDM denoising model."""
    def __init__(self, model: nn.Module, sigma_data: float = 0.5):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        
    def _c_in(self, sigma):
        return 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
    
    def _c_out(self, sigma):
        return sigma / (sigma ** 2 + self.sigma_data ** 2).sqrt()
    
    def _c_skip(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
    
    def _c_noise(self, sigma):
        return sigma.log() / 4
    
    def forward(self, x, sigma, class_labels=None, augment_labels=None):
        sigma = sigma.to(torch.float32)
        
        c_skip = expand_dims(self._c_skip(sigma), x.ndim)
        c_out = expand_dims(self._c_out(sigma), x.ndim)
        c_in = expand_dims(self._c_in(sigma), x.ndim)
        c_noise =self._c_noise(sigma)
        
        return c_skip * x + c_out * self.model(x * c_in, c_noise, class_labels, augment_labels)

#TODO: Adapt the following preconditioning schemes to the new setting
class VPPrecond(Denoiser):
    """Variance Preserving (VP) preconditioning scheme."""
    
    def __init__(
        self,
        beta_d: float = 19.9,
        beta_min: float = 0.1,
        M: int = 1000,
        epsilon_t: float = 1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Convert t to sigma."""
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
        
    def sigma_inv(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to t."""
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d
        
    def forward(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma = expand_dims(sigma.to(torch.float32), x.ndim)
        
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)
        
        return c_skip, c_out, c_in, c_noise

class VEPrecond(Denoiser):
    """Variance Exploding (VE) preconditioning scheme."""
    
    def forward(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma = expand_dims(sigma.to(torch.float32), x.ndim)
        
        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()
        
        return c_skip, c_out, c_in, c_noise

class iDDPMPrecond(Denoiser):
    """Improved DDPM (iDDPM) preconditioning scheme."""
    
    def __init__(
        self,
        C_1: float = 0.001,
        C_2: float = 0.008,
        M: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        
        # Precompute u values
        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        
    def alpha_bar(self, j: Union[int, torch.Tensor]) -> torch.Tensor:
        """Compute alpha_bar value."""
        j = torch.as_tensor(j)
        return (0.5 * torch.pi * j / self.M / (self.C_2 + 1)).sin() ** 2
        
    def forward(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)
        
        return c_skip, c_out, c_in, c_noise
        
    def round_sigma(self, sigma: torch.Tensor, return_index: bool = False) -> torch.Tensor:
        """Round sigma to nearest u value."""
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), 
                          self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()]
        return result.reshape(sigma.shape).to(sigma.device) 