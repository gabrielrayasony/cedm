"""
Entropy Analysis Module for EDM

This module provides functionality for analyzing and visualizing entropy dynamics in EDM.
It includes tools for computing entropy estimates based on the model's predicted score function
and visualizing the results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from pathlib import Path
from utils.data_utils import expand_dims
from tqdm import tqdm

class EntropyAnalyzer:
    """Analyzer for computing and visualizing entropy dynamics in EDM."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        num_steps: int = 1000,
        device: str = "cuda"
    ):
        """Initialize entropy analyzer.
        
        Args:
            model: EDM model instance
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            num_steps: Number of steps for entropy computation
            device: Device to use for computation
        """
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_steps = num_steps
        self.device = device
        
        # Create noise schedule
        self.sigmas = torch.linspace(sigma_min, sigma_max, num_steps, device=device)
    
    def compute_time_derivative_entropy(self, x, t):
        """Compute the time derivative of the conditional entropy H(X_t | X_0). """
        x, t = x.to(self.device), t.to(self.device)
        _, D = x.shape

        # Get kernel components 
        signal_t = self.model.noise_sampler.signal(t)
        sigma_t = self.model.noise_sampler.sigma(t)
        sigma_t = expand_dims(sigma_t, 2)

        # Noisy input 
        noise = torch.randn_like(x, device=self.device)
        x_noised = x + noise * sigma_t

        # Get score function
        score = self.model.compute_score(x_noised, sigma_t)

        # Compute entropy estimate using score function
        mean_square_l2_norm_score = torch.mean(torch.sum(score, dim=1))
        entropy_derivative = D/t - signal_t**2 * sigma_t * mean_square_l2_norm_score 

        return entropy_derivative.mean()
            
    @torch.no_grad()
    def get_entropy_derivative(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None
    ):
        """Compute the time derivative of the conditional entropy H(X_t | X_0) over all timesteps."""
        entropy_derivatives = []

        with torch.no_grad():
            for t in tqdm(t, desc="Computing entropy derivatives"):
                t = torch.ones(x.shape[0], device=self.device) * t
                entropy_derivatives.append(self.compute_time_derivative_entropy(x, t))

        entropy_derivatives = torch.stack(entropy_derivatives)
        return entropy_derivatives
    
    @torch.no_grad()
    def compute_conditional_entropy(self, entropy_derivatives, timesteps):
        """Compute the conditional entropy H(X_t | X_0) by integrating the time derivative using the Trapezoidal Rule."""
        entropy_derivatives = entropy_derivatives.to(self.device)
        timesteps = timesteps.to(self.device)

        # Compute timestep differences
        dt = torch.diff(timesteps)

        # Apply numerical integration (Trapezoidal Rule)
        entropy_curve = torch.cumsum(0.5 * (entropy_derivatives[:-1] + entropy_derivatives[1:]) * dt, dim=0)
        # **Fix**: Append last entropy value to match `timesteps` length 
        return torch.cat([entropy_curve, entropy_curve[-1].unsqueeze(0)])
    
    def plot_entropy_over_time(
        self,
        times: torch.Tensor,
        entropies: List[float],
        save_path: Optional[str] = None,
        title: str = "Entropy Evolution",
        xlabel: str = "Noise Level (σ)",
        ylabel: str = "Entropy Estimate"
    ):
        """Create and save plot of entropy evolution over time.
        
        Args:
            times: Time points (sigma values)
            entropies: Entropy values
            save_path: Optional path to save the plot
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
        """
        plt.figure(figsize=(10, 6))
        plt.plot(times, entropies)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        else:
            plt.show()
        plt.close()
    
    def analyze_trajectory(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None
    ) -> Tuple[torch.Tensor, List[float]]:
        """Analyze entropy dynamics along a trajectory.
        
        Args:
            x: Input samples
            labels: Optional conditioning labels
            class_labels: Optional class conditioning labels
            save_dir: Optional directory to save plots
            
        Returns:
            Tuple of (sigma values, entropy values)
        """
        sigmas, entropies = self.compute_score_entropy(x, labels, class_labels)
        
        if save_dir:
            save_path = Path(save_dir) / "entropy_evolution.png"
            self.plot_entropy_over_time(sigmas, entropies, save_path=str(save_path))
            
        return sigmas, entropies 