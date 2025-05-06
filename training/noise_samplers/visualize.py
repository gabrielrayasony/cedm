"""Visualization utilities for noise samplers."""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Tuple
from .schedules import NoiseSampler, LinearNoiseSampler, CosineNoiseSampler


def plot_sigma_distributions(
    samplers: Dict[str, NoiseSampler],
    num_samples: int = 10000,
    device: Optional[str] = None,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
):
    """Plot the distribution of sigma values for different samplers.
    
    Args:
        samplers: Dictionary mapping sampler names to sampler instances
        num_samples: Number of samples to generate for each sampler
        device: Device to use for sampling
        figsize: Figure size for the plot
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=figsize)
    
    # Plot each sampler's distribution
    for name, sampler in samplers.items():
        sigmas = sampler.sample_sigmas(num_samples, device=device)
        sigmas = sigmas.cpu().numpy()
        
        # Plot histogram
        sns.histplot(
            sigmas,
            label=name,
            alpha=0.5,
            stat='density',
            bins=50
        )
    
    plt.xlabel('Sigma Value')
    plt.ylabel('Density')
    plt.title('Noise Level Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()


def plot_sigma_trajectories(
    samplers: Dict[str, NoiseSampler],
    num_steps: int = 100,
    device: Optional[str] = None,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
):
    """Plot the trajectory of sigma values over time for different samplers.
    
    Args:
        samplers: Dictionary mapping sampler names to sampler instances
        num_steps: Number of time steps to plot
        device: Device to use for sampling
        figsize: Figure size for the plot
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=figsize)
    
    # Generate time points
    t = torch.linspace(0, 1, num_steps, device=device)
    
    # Plot each sampler's trajectory
    for name, sampler in samplers.items():
        if isinstance(sampler, (CosineNoiseSampler, LinearNoiseSampler)):
            # For deterministic schedules, we can plot the exact trajectory
            sigmas = sampler.sample_sigmas(num_steps, device=device)
            plt.plot(t.cpu(), sigmas.cpu(), label=name, alpha=0.7)
        else:
            # For stochastic samplers, plot mean and std
            num_samples = 100
            all_sigmas = []
            for _ in range(num_samples):
                sigmas = sampler.sample_sigmas(num_steps, device=device)
                all_sigmas.append(sigmas)
            
            all_sigmas = torch.stack(all_sigmas)
            mean_sigmas = all_sigmas.mean(dim=0)
            std_sigmas = all_sigmas.std(dim=0)
            
            plt.plot(t.cpu(), mean_sigmas.cpu(), label=name, alpha=0.7)
            plt.fill_between(
                t.cpu(),
                (mean_sigmas - std_sigmas).cpu(),
                (mean_sigmas + std_sigmas).cpu(),
                alpha=0.2
            )
    
    plt.xlabel('Time Step')
    plt.ylabel('Sigma Value')
    plt.title('Noise Level Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()


def plot_forward_process(
    data: torch.Tensor,
    sampler: NoiseSampler,
    num_steps: int = 10,
    device: Optional[str] = None,
    figsize: tuple = (15, 5),
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """Visualize the forward process of adding noise to data.
    
    Args:
        data: Input data tensor of shape [batch_size, ...]
        sampler: Noise sampler to use
        num_steps: Number of noise levels to visualize
        device: Device to use for computation
        figsize: Figure size for the plot
        save_path: If provided, save the plot to this path
        title: Optional title for the plot
    """
    if device is not None:
        data = data.to(device)
    
    # Generate linearly spaced noise levels
    sigmas = get_linear_noise_levels(sampler, num_steps, device=device)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot original data
    plt.subplot(1, num_steps + 1, 1)
    plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), alpha=0.5)
    plt.title('Original Data')
    plt.grid(True, alpha=0.3)
    
    # Plot noisy data at each step
    for i, sigma in enumerate(sigmas):
        # Add noise
        noise = sampler.sample_noise(data.shape, device=device)
        noisy_data = data + sigma * noise
        
        # Plot
        plt.subplot(1, num_steps + 1, i + 2)
        plt.scatter(noisy_data[:, 0].cpu(), noisy_data[:, 1].cpu(), alpha=0.5)
        plt.title(f'σ = {sigma:.3f}')
        plt.grid(True, alpha=0.3)
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()


def plot_forward_process_animation(
    data: torch.Tensor,
    sampler: NoiseSampler,
    num_steps: int = 50,
    device: Optional[str] = None,
    figsize: tuple = (8, 8),
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """Create an animation of the forward process.
    
    Args:
        data: Input data tensor of shape [batch_size, ...]
        sampler: Noise sampler to use
        num_steps: Number of frames in the animation
        device: Device to use for computation
        figsize: Figure size for the plot
        save_path: If provided, save the animation to this path
        title: Optional title for the plot
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        raise ImportError("matplotlib.animation is required for animation. Please install matplotlib>=3.0.0")
    
    if device is not None:
        data = data.to(device)
    
    # Generate linearly spaced noise levels
    sigmas = get_linear_noise_levels(sampler, num_steps, device=device)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    x_min, x_max = data[:, 0].min().item(), data[:, 0].max().item()
    y_min, y_max = data[:, 1].min().item(), data[:, 1].max().item()
    padding = 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    def update(frame):
        sigma = sigmas[frame]
        noise = sampler.sample_noise(data.shape, device=device)
        noisy_data = data + sigma * noise
        
        scatter.set_offsets(noisy_data.cpu().numpy())
        ax.set_title(f'σ = {sigma:.3f}')
        return scatter,
    
    anim = FuncAnimation(
        fig, update, frames=num_steps,
        interval=100, blit=True
    )
    
    if title:
        plt.suptitle(title)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
    
    plt.show()


# Example usage:
if __name__ == "__main__":
    from .schedules import (
        UniformNoiseSampler,
        LogNormalNoiseSampler,
        CosineNoiseSampler,
        LinearNoiseSampler
    )
    
    # Create sample 2D data (e.g., two moons)
    from sklearn.datasets import make_moons
    import numpy as np
    
    # Generate two moons dataset
    data, _ = make_moons(n_samples=1000, noise=0.05)
    data = torch.from_numpy(data).float()
    
    # Create samplers
    samplers = {
        'Uniform': UniformNoiseSampler(),
        'LogNormal': LogNormalNoiseSampler(),
        'Cosine': CosineNoiseSampler(),
        'Linear': LinearNoiseSampler()
    }
    
    # Visualize forward process for each sampler
    for name, sampler in samplers.items():
        plot_forward_process(
            data, sampler,
            num_steps=5,
            title=f'Forward Process: {name} Sampler'
        )
        
        # Create animation
        plot_forward_process_animation(
            data, sampler,
            num_steps=50,
            title=f'Forward Process Animation: {name} Sampler'
        )
    
    # Plot distributions
    plot_sigma_distributions(samplers, num_samples=10000)
    
    # Plot trajectories
    plot_sigma_trajectories(samplers, num_steps=100) 