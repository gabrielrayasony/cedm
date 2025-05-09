import torch
import numpy as np
from scipy.interpolate import PchipInterpolator
from typing import Optional, Tuple, Callable
from models.entropy_analysis import EntropyAnalyzer
from training.noise_samplers.schedules import NoiseSampler
import logging

logger = logging.getLogger(__name__)

class EDMEntropicSampler(NoiseSampler):
    """Entropy-based sampler for EDM parameterization.
    
    This sampler uses conditional entropy to guide the sampling of noise levels (σ).
    It requires the cdf values and then uses it to sample σ values
    that are more likely to be in regions of high entropy.
    """
    
    def __init__(
        self,
        base_sampler: NoiseSampler,
        cdf: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        device: Optional[torch.device] = None
    ):
        """Initialize the EDM entropy-based sampler.
        
        Args:
            base_sampler: Base noise sampler to inherit abstract methods from
            cdf: Cumulative distribution function values
            timesteps: Time steps corresponding to CDF values
            device: Device to use for computation
        """
        super().__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_sampler = base_sampler
        logger.debug(f"Initializing sampler on device: {self.device}")
        
        if cdf is None or timesteps is None:
            raise ValueError("cdf and timesteps must be provided")
        
        # Initialize grid and entropy-related attributes
        self.time_steps = timesteps 
        self.cdf = cdf
        self.inverse_cdf = None
        
        # Build inverse CDF on initialization
        self._build_inverse_cdf(self.time_steps, self.cdf)
        
    def _build_inverse_cdf(self, time_steps: torch.Tensor, cdf: torch.Tensor):
        """Build the inverse CDF interpolator."""
        logger.debug(f"Building inverse CDF: time_steps shape={time_steps.shape}, cdf shape={cdf.shape}")
        logger.debug(f"Time steps range: [{time_steps.min().item():.4f}, {time_steps.max().item():.4f}]")
        logger.debug(f"CDF range: [{cdf.min().item():.4f}, {cdf.max().item():.4f}]")
        
        # Convert to numpy for scipy interpolation
        time_steps_np = time_steps.cpu().numpy()
        cdf_np = cdf.cpu().numpy()
        
        # Ensure CDF starts at 0 and ends at 1
        cdf_np[0] = 0.0
        cdf_np[-1] = 1.0
        
        # Add small epsilon to cdf to avoid zero values
        eps = np.finfo(cdf_np.dtype).eps
        cdf_np = cdf_np + np.arange(len(cdf_np)) * eps
        logger.debug(f"CDF after epsilon: min={cdf_np.min():.4f}, max={cdf_np.max():.4f}")
        
        # Create monotone interpolator
        self.inverse_cdf = PchipInterpolator(cdf_np, time_steps_np)
        logger.debug("Built inverse CDF interpolator")
        
        # Test interpolation with some values
        test_points = np.linspace(0, 1, 5)
        test_values = self.inverse_cdf(test_points)
        logger.debug(f"Test interpolation: input={test_points}, output={test_values}")
    
    def update_entropy_distribution(self, time_steps: torch.Tensor, cdf: torch.Tensor):
        """Update the entropy distribution using the given data batch."""
        logger.debug(f"Updating entropy distribution with input: shape={time_steps.shape}, requires_grad={time_steps.requires_grad}")
        
        # Update attributes
        self.time_steps = time_steps
        self.cdf = cdf
        
        # Build inverse CDF
        self._build_inverse_cdf(time_steps, cdf)
        
        # Verify monotonicity
        self._verify_monotonicity()
        
    def _verify_monotonicity(self):
        """Verify that the CDF is monotonic."""
        if self.cdf is None:
            return
            
        # Check for any decreasing values
        is_monotonic = torch.all(torch.diff(self.cdf) >= 0)
        logger.debug(f"CDF monotonicity check: {is_monotonic}")
        if not is_monotonic:
            raise ValueError("CDF is not monotonic!")
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Convert timesteps to sigma values using base sampler."""
        return self.base_sampler.sigma(t)
    
    def signal(self, t: torch.Tensor) -> torch.Tensor:
        """Get signal level using base sampler."""
        return self.base_sampler.signal(t)
    
    def sample(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Sample σ values using inverse transform sampling."""
        if self.inverse_cdf is None:
            raise ValueError("Entropy distribution not initialized. Call update_entropy_distribution first.")
            
        device = device or self.device
        logger.debug(f"Sampling {batch_size} values on device: {device}")
            
        # Generate uniform samples
        u = torch.rand(batch_size, device=device)
        logger.debug(f"Generated uniform samples: min={u.min().item():.4f}, max={u.max().item():.4f}, mean={u.mean().item():.4f}")
        
        # Convert to numpy for interpolation
        u_np = u.cpu().numpy()
        
        # Ensure uniform samples are within valid range
        u_np = np.clip(u_np, 0.0, 1.0)
        
        # Sample using inverse CDF
        sigma_np = self.inverse_cdf(u_np)
        logger.debug(f"Sampled sigma values: min={sigma_np.min():.4f}, max={sigma_np.max():.4f}, mean={sigma_np.mean():.4f}")
        
        # Check for invalid values
        if np.isnan(sigma_np).any():
            logger.error(f"NaN values detected in sigma_np! CDF range: [{self.cdf.min().item():.4f}, {self.cdf.max().item():.4f}]")
            # Fallback to base sampler
            logger.warning("Falling back to base sampler due to NaN values")
            return self.base_sampler(batch_size, device)
        
        # Convert back to tensor and detach from computation graph
        sigma = torch.from_numpy(sigma_np).to(device).to(torch.float32).detach()
        logger.debug(f"Final sigma tensor: min={sigma.min().item():.4f}, max={sigma.max().item():.4f}, mean={sigma.mean().item():.4f}")
        
        return sigma
        
    def __call__(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Alias for sample method to match NoiseSampler interface."""
        return self.sample(batch_size, device)