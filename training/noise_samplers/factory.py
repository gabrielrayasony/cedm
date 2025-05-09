"""Factory functions for creating matching noise samplers and losses."""

from typing import Tuple, Dict, Any
from .schedules import NoiseSampler, LogNormalNoiseSampler, LinearNoiseSampler, LogUniformNoiseSampler
from ..losses import DiffusionLoss, EDMLoss, VPLoss, VELoss


def get_matching_sampler_and_loss(config: Dict[str, Any]) -> Tuple[NoiseSampler, DiffusionLoss]:
    """Get a matching pair of noise sampler and loss function.
    
    Args:
        config: Configuration dictionary containing loss and preconditioning settings
        
    Returns:
        Tuple of (noise_sampler, loss_fn)
        
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
    
    # Create matching pair based on type
    if precond_type == 'edm':
        sampler = LogNormalNoiseSampler(
            P_mean=config['precond'].get('P_mean', -1.2),
            P_std=config['precond'].get('P_std', 1.2)
        )
        loss = EDMLoss(
            sigma_data=config['precond'].get('sigma_data', 0.5)
        )
    elif precond_type == 'vp':
        sampler = LinearNoiseSampler(
            beta_min=config['precond'].get('beta_min', 0.1),
            beta_max=config['precond'].get('beta_d', 19.9),
            epsilon_t=config['precond'].get('epsilon_t', 1e-5)
        )
        loss = VPLoss()
    elif precond_type == 've':
        sampler = LogUniformNoiseSampler(
            sigma_min=config['precond'].get('sigma_min', 0.02),
            sigma_max=config['precond'].get('sigma_max', 100)
        )
        loss = VELoss()
    else:
        raise ValueError(f"Invalid preconditioning type: {precond_type}. Must be one of ['edm', 'vp', 've']")
        
    return sampler, loss 