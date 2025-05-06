"""Model utilities for creating and configuring models."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from networks.toy_network import MLP
from networks.preconditioning import (
    EDMPrecond,
    VPPrecond,
    VEPrecond,
    iDDPMPrecond
)
from models.edm import EDM

def get_model(config: Dict[str, Any]) -> nn.Module:
    """Create a model based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured model
    """
    # Get model type
    model_type = config.get('type', 'edm')
    
    # Create base network
    network_config = config.get('network', {})
    base_model = MLP(
        input_dim=network_config.get('input_dim', 2),
        hidden_dim=network_config.get('hidden_dim', 128),
        num_hidden_layers=network_config.get('num_hidden_layers', 4),
        time_embedding_dim=network_config.get('time_embedding_dim', 8)
    )
    
    # Create preconditioning based on model type
    if model_type == 'edm':
        precond = EDMPrecond(
            sigma_data=config['edm']['sigma_data'],
            sigma_min=config['edm']['sigma_min'],
            sigma_max=config['edm']['sigma_max']
        )
    elif model_type == 'vp':
        precond = VPPrecond(
            beta_d=config['vp']['beta_d'],
            beta_min=config['vp']['beta_min'],
            M=config['vp']['M'],
            epsilon_t=config['vp']['epsilon_t'],
            sigma_data=config['vp']['sigma_data']
        )
    elif model_type == 've':
        precond = VEPrecond(
            sigma_min=config['ve']['sigma_min'],
            sigma_max=config['ve']['sigma_max'],
            sigma_data=config['ve']['sigma_data']
        )
    elif model_type == 'iddpm':
        precond = iDDPMPrecond(
            C_1=config['iddpm']['C_1'],
            C_2=config['iddpm']['C_2'],
            M=config['iddpm']['M'],
            sigma_data=config['iddpm']['sigma_data']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create EDM model with base network and preconditioning
    model = EDM(
        base_model=base_model,
        precond=precond
    )
    
    return model 