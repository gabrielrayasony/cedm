"""Model utilities for creating and configuring models."""

import torch.nn as nn
import dnnlib
import pickle
import logging
from typing import Dict, Any
from networks.toy_network import MLP, AdvancedMLP
from networks.preconditioning import (
    EDMDenoiser,
    VPPrecond,
    VEPrecond,
    iDDPMPrecond
)

from networks.edm_networks import SongUNet, DhariwalUNet
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------
# Download a pre-trained network from URL
#-------------------------------------------------------------------------
def download_network(network_url: str) -> nn.Module:
    """Download a pre-trained network from URL."""
    logger.info(f"Downloading network from {network_url}...")
    with dnnlib.util.open_url(network_url) as f:
        logger.info("Downloaded model")
        net = pickle.load(f)['ema']
    return net

#-------------------------------------------------------------------------
# Get a neural network based on configuration
#-------------------------------------------------------------------------
def get_neural_net(config: Dict[str, Any]) -> nn.Module:
    if config.model.class_conditional:
        label_dim = config.model.label_dim
    else:
        label_dim = 0

    if config.network.name == 'mlp':
        net = MLP(
            input_dim=config.network.input_dim,
            hidden_dim=config.network.hidden_dim,
            num_hidden_layers=config.network.num_hidden_layers
        )
    elif config.network.name == 'advanced_mlp':
        net = AdvancedMLP(
            input_dim=config.network.input_dim,
            hidden_dim=config.network.hidden_dim,
            num_hidden_layers=config.network.num_hidden_layers, 
        )

    elif config.network.name in ['ddpmpp', 'ncsnpp']:
        net = SongUNet(
            img_resolution=config.dataset.img_resolution,
            in_channels=config.dataset.in_channels,
            out_channels=config.dataset.out_channels,
            label_dim=label_dim,
            embedding_type=config.network.embedding_type,
            encoder_type=config.network.encoder_type,
            decoder_type=config.network.decoder_type,
            channel_mult_noise=config.network.channel_mult_noise,
            resample_filter=list(config.network.resample_filter),
            model_channels=config.network.model_channels,
            channel_mult=list(config.network.channel_mult),
            dropout=config.network.dropout,
            num_blocks=config.network.num_blocks,
        )
    elif config.network.name == 'adm':
        net = DhariwalUNet(
            img_resolution=config.dataset.img_resolution,
            in_channels=config.dataset.in_channels,
            out_channels=config.dataset.out_channels,
            label_dim=label_dim,
            model_channels=config.network.model_channels,
            channel_mult=config.network.channel_mult,
            dropout=config.network.dropout,
            num_blocks=config.network.num_blocks,
        )
    else:
        raise ValueError(f"Unknown network type: {config.network.name}")
    
    return net

#-------------------------------------------------------------------------
# Get a Denoiser model based on configuration as described in the EDM paper
#-------------------------------------------------------------------------
def get_model(config: Dict[str, Any]) -> nn.Module:
    """Create a model based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured model
    """
    # Get model type
    precond_type = config.precond.name
    
    # Get base model
    net = get_neural_net(config)

    # Create preconditioning based on model type
    if precond_type == 'edm':
        denoiser = EDMDenoiser(
            model=net,
            sigma_data=config.precond.sigma_data,
        )
    elif precond_type == 'vp':
        denoiser = VPPrecond(
            beta_d=config.precond.beta_d,
            beta_min=config.precond.beta_min,
            M=config.precond.M,
            epsilon_t=config.precond.epsilon_t,
            sigma_data=config.precond.sigma_data
        )
    elif precond_type == 've':
        denoiser = VEPrecond(
            sigma_min=config.precond.sigma_min,
            sigma_max=config.precond.sigma_max,
            sigma_data=config.precond.sigma_data
        )
    elif precond_type == 'iddpm':
        denoiser = iDDPMPrecond(
            C_1=config.precond.C_1,
            C_2=config.precond.C_2,
            M=config.precond.M,
            sigma_data=config.precond.sigma_data
        )
    else:
        raise ValueError(f"Unknown preconditioning type: {precond_type}")

    return denoiser, net



