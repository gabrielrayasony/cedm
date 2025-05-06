"""Diffusion models architecture for 2D data"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Optional

#----------------------------------------------------------------------------#
# Positional embedding
#----------------------------------------------------------------------------#

class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps.
    
    This module encodes a scalar timestep t into a high-dimensional vector using
    sine and cosine functions at exponentially spaced frequencies (harmonic embedding).
    
    Example usage:
        time_embedding = PositionEmbedding(dim=256)
        timesteps = torch.tensor([0, 1, 2, 3])
        embeddings = time_embedding(timesteps)  # Shape: [4, 256]
    """
    def __init__(self, embedding_dim: int, max_frequency: float = 10000.0):
        """
        Args:
            embedding_dim (int): Total dimensionality of the embedding (must be even).
            max_frequency (float): The maximum frequency scale for the embedding.
        """
        super().__init__()
        assert embedding_dim % 2 == 0, "embedding_dim must be even"

        half_dim = embedding_dim // 2  # Number of frequency bands
        frequency_indices = torch.arange(half_dim, dtype=torch.float32)

        # Logarithmically spaced frequencies: omega_i = 1 / (max_frequency^{i / half_dim})
        frequency_exponents = -np.log(max_frequency) * frequency_indices / (half_dim - 1)
        frequencies = torch.exp(frequency_exponents)  # shape: [half_dim]

        self.register_buffer("frequencies", frequencies)  # Shape: [half_dim]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (Tensor): Input timesteps, shape [batch_size] or [batch_size, 1].

        Returns:
            Tensor of shape [batch_size, embedding_dim], the harmonic embedding of t.
        """
        t = t.float()#.unsqueeze(-1)  # Shape: [B, 1]
        freqs = self.frequencies  # Shape: [half_dim]
        args = t * freqs  # Shape: [B, half_dim]

        sin_embed = torch.sin(args)  # Shape: [B, half_dim]
        cos_embed = torch.cos(args)  # Shape: [B, half_dim]

        return torch.cat([sin_embed, cos_embed], dim=-1)  # Shape: [B, embedding_dim]

#----------------------------------------------------------------------------#
# MLP
#----------------------------------------------------------------------------#  

class MLP(nn.Module):
    """MLP for diffusion models with time embedding.
    
    A simple MLP that takes both the input data and time embedding as input.
    The network consists of:
    1. Time embedding layer (sinusoidal)
    2. Input layer (concatenates data and time embedding)
    3. Hidden layers with SiLU activation
    4. Output layer
    
    Example usage:
        model = MLP(input_dim=2, hidden_dim=128, num_hidden_layers=4, time_embedding_dim=4)
        x = torch.randn(32, 2)  # batch of 32, 2D data
        t = torch.randn(32)     # batch of 32 timesteps
        out = model(x, t)       # shape: [32, 2]
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_hidden_layers: int = 4,
        time_embedding_dim: int = 4
    ):
        """Initialize the MLP.
        
        Args:
            input_dim: Dimension of input data
            hidden_dim: Dimension of hidden layers
            num_hidden_layers: Number of hidden layers
            time_embedding_dim: Dimension of time embedding
        """
        super().__init__()
        
        # Time embedding
        self.time_embedding = PositionalEmbedding(time_embedding_dim)
        
        # Network layers
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(input_dim + time_embedding_dim, hidden_dim),
            nn.SiLU()
        ])
        
        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input data of shape [batch_size, input_dim]
            t: Timesteps of shape [batch_size]
            labels: Optional labels of shape [batch_size, num_labels]
        Returns:
            Output of shape [batch_size, input_dim]
        """
        # Get time embedding
        t_emb = self.time_embedding(t)  # [batch_size, time_embedding_dim]
        # Concatenate input and time embedding
        x = torch.cat([x, t_emb], dim=-1)  # [batch_size, input_dim + time_embedding_dim]
        
        # Pass through network
        return self.net(x)


# Create the model
# model = MLP(input_dim=2, hidden_dim=128, num_hidden_layers=4, time_embedding_dim=4)

# # # Dummy tensors for testing
# dummy_x = torch.randn(5, 2)  # Batch of 5, input_dim of 2
# dummy_t = torch.rand(5, 1)      # Batch of 5, single time dimension

# # # Test the network with dummy tensors
# test_output = model(dummy_x, dummy_t)  # Get the output from the model
     

#----------------------------------------------------------------------------#
# More Complex MLP inspired by DDPM network
#----------------------------------------------------------------------------#


class BaseBlockMLP(nn.Module):
    """A single block of the point denoising MLP model."""

    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        time_emb_dim: int,
        norm_cls: nn.Module = nn.LayerNorm,
        activation_fn: nn.Module = nn.LeakyReLU(negative_slope=0.02, inplace=True)
    ) -> None:
        """Initialize the MLP block.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            time_emb_dim: Time embedding dimension
            norm_cls: Normalization layer class
            activation_fn: Activation function
        """
        super().__init__()

        # Coordinate transform layers
        self.input_norm = norm_cls(input_dim)
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)
        self.hidden_norm = norm_cls(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim, bias=False)
        self.skip_fc = nn.Linear(input_dim, output_dim, bias=False)

        # Time transform layers
        self.time_fc = nn.Linear(time_emb_dim, output_dim, bias=False)
        
        # Store activation
        self.activation = activation_fn

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            t_emb: Time embedding tensor of shape (batch_size, time_emb_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        h = self.fc1(self.activation(self.input_norm(x)))
        h += self.time_fc(t_emb)
        h = self.fc2(self.activation(self.hidden_norm(h)))
        return h + self.skip_fc(x)

class AdvancedMLP(nn.Module):
    """Advanced MLP for diffusion models with time embedding.
    
    Architecture:
    - Input: (N, D) tensor of N points in D dimensions (D=2 or D=3)
    - Output: (N, D) tensor of N points in D dimensions (D=2 or D=3)
    - The model consists of multiple PointDenoisingMLP blocks
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 6,
        norm_cls: nn.Module = nn.LayerNorm,
        activation_fn: nn.Module = nn.LeakyReLU(negative_slope=0.02, inplace=True)
    ):
        """Initialize the model.
        
        Args:
            input_dim: Input dimension (2 for 2D points, 3 for 3D points)
            hidden_dim: Hidden dimension
            num_layers: Number of MLP blocks
            norm_cls: Normalization layer class
            activation_fn: Activation function
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

        # Input coordinates and time transforms
        self.coord_transform = nn.Linear(input_dim, hidden_dim, bias=False)
        self.time_transform = nn.Sequential(
            PositionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
        )

        # MLP blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                BaseBlockMLP(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    time_emb_dim=hidden_dim,
                    norm_cls=norm_cls,
                    activation_fn=activation_fn
                )
            )

        # Output transform layers
        self.output_norm = norm_cls(hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, input_dim)

        # Initialize output layer weights with zeros
        nn.init.zeros_(self.output_fc.weight)
        nn.init.zeros_(self.output_fc.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            t: Time steps tensor of shape (batch_size,)
            class_labels: Optional class labels tensor of shape (batch_size,)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Transform input coordinates
        x = self.coord_transform(x)
        
        # Transform time
        t_emb = self.time_transform(t)
        
        # Apply MLP blocks
        for block in self.blocks:
            x = block(x, t_emb)
            
        # Transform output
        output = self.output_fc(self.activation(self.output_norm(x)))
        
        return output 
        
    
    
    