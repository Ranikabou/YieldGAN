"""
GAN models for treasury curve data generation.
Implements Generator, Discriminator, and various GAN architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np

class Generator(nn.Module):
    """
    Generator network that transforms random noise into synthetic treasury data.
    """
    
    def __init__(self, latent_dim: int, hidden_dims: List[int], 
                 output_dim: int, sequence_length: int, dropout: float = 0.3):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = nn.Linear(latent_dim, hidden_dims[0] * sequence_length)
        
        # Main network layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1] range
        
        self.main = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator.
        
        Args:
            z: Random noise tensor of shape (batch_size, latent_dim)
            
        Returns:
            Generated treasury data of shape (batch_size, sequence_length, output_dim)
        """
        batch_size = z.size(0)
        
        # Project noise to initial dimensions
        x = self.input_projection(z)  # (batch_size, hidden_dims[0] * sequence_length)
        x = x.view(batch_size, self.sequence_length, -1)  # (batch_size, sequence_length, hidden_dims[0])
        
        # Process through main network
        x = x.view(-1, x.size(-1))  # Flatten for linear layers
        x = self.main(x)
        x = x.view(batch_size, self.sequence_length, self.output_dim)
        
        return x

class Discriminator(nn.Module):
    """
    Discriminator network that distinguishes real from synthetic treasury data.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 sequence_length: int, dropout: float = 0.3):
        super(Discriminator, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Main network layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
        
        # Output layer (real/fake classification)
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.main = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.
        
        Args:
            x: Input treasury data of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Probability of real data of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Process each timestep
        timestep_outputs = []
        for t in range(self.sequence_length):
            timestep_input = x[:, t, :]  # (batch_size, input_dim)
            timestep_output = self.input_projection(timestep_input)
            timestep_outputs.append(timestep_output)
        
        # Average across timesteps
        x = torch.stack(timestep_outputs, dim=1)  # (batch_size, sequence_length, hidden_dims[0])
        x = x.mean(dim=1)  # (batch_size, hidden_dims[0])
        
        # Process through main network
        x = self.main(x)
        
        return x

class ConditionalGenerator(nn.Module):
    """
    Conditional generator that takes additional context (e.g., economic indicators).
    """
    
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dims: List[int],
                 output_dim: int, sequence_length: int, dropout: float = 0.3):
        super(ConditionalGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        
        # Combined input projection
        combined_dim = latent_dim + condition_dim
        self.input_projection = nn.Linear(combined_dim, hidden_dims[0] * sequence_length)
        
        # Main network layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Tanh())
        
        self.main = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditional input.
        
        Args:
            z: Random noise tensor
            condition: Conditional input tensor
            
        Returns:
            Generated treasury data
        """
        batch_size = z.size(0)
        
        # Concatenate noise and condition
        combined_input = torch.cat([z, condition], dim=1)
        
        # Project combined input
        x = self.input_projection(combined_input)
        x = x.view(batch_size, self.sequence_length, -1)
        
        # Process through main network
        x = x.view(-1, x.size(-1))
        x = self.main(x)
        x = x.view(batch_size, self.sequence_length, self.output_dim)
        
        return x

class WassersteinGenerator(nn.Module):
    """
    Generator for Wasserstein GAN (WGAN).
    """
    
    def __init__(self, latent_dim: int, hidden_dims: List[int],
                 output_dim: int, sequence_length: int, dropout: float = 0.3):
        super(WassersteinGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = nn.Linear(latent_dim, hidden_dims[0] * sequence_length)
        
        # Main network layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1]),  # LayerNorm instead of BatchNorm for WGAN
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        # Output layer (no activation for WGAN)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.main = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for WGAN."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass for WGAN generator."""
        batch_size = z.size(0)
        
        x = self.input_projection(z)
        x = x.view(batch_size, self.sequence_length, -1)
        
        x = x.view(-1, x.size(-1))
        x = self.main(x)
        x = x.view(batch_size, self.sequence_length, self.output_dim)
        
        return x

class WassersteinCritic(nn.Module):
    """
    Critic network for Wasserstein GAN (WGAN).
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int],
                 sequence_length: int, dropout: float = 0.3):
        super(WassersteinCritic, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Main network layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        # Output layer (no activation for WGAN critic)
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.main = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for WGAN critic."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for WGAN critic."""
        batch_size = x.size(0)
        
        # Process each timestep
        timestep_outputs = []
        for t in range(self.sequence_length):
            timestep_input = x[:, t, :]
            timestep_output = self.input_projection(timestep_input)
            timestep_outputs.append(timestep_output)
        
        # Average across timesteps
        x = torch.stack(timestep_outputs, dim=1)
        x = x.mean(dim=1)
        
        # Process through main network
        x = self.main(x)
        
        return x

def create_gan_models(config: dict, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """
    Factory function to create GAN models based on configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to place models on
        
    Returns:
        Tuple of (generator, discriminator/critic)
    """
    model_config = config['model']
    data_config = config['data_processing']
    
    # Extract parameters
    latent_dim = model_config['generator']['latent_dim']
    hidden_dims = model_config['generator']['hidden_dims']
    output_dim = data_config['num_features']
    sequence_length = data_config['sequence_length']
    dropout = model_config['generator']['dropout']
    
    # Create models
    generator = Generator(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        sequence_length=sequence_length,
        dropout=dropout
    ).to(device)
    
    discriminator = Discriminator(
        input_dim=output_dim,
        hidden_dims=model_config['discriminator']['hidden_dims'],
        sequence_length=sequence_length,
        dropout=model_config['discriminator']['dropout']
    ).to(device)
    
    return generator, discriminator

def create_wgan_models(config: dict, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """
    Factory function to create WGAN models.
    
    Args:
        config: Configuration dictionary
        device: Device to place models on
        
    Returns:
        Tuple of (generator, critic)
    """
    model_config = config['model']
    data_config = config['data_processing']
    
    instrument_count = len(config.get('instruments', ['2Y', '5Y', '10Y', '30Y']))
    # Calculate total features: instruments + spreads + volatility features
    num_features = instrument_count + 3 + instrument_count  # instruments + spreads + volatility
    
    # Extract parameters
    latent_dim = model_config['generator']['latent_dim']
    hidden_dims = model_config['generator']['hidden_dims']
    output_dim = num_features
    sequence_length = data_config['sequence_length']
    dropout = model_config['generator']['dropout']
    
    # Create WGAN models
    generator = WassersteinGenerator(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        sequence_length=sequence_length,
        dropout=dropout
    ).to(device)
    
    critic = WassersteinCritic(
        input_dim=output_dim,
        hidden_dims=model_config['discriminator']['hidden_dims'],
        sequence_length=sequence_length,
        dropout=model_config['discriminator']['dropout']
    ).to(device)
    
    return generator, critic 