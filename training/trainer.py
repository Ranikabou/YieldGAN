"""
GAN training module for treasury curve data.
Implements training loops, loss functions, and monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import os
from typing import Dict, List, Tuple, Optional
import yaml
from datetime import datetime

from models.gan_models import create_gan_models, create_wgan_models
from utils.data_utils import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GANTrainer:
    """
    Main training class for GAN models on treasury data.
    """
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.gan_type = config.get('gan_type', 'standard')  # 'standard' or 'wgan'
        
        # Initialize models
        if self.gan_type == 'wgan':
            self.generator, self.critic = create_wgan_models(config, device)
            self.discriminator = self.critic  # For compatibility
        else:
            self.generator, self.discriminator = create_gan_models(config, device)
            self.critic = self.discriminator  # For compatibility
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config['training']['learning_rate_generator'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config['training']['learning_rate_discriminator'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        
        # Loss functions
        self.criterion = nn.BCELoss()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.generator_losses = []
        self.discriminator_losses = []
        self.discriminator_real_scores = []
        self.discriminator_fake_scores = []
        
        # Create checkpoint directory
        os.makedirs('checkpoints', exist_ok=True)
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_real_scores = 0.0
        epoch_fake_scores = 0.0
        
        batch_count = 0
        
        for batch_idx, (real_data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {self.current_epoch}")):
            batch_size = real_data.size(0)
            real_data = real_data.to(self.device)
            
            # Create labels
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
            # Train Discriminator
            if self.gan_type == 'wgan':
                # WGAN training
                for _ in range(self.config['training']['critic_iterations']):
                    self.optimizer_d.zero_grad()
                    
                    # Real data
                    real_output = self.discriminator(real_data)
                    
                    # Fake data
                    noise = torch.randn(batch_size, self.config['model']['generator']['latent_dim']).to(self.device)
                    fake_data = self.generator(noise)
                    fake_output = self.discriminator(fake_data.detach())
                    
                    # WGAN loss
                    d_loss = fake_output.mean() - real_output.mean()
                    
                    # Gradient penalty
                    alpha = torch.rand(batch_size, 1, 1).to(self.device)
                    interpolated = alpha * real_data + (1 - alpha) * fake_data.detach()
                    interpolated.requires_grad_(True)
                    interpolated_output = self.discriminator(interpolated)
                    
                    gradients = torch.autograd.grad(
                        outputs=interpolated_output,
                        inputs=interpolated,
                        grad_outputs=torch.ones_like(interpolated_output),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                    d_loss += self.config['training']['lambda_gp'] * gradient_penalty
                    
                    d_loss.backward()
                    self.optimizer_d.step()
                    
            else:
                # Standard GAN training
                self.optimizer_d.zero_grad()
                
                # Real data
                real_output = self.discriminator(real_data)
                d_real_loss = self.criterion(real_output, real_labels)
                
                # Fake data
                noise = torch.randn(batch_size, self.config['model']['generator']['latent_dim']).to(self.device)
                fake_data = self.generator(noise)
                fake_output = self.discriminator(fake_data.detach())
                d_fake_loss = self.criterion(fake_output, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.optimizer_d.step()
            
            # Train Generator
            self.optimizer_g.zero_grad()
            
            if self.gan_type == 'wgan':
                # WGAN generator loss
                fake_output = self.discriminator(fake_data)
                g_loss = -fake_output.mean()
            else:
                # Standard GAN generator loss
                fake_output = self.discriminator(fake_data)
                g_loss = self.criterion(fake_output, real_labels)
            
            g_loss.backward()
            self.optimizer_g.step()
            
            # Update metrics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_real_scores += real_output.mean().item()
            epoch_fake_scores += fake_output.mean().item()
            batch_count += 1
        
        # Calculate average metrics
        avg_g_loss = epoch_g_loss / batch_count
        avg_d_loss = epoch_d_loss / batch_count
        avg_real_scores = epoch_real_scores / batch_count
        avg_fake_scores = epoch_fake_scores / batch_count
        
        return {
            'generator_loss': avg_g_loss,
            'discriminator_loss': avg_d_loss,
            'real_scores': avg_real_scores,
            'fake_scores': avg_fake_scores
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.generator.eval()
        self.discriminator.eval()
        
        val_g_loss = 0.0
        val_d_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for real_data, _ in val_loader:
                batch_size = real_data.size(0)
                real_data = real_data.to(self.device)
                
                # Create labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Real data
                real_output = self.discriminator(real_data)
                
                # Fake data
                noise = torch.randn(batch_size, self.config['model']['generator']['latent_dim']).to(self.device)
                fake_data = self.generator(noise)
                fake_output = self.discriminator(fake_data)
                
                # Calculate losses
                d_real_loss = self.criterion(real_output, real_labels)
                d_fake_loss = self.criterion(fake_output, fake_labels)
                d_loss = d_real_loss + d_fake_loss
                
                g_loss = self.criterion(fake_output, real_labels)
                
                val_g_loss += g_loss.item()
                val_d_loss += d_loss.item()
                batch_count += 1
        
        return {
            'val_generator_loss': val_g_loss / batch_count,
            'val_discriminator_loss': val_d_loss / batch_count
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        logger.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update metrics
            self.generator_losses.append(train_metrics['generator_loss'])
            self.discriminator_losses.append(train_metrics['discriminator_loss'])
            self.discriminator_real_scores.append(train_metrics['real_scores'])
            self.discriminator_fake_scores.append(train_metrics['fake_scores'])
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.config['training']['epochs']}")
                logger.info(f"Generator Loss: {train_metrics['generator_loss']:.4f}")
                logger.info(f"Discriminator Loss: {train_metrics['discriminator_loss']:.4f}")
                logger.info(f"Real Scores: {train_metrics['real_scores']:.4f}")
                logger.info(f"Fake Scores: {train_metrics['fake_scores']:.4f}")
                logger.info(f"Val Generator Loss: {val_metrics['val_generator_loss']:.4f}")
                logger.info(f"Val Discriminator Loss: {val_metrics['val_discriminator_loss']:.4f}")
                logger.info("-" * 50)
            
            # Save checkpoint
            if epoch % 50 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
            
            # Early stopping
            current_loss = val_metrics['val_generator_loss']
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience_counter = 0
                self.save_checkpoint("best_model.pth")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config['training']['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info("Training completed!")
        self.plot_training_curves()
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'metrics': {
                'generator_losses': self.generator_losses,
                'discriminator_losses': self.discriminator_losses,
                'real_scores': self.discriminator_real_scores,
                'fake_scores': self.discriminator_fake_scores
            }
        }
        
        torch.save(checkpoint, os.path.join('checkpoints', filename))
        logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        if 'metrics' in checkpoint:
            self.generator_losses = checkpoint['metrics']['generator_losses']
            self.discriminator_losses = checkpoint['metrics']['discriminator_losses']
            self.discriminator_real_scores = checkpoint['metrics']['real_scores']
            self.discriminator_fake_scores = checkpoint['metrics']['fake_scores']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def plot_training_curves(self) -> None:
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Generator loss
        axes[0, 0].plot(self.generator_losses)
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # Discriminator loss
        axes[0, 1].plot(self.discriminator_losses)
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        
        # Real scores
        axes[1, 0].plot(self.discriminator_real_scores)
        axes[1, 0].set_title('Real Data Scores')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        
        # Fake scores
        axes[1, 1].plot(self.discriminator_fake_scores)
        axes[1, 1].set_title('Fake Data Scores')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Generate synthetic samples.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated synthetic data
        """
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.config['model']['generator']['latent_dim']).to(self.device)
            synthetic_data = self.generator(noise)
        
        return synthetic_data

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main training function."""
    # Load configuration
    config = load_config('config/gan_config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create trainer
    trainer = GANTrainer(config, device)
    
    # Prepare data
    from utils.data_utils import TreasuryDataProcessor, create_data_loaders
    
    processor = TreasuryDataProcessor(
        instruments=config['instruments'],
        sequence_length=config['data']['sequence_length']
    )
    
    # Get data (using last 2 years for demonstration)
    sequences, targets, scaler = processor.prepare_data(
        start_date='2022-01-01',
        end_date='2024-01-01'
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        sequences, targets,
        batch_size=config['data']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['validation_split']
    )
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Generate sample
    synthetic_data = trainer.generate_sample(num_samples=5)
    logger.info(f"Generated synthetic data shape: {synthetic_data.shape}")

if __name__ == "__main__":
    main() 