#!/usr/bin/env python3
"""
Main training script for Treasury Curve GAN using CSV data sources.
Modified version for loading data from CSV files instead of APIs.
Now includes dashboard channel integration for real-time monitoring.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
import requests
import time
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.csv_collector import CSVDataCollector
from training.trainer import GANTrainer, load_config
from utils.data_utils import create_data_loaders
from evaluation.metrics import evaluate_treasury_gan
import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardChannelSender:
    """Class to send training data and progress to dashboard channels."""
    
    def __init__(self, dashboard_url="http://localhost:8081"):
        self.dashboard_url = dashboard_url
        self.total_epochs = None
    
    def set_total_epochs(self, total_epochs):
        """Set the total number of epochs for progress calculation."""
        self.total_epochs = total_epochs
    
    def send_training_data(self, epoch, generator_loss, discriminator_loss, real_scores, fake_scores):
        """Send training metrics to training channel."""
        training_data = {
            "type": "training_update",
            "data": {
                "epoch": epoch,
                "total_epochs": self.total_epochs,
                "generator_loss": round(float(generator_loss), 4),
                "discriminator_loss": round(float(discriminator_loss), 4),
                "real_scores": round(float(real_scores), 4),
                "fake_scores": round(float(fake_scores), 4)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(
                f"{self.dashboard_url}/training_data",
                json=training_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"🎯 Training data sent to dashboard: Epoch {epoch}, Gen Loss: {generator_loss:.4f}, Disc Loss: {discriminator_loss:.4f}")
                return True
            else:
                logger.warning(f"Training data failed to send: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Training data error: {e}")
            return False
    
    def send_progress_data(self, epoch, progress_percent):
        """Send progress update to progress channel."""
        progress_data = {
            "type": "progress",
            "epoch": epoch,
            "progress_percent": progress_percent,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(
                f"{self.dashboard_url}/progress_data",
                json=progress_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"📊 Progress {progress_percent}% sent to dashboard for epoch {epoch}")
                return True
            else:
                logger.warning(f"Progress {progress_percent}% failed to send: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Progress {progress_percent}% error: {e}")
            return False

def setup_environment():
    """Setup training environment and check dependencies."""
    logger.info("Setting up training environment...")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        logger.info("CUDA not available, using CPU")
        device = torch.device('cpu')
    
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/csv', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    return device

def collect_csv_data(config: dict, sequence_length: int = 100) -> tuple:
    """
    Collect and prepare data from CSV files for training.
    
    Args:
        config: Configuration dictionary
        sequence_length: Length of sequences for training
        
    Returns:
        Tuple of (sequences, targets, scaler)
    """
    logger.info("Collecting and preparing CSV data...")
    
    # Check if processed data already exists
    sequences_path = 'data/sequences.npy'
    targets_path = 'data/targets.npy'
    
    if os.path.exists(sequences_path) and os.path.exists(targets_path):
        logger.info("Loading existing processed data...")
        sequences = np.load(sequences_path)
        targets = np.load(targets_path)
        
        # Create a dummy scaler (in production, you'd save/load the actual scaler)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        logger.info(f"Loaded data: sequences {sequences.shape}, targets {targets.shape}")
        return sequences, targets, scaler
    
    # Get CSV file from config
    csv_file = config.get('data_source', {}).get('csv_file', 'treasury_orderbook_sample.csv')
    csv_directory = 'data/csv'
    csv_path = os.path.join(csv_directory, csv_file)
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        logger.error(f"CSV file {csv_path} does not exist!")
        logger.info("Please ensure the CSV file is in the data/csv directory:")
        logger.info(f"  {csv_path}")
        raise FileNotFoundError(f"CSV file {csv_path} not found")
    
    logger.info(f"Using CSV file: {csv_file}")
    
    # Collect new data from CSV
    logger.info("Processing CSV data...")
    collector = CSVDataCollector(csv_directory)
    
    try:
        sequences, targets = collector.collect_and_process(sequence_length)
        
        # Save processed data
        np.save(sequences_path, sequences)
        np.save(targets_path, targets)
        
        logger.info(f"Data collection completed: sequences {sequences.shape}, targets {targets.shape}")
        
        # Create a dummy scaler (in production, you'd save/load the actual scaler)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        return sequences, targets, scaler
        
    except Exception as e:
        logger.error(f"Error processing CSV data: {e}")
        raise

def train_model(config: dict, device: torch.device, sequences: np.ndarray, targets: np.ndarray, dashboard_sender=None):
    """
    Train the GAN model with dashboard channel integration.
    
    Args:
        config: Configuration dictionary
        device: Device to train on
        sequences: Training sequences
        targets: Training targets
        dashboard_sender: Optional DashboardChannelSender instance
        
    Returns:
        Trained GANTrainer instance
    """
    logger.info("Starting GAN training...")
    
    # Create trainer
    trainer = GANTrainer(config, device)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        sequences, targets,
        batch_size=config['training']['batch_size'],
        train_split=config['data_processing']['train_split'],
        val_split=config['data_processing']['validation_split']
    )
    
    # If dashboard sender is provided, use custom training loop
    if dashboard_sender:
        logger.info("Using dashboard-integrated training loop...")
        custom_train_with_dashboard(trainer, train_loader, val_loader, dashboard_sender)
    else:
        # Use default training
        logger.info("Using default training loop...")
        trainer.train(train_loader, val_loader)
    
    return trainer

def custom_train_with_dashboard(trainer, train_loader, val_loader, dashboard_sender):
    """
    Custom training loop that sends data to dashboard channels.
    
    Args:
        trainer: GANTrainer instance
        train_loader: Training data loader
        val_loader: Validation data loader
        dashboard_sender: DashboardChannelSender instance
    """
    logger.info("Starting custom training loop with dashboard integration...")
    
    total_epochs = trainer.config['training']['epochs']
    dashboard_sender.set_total_epochs(total_epochs)
    
    for epoch in range(total_epochs):
        trainer.current_epoch = epoch
        
        # Send progress update for epoch start
        dashboard_sender.send_progress_data(epoch, 0)
        
        # Train epoch
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Update metrics
        trainer.generator_losses.append(train_metrics['generator_loss'])
        trainer.discriminator_losses.append(train_metrics['discriminator_loss'])
        trainer.discriminator_real_scores.append(train_metrics['real_scores'])
        trainer.discriminator_fake_scores.append(train_metrics['fake_scores'])
        
        # Send training data to dashboard
        dashboard_sender.send_training_data(
            epoch + 1,  # epoch is 0-indexed, but dashboard expects 1-indexed
            train_metrics['generator_loss'],
            train_metrics['discriminator_loss'],
            train_metrics['real_scores'],
            train_metrics['fake_scores']
        )
        
        # Send progress update for epoch completion
        dashboard_sender.send_progress_data(epoch, 100)
        
        # Log progress
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{total_epochs}")
            logger.info(f"Generator Loss: {train_metrics['generator_loss']:.4f}")
            logger.info(f"Discriminator Loss: {train_metrics['discriminator_loss']:.4f}")
            logger.info(f"Real Scores: {train_metrics['real_scores']:.4f}")
            logger.info(f"Fake Scores: {train_metrics['fake_scores']:.4f}")
            logger.info(f"Val Generator Loss: {val_metrics['val_generator_loss']:.4f}")
            logger.info(f"Val Discriminator Loss: {val_metrics['val_discriminator_loss']:.4f}")
            logger.info("-" * 50)
        
        # Save checkpoint
        if epoch % 50 == 0:
            trainer.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
        
        # Early stopping
        current_loss = val_metrics['val_generator_loss']
        if current_loss < trainer.best_loss:
            trainer.best_loss = current_loss
            trainer.patience_counter = 0
            trainer.save_checkpoint("best_model.pth")
        else:
            trainer.patience_counter += 1
            
        if trainer.patience_counter >= trainer.config['training']['patience']:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        # Small delay to allow dashboard to process updates
        time.sleep(0.1)
    
    logger.info("Training completed!")
    trainer.plot_training_curves()

def evaluate_model(trainer: GANTrainer, test_loader, scaler, config: dict):
    """
    Evaluate the trained model.
    
    Args:
        trainer: Trained GANTrainer instance
        test_loader: Test data loader
        scaler: Data scaler for denormalization
        config: Configuration dictionary
    """
    logger.info("Evaluating model...")
    
    # Generate synthetic data
    synthetic_data = trainer.generate_synthetic_data(len(test_loader.dataset))
    
    # Get real data
    real_data = []
    for batch in test_loader:
        real_data.append(batch[0].cpu().numpy())
    real_data = np.concatenate(real_data, axis=0)
    
    # Denormalize data if scaler is available
    if scaler and hasattr(scaler, 'inverse_transform'):
        try:
            real_shape = real_data.shape
            real_denorm = scaler.inverse_transform(real_data.reshape(-1, real_data.shape[-1]))
            real_data_np = real_denorm.reshape(real_shape)
            
            syn_shape = synthetic_data.shape
            syn_denorm = scaler.inverse_transform(synthetic_data.reshape(-1, synthetic_data.shape[-1]))
            synthetic_data_np = syn_denorm.reshape(syn_shape)
            logger.info("Data denormalized successfully")
        except Exception as e:
            logger.warning(f"Could not denormalize data: {e}. Using normalized data for evaluation.")
            real_data_np = real_data
            synthetic_data_np = synthetic_data
    else:
        logger.info("Scaler not fitted, using normalized data for evaluation")
        real_data_np = real_data
        synthetic_data_np = synthetic_data
    
    # Run evaluation
    evaluation_results = evaluate_treasury_gan(
        real_data_np, 
        synthetic_data_np, 
        save_plots=True
    )
    
    # Save evaluation results
    import json
    with open('results/evaluation_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in evaluation_results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        json_results[key][sub_key] = sub_value.tolist()
                    else:
                        json_results[key][sub_key] = sub_value
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)
    
    logger.info("Evaluation completed and results saved to results/evaluation_results.json")

def main():
    """Main training function for CSV data."""
    parser = argparse.ArgumentParser(description='Train Treasury Curve GAN with CSV data')
    parser.add_argument('--config', type=str, default='config/gan_config.yaml',
                       help='Path to GAN configuration file')
    parser.add_argument('--data', type=str, default='treasury_orderbook_sample.csv',
                       help='Data source CSV file to use for training')
    parser.add_argument('--sequence-length', type=int, default=100,
                       help='Length of sequences for training')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only evaluate existing model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file to load')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Update config with data source if provided
    if args.data:
        if 'data_source' not in config:
            config['data_source'] = {}
        config['data_source']['csv_file'] = args.data
        logger.info(f"Using data source: {args.data}")
    
    # Setup environment
    device = setup_environment()
    
    # Initialize dashboard sender
    dashboard_url = config.get('dashboard', {}).get('url', "http://localhost:8081")
    dashboard_sender = DashboardChannelSender(dashboard_url)
    
    # Collect CSV data
    sequence_length = args.sequence_length or config.get('data_processing', {}).get('sequence_length', 100)
    sequences, targets, scaler = collect_csv_data(config, sequence_length)
    
    if args.skip_training and args.checkpoint:
        # Load existing model for evaluation
        logger.info("Loading existing model for evaluation...")
        trainer = GANTrainer(config, device)
        trainer.load_checkpoint(args.checkpoint)
        
        # Create test data loader
        _, _, test_loader = create_data_loaders(
            sequences, targets,
            batch_size=config['training']['batch_size'],
            train_split=config['data_processing']['train_split'],
            val_split=config['data_processing']['validation_split']
        )
        
        evaluate_model(trainer, test_loader, scaler, config)
    else:
        # Train model
        trainer = train_model(config, device, sequences, targets, dashboard_sender)
        
        # Evaluate model
        _, _, test_loader = create_data_loaders(
            sequences, targets,
            batch_size=config['training']['batch_size'],
            train_split=config['data_processing']['train_split'],
            val_split=config['data_processing']['validation_split']
        )
        
        evaluate_model(trainer, test_loader, scaler, config)
    
    logger.info("CSV-based training pipeline completed successfully!")

if __name__ == "__main__":
    main() 