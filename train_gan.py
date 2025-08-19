#!/usr/bin/env python3
"""
Main training script for Treasury Curve GAN.
Orchestrates the entire training pipeline from data collection to model evaluation.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from training.trainer import GANTrainer, load_config
from utils.data_utils import TreasuryDataProcessor, create_data_loaders
from evaluation.metrics import evaluate_treasury_gan
import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    return device

def collect_data(config: dict, start_date: str, end_date: str) -> tuple:
    """
    Collect and prepare data for training.
    
    Args:
        config: Configuration dictionary
        start_date: Start date for data collection
        end_date: End date for data collection
        
    Returns:
        Tuple of (sequences, targets, scaler)
    """
    logger.info("Collecting and preparing data...")
    
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
    
    # Collect new data
    logger.info("Collecting new data...")
    processor = TreasuryDataProcessor(
        instruments=config['instruments'],
        sequence_length=config['data']['sequence_length']
    )
    
    sequences, targets, scaler = processor.prepare_data(start_date, end_date)
    
    # Save processed data
    np.save(sequences_path, sequences)
    np.save(targets_path, targets)
    
    logger.info(f"Data collection completed: sequences {sequences.shape}, targets {targets.shape}")
    return sequences, targets, scaler

def train_model(config: dict, device: torch.device, sequences: np.ndarray, targets: np.ndarray):
    """
    Train the GAN model.
    
    Args:
        config: Configuration dictionary
        device: Device to train on
        sequences: Input sequences
        targets: Target values
    """
    logger.info("Starting model training...")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        sequences, targets,
        batch_size=config['data']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['validation_split']
    )
    
    # Create trainer
    trainer = GANTrainer(config, device)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    return trainer

def evaluate_model(trainer: GANTrainer, test_loader, scaler, config: dict):
    """
    Evaluate the trained model.
    
    Args:
        trainer: Trained GAN trainer
        test_loader: Test data loader
        scaler: Data scaler
        config: Configuration dictionary
    """
    logger.info("Evaluating model...")
    
    # Generate synthetic data
    num_samples = min(100, len(test_loader.dataset))
    synthetic_data = trainer.generate_sample(num_samples=num_samples)
    
    # Get real test data
    real_data = []
    for batch_idx, (data, _) in enumerate(test_loader):
        if batch_idx * test_loader.batch_size >= num_samples:
            break
        real_data.append(data)
    
    real_data = torch.cat(real_data, dim=0)[:num_samples]
    
    # Convert to numpy for evaluation
    real_data_np = real_data.cpu().numpy()
    synthetic_data_np = synthetic_data.cpu().numpy()
    
    # Denormalize data if scaler is available and fitted
    if hasattr(scaler, 'inverse_transform') and hasattr(scaler, 'mean_'):
        try:
            real_shape = real_data_np.shape
            real_denorm = scaler.inverse_transform(real_data_np.reshape(-1, real_data_np.shape[-1]))
            real_data_np = real_denorm.reshape(real_shape)
            
            syn_shape = synthetic_data_np.shape
            syn_denorm = scaler.inverse_transform(synthetic_data_np.reshape(-1, synthetic_data_np.shape[-1]))
            synthetic_data_np = syn_denorm.reshape(syn_shape)
            logger.info("Data denormalized successfully")
        except Exception as e:
            logger.warning(f"Could not denormalize data: {e}. Using normalized data for evaluation.")
    else:
        logger.info("Scaler not fitted, using normalized data for evaluation")
    
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
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Treasury Curve GAN')
    parser.add_argument('--config', type=str, default='config/gan_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--start-date', type=str, default='2022-01-01',
                       help='Start date for data collection (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-01-01',
                       help='End date for data collection (YYYY-MM-DD)')
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
    
    # Setup environment
    device = setup_environment()
    
    # Collect data
    sequences, targets, scaler = collect_data(config, args.start_date, args.end_date)
    
    if args.skip_training and args.checkpoint:
        # Load existing model for evaluation
        logger.info("Loading existing model for evaluation...")
        trainer = GANTrainer(config, device)
        trainer.load_checkpoint(args.checkpoint)
        
        # Create test data loader
        _, _, test_loader = create_data_loaders(
            sequences, targets,
            batch_size=config['data']['batch_size'],
            train_split=config['data']['train_split'],
            val_split=config['data']['validation_split']
        )
        
        evaluate_model(trainer, test_loader, scaler, config)
    else:
        # Train model
        trainer = train_model(config, device, sequences, targets)
        
        # Evaluate model
        _, _, test_loader = create_data_loaders(
            sequences, targets,
            batch_size=config['data']['batch_size'],
            train_split=config['data']['train_split'],
            val_split=config['data']['validation_split']
        )
        
        evaluate_model(trainer, test_loader, scaler, config)
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main() 