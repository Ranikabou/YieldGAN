#!/usr/bin/env python3
"""
Main training script for Treasury Curve GAN using CSV data sources.
Modified version for loading data from CSV files instead of APIs.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

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
    
    # Get CSV directory from config
    csv_directory = config.get('data_source', {}).get('csv_directory', 'data/csv')
    
    # Check if CSV directory exists and contains files
    if not os.path.exists(csv_directory):
        logger.error(f"CSV directory {csv_directory} does not exist!")
        logger.info("Please create the directory and add your CSV files:")
        logger.info(f"  mkdir -p {csv_directory}")
        logger.info("  # Then add your CSV files:")
        logger.info("  # - treasury_yields.csv")
        logger.info("  # - order_book_data.csv")
        logger.info("  # - features.csv")
        raise FileNotFoundError(f"CSV directory {csv_directory} not found")
    
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    if not csv_files:
        logger.error(f"No CSV files found in {csv_directory}")
        logger.info("Please add CSV files to the directory:")
        logger.info(f"  {csv_directory}/")
        raise FileNotFoundError(f"No CSV files found in {csv_directory}")
    
    logger.info(f"Found CSV files: {csv_files}")
    
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

def train_model(config: dict, device: torch.device, sequences: np.ndarray, targets: np.ndarray):
    """
    Train the GAN model.
    
    Args:
        config: Configuration dictionary
        device: Device to train on
        sequences: Training sequences
        targets: Training targets
        
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
    
    # Train the model
    trainer.train(train_loader, val_loader)
    
    return trainer

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
    parser.add_argument('--config', type=str, default='config/csv_config.yaml',
                       help='Path to CSV configuration file')
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
    
    # Setup environment
    device = setup_environment()
    
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
        trainer = train_model(config, device, sequences, targets)
        
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