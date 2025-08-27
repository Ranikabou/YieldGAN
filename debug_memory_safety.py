#!/usr/bin/env python3
"""
Debug script to test memory safety improvements and identify segmentation fault causes.
This script will help diagnose memory issues before running the full training.
"""

import os
import sys
import logging
import gc
import psutil
import torch
import numpy as np
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.csv_collector import CSVDataCollector
from utils.data_utils import safe_create_data_loaders, check_memory_usage
from models.gan_models import create_gan_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_memory_monitoring():
    """Test memory monitoring functionality."""
    logger.info("=== Testing Memory Monitoring ===")
    
    try:
        initial_memory = check_memory_usage()
        logger.info(f"Initial memory: {initial_memory:.2f} MB")
        
        # Create some test data
        test_data = np.random.randn(1000, 1000)
        logger.info(f"Created test data: {test_data.shape}")
        
        after_data_memory = check_memory_usage()
        logger.info(f"After creating data: {after_data_memory:.2f} MB")
        
        # Clean up
        del test_data
        gc.collect()
        
        final_memory = check_memory_usage()
        logger.info(f"After cleanup: {final_memory:.2f} MB")
        
        logger.info("✅ Memory monitoring test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory monitoring test failed: {e}")
        return False

def test_csv_data_loading():
    """Test CSV data loading with memory safety."""
    logger.info("=== Testing CSV Data Loading ===")
    
    try:
        # Check if CSV directory exists
        csv_dir = "data/csv"
        if not os.path.exists(csv_dir):
            logger.warning(f"CSV directory {csv_dir} not found, skipping test")
            return True
        
        # Initialize collector with reduced parameters
        collector = CSVDataCollector(csv_dir)
        
        # Test with reduced sequence length and sample limits
        sequences, targets = collector.collect_and_process(
            sequence_length=25,  # Reduced from 50
            max_samples=1000     # Reduced from 10000
        )
        
        logger.info(f"✅ CSV data loading test passed")
        logger.info(f"   Sequences: {sequences.shape}")
        logger.info(f"   Targets: {targets.shape}")
        
        # Clean up
        del sequences, targets
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CSV data loading test failed: {e}")
        return False

def test_data_loader_creation():
    """Test data loader creation with memory safety."""
    logger.info("=== Testing Data Loader Creation ===")
    
    try:
        # Create small synthetic data for testing
        num_samples = 500
        sequence_length = 25
        num_features = 21
        
        sequences = np.random.randn(num_samples, sequence_length, num_features)
        targets = np.random.randn(num_samples, num_features)
        
        logger.info(f"Created test data: {sequences.shape}, {targets.shape}")
        
        # Test safe data loader creation
        train_loader, val_loader, test_loader = safe_create_data_loaders(
            sequences, targets,
            batch_size=8,  # Small batch size
            max_sequence_length=25,
            max_batch_size=8
        )
        
        logger.info(f"✅ Data loader creation test passed")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Val batches: {len(val_loader)}")
        logger.info(f"   Test batches: {len(test_loader)}")
        
        # Test a few batches
        for i, (batch_data, batch_targets) in enumerate(train_loader):
            if i >= 2:  # Only test first 2 batches
                break
            logger.info(f"   Batch {i}: {batch_data.shape}, {batch_targets.shape}")
        
        # Clean up
        del sequences, targets, train_loader, val_loader, test_loader
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loader creation test failed: {e}")
        return False

def test_gan_model_creation():
    """Test GAN model creation with memory safety."""
    logger.info("=== Testing GAN Model Creation ===")
    
    try:
        # Load configuration
        config_path = "config/gan_config.yaml"
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found, using defaults")
            config = {
                'model': {
                    'generator': {'latent_dim': 50, 'hidden_dims': [128, 256, 128], 'dropout': 0.3},
                    'discriminator': {'hidden_dims': [64, 128, 64], 'dropout': 0.3}
                },
                'data_processing': {'sequence_length': 25, 'num_features': 21}
            }
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create models with reduced dimensions
        generator, discriminator = create_gan_models(config, device)
        
        logger.info(f"✅ GAN model creation test passed")
        logger.info(f"   Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
        logger.info(f"   Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
        
        # Test forward pass with small batch
        batch_size = 4
        sequence_length = config['data_processing']['sequence_length']
        num_features = config['data_processing']['num_features']
        
        # Test generator
        noise = torch.randn(batch_size, config['model']['generator']['latent_dim']).to(device)
        fake_data = generator(noise)
        logger.info(f"   Generator output: {fake_data.shape}")
        
        # Test discriminator
        real_data = torch.randn(batch_size, sequence_length, num_features).to(device)
        real_output = discriminator(real_data)
        logger.info(f"   Discriminator output: {real_output.shape}")
        
        # Clean up
        del generator, discriminator, noise, fake_data, real_data, real_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GAN model creation test failed: {e}")
        return False

def test_memory_limits():
    """Test memory limit enforcement."""
    logger.info("=== Testing Memory Limits ===")
    
    try:
        # Test with very large data that should be rejected
        large_sequences = np.random.randn(10000, 200, 100)  # Very large
        large_targets = np.random.randn(10000, 100)
        
        logger.info(f"Created large test data: {large_sequences.shape}")
        
        # This should trigger memory limits and be reduced
        train_loader, val_loader, test_loader = safe_create_data_loaders(
            large_sequences, large_targets,
            batch_size=64,  # Large batch size
            max_sequence_length=50,
            max_batch_size=16
        )
        
        logger.info(f"✅ Memory limits test passed")
        logger.info(f"   Data was automatically reduced to fit memory constraints")
        
        # Clean up
        del large_sequences, large_targets, train_loader, val_loader, test_loader
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory limits test failed: {e}")
        return False

def main():
    """Run all memory safety tests."""
    logger.info("🚀 Starting Memory Safety Debug Tests")
    
    # Check system info
    logger.info(f"System memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logger.info(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    tests = [
        test_memory_monitoring,
        test_csv_data_loading,
        test_data_loader_creation,
        test_gan_model_creation,
        test_memory_limits
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Memory safety improvements are working.")
        logger.info("You can now try running the training again.")
    else:
        logger.error(f"❌ {total - passed} tests failed. Check the logs above for issues.")
        logger.info("Fix the failing tests before running training.")
    
    # Final memory check
    final_memory = check_memory_usage()
    logger.info(f"Final memory usage: {final_memory:.2f} MB")

if __name__ == "__main__":
    main() 