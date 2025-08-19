#!/usr/bin/env python3
"""
Test script for Treasury Curve GAN project.
Verifies basic functionality and project setup.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        import torch
        import numpy as np
        import pandas as pd
        import yfinance as yf
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.preprocessing import StandardScaler
        
        logger.info("✓ All external dependencies imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import external dependency: {e}")
        return False
    
    try:
        # Import project modules
        from utils.data_utils import TreasuryDataProcessor
        from models.gan_models import create_gan_models
        from training.trainer import GANTrainer
        from evaluation.metrics import TreasuryDataEvaluator
        
        logger.info("✓ All project modules imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import project module: {e}")
        return False
    
    return True

def test_data_processor():
    """Test data processor functionality."""
    logger.info("Testing data processor...")
    
    try:
        from utils.data_utils import TreasuryDataProcessor
        
        processor = TreasuryDataProcessor(
            instruments=['2Y', '5Y', '10Y', '10Y', '30Y', 'SOFR'],
            sequence_length=50
        )
        logger.info("✓ Data processor created successfully")
        
        # Test data fetching (small date range)
        sequences, targets, scaler = processor.prepare_data('2023-01-01', '2023-02-01')
        
        if sequences is not None and targets is not None:
            logger.info(f"✓ Data processing successful: {sequences.shape}, {targets.shape}")
            return True
        else:
            logger.error("✗ Data processing failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Data processor test failed: {e}")
        return False

def test_model_creation():
    """Test GAN model creation."""
    logger.info("Testing model creation...")
    
    try:
        from models.gan_models import create_gan_models
        import torch
        
        # Simple config for testing
        test_config = {
            'model': {
                'generator': {
                    'latent_dim': 50,
                    'hidden_dims': [128, 256, 128],
                    'dropout': 0.3
                },
                'discriminator': {
                    'hidden_dims': [128, 256, 128],
                    'dropout': 0.3
                }
            },
            'data': {
                'num_features': 25,
                'sequence_length': 50
            }
        }
        
        # Test model creation
        generator, discriminator = create_gan_models(test_config, torch.device('cpu'))
        
        # Test forward pass
        batch_size = 4
        test_noise = torch.randn(batch_size, 50)
        fake_data = generator(test_noise)
        real_score = discriminator(fake_data)
        
        logger.info(f"✓ Models created successfully: {fake_data.shape}, {real_score.shape}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model creation test failed: {e}")
        return False

def test_training_setup():
    """Test training setup."""
    logger.info("Testing training setup...")
    
    try:
        from training.trainer import GANTrainer
        import torch
        
        # Simple config for testing
        test_config = {
            'gan_type': 'standard',
            'model': {
                'generator': {
                    'latent_dim': 50,
                    'hidden_dims': [128, 256, 128],
                    'dropout': 0.3
                },
                'discriminator': {
                    'hidden_dims': [128, 256, 128],
                    'dropout': 0.3
                }
            },
            'data': {
                'num_features': 25,
                'sequence_length': 50
            },
            'training': {
                'epochs': 1,
                'learning_rate_generator': 0.0002,
                'learning_rate_discriminator': 0.0002,
                'beta1': 0.5,
                'beta2': 0.999,
                'critic_iterations': 5,
                'lambda_gp': 10,
                'patience': 10,
                'min_delta': 0.001
            },
            'loss_weights': {
                'generator_loss': 1.0,
                'discriminator_loss': 1.0,
                'feature_matching': 0.1
            },
            'instruments': ['2Y', '5Y', '10Y', '30Y', 'SOFR']
        }
        
        # Create trainer
        trainer = GANTrainer(test_config, torch.device('cpu'))
        logger.info("✓ Trainer created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Training setup test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting Treasury Curve GAN project tests...")
    
    tests = [
        test_imports,
        test_data_processor,
        test_model_creation,
        test_training_setup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        logger.info("-" * 50)
    
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Project is ready to use.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 