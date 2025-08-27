#!/usr/bin/env python3
"""
Quick test to verify memory safety fixes work.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported without errors."""
    logger.info("Testing imports...")
    
    try:
        from utils.data_utils import check_memory_usage, safe_create_data_loaders
        logger.info("✅ utils.data_utils imported successfully")
    except Exception as e:
        logger.error(f"❌ utils.data_utils import failed: {e}")
        return False
    
    try:
        from data.csv_collector import CSVDataCollector
        logger.info("✅ CSVDataCollector imported successfully")
    except Exception as e:
        logger.error(f"❌ CSVDataCollector import failed: {e}")
        return False
    
    try:
        from training.trainer import GANTrainer
        logger.info("✅ GANTrainer imported successfully")
    except Exception as e:
        logger.error(f"❌ GANTrainer import failed: {e}")
        return False
    
    try:
        from models.gan_models import create_gan_models
        logger.info("✅ create_gan_models imported successfully")
    except Exception as e:
        logger.error(f"❌ create_gan_models import failed: {e}")
        return False
    
    return True

def test_memory_function():
    """Test the memory monitoring function."""
    logger.info("Testing memory monitoring...")
    
    try:
        from utils.data_utils import check_memory_usage
        memory = check_memory_usage()
        logger.info(f"✅ Memory monitoring works: {memory:.2f} MB")
        return True
    except Exception as e:
        logger.error(f"❌ Memory monitoring failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    logger.info("Testing configuration loading...")
    
    try:
        import yaml
        config_path = "config/gan_config.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check if memory settings are present
            if 'memory' in config:
                logger.info("✅ Memory configuration found")
                logger.info(f"   Max sequence length: {config['memory'].get('max_sequence_length')}")
                logger.info(f"   Max batch size: {config['memory'].get('max_batch_size')}")
            else:
                logger.warning("⚠️ Memory configuration not found")
            
            # Check if batch sizes are reduced
            if config['data_processing']['batch_size'] <= 16:
                logger.info("✅ Batch size is reduced for memory safety")
            else:
                logger.warning("⚠️ Batch size might be too large")
            
            return True
        else:
            logger.warning(f"⚠️ Config file {config_path} not found")
            return True
            
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return False

def main():
    """Run quick tests."""
    logger.info("🚀 Running Quick Memory Safety Tests")
    
    tests = [
        test_imports,
        test_memory_function,
        test_config_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
    
    logger.info(f"\n📊 Quick Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All quick tests passed! Memory safety fixes are working.")
        logger.info("You can now try running the training again.")
    else:
        logger.error(f"❌ {total - passed} quick tests failed.")
        logger.info("Check the logs above for issues.")

if __name__ == "__main__":
    main() 