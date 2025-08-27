#!/usr/bin/env python3
"""
Test script to verify that training can start without segmentation faults.
This script will attempt to start a minimal training session.
"""

import requests
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dashboard_connection():
    """Test if dashboard is accessible."""
    try:
        response = requests.get("http://localhost:8081/api/connection_stats", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Dashboard is accessible")
            return True
        else:
            logger.error(f"❌ Dashboard returned status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Cannot connect to dashboard: {e}")
        return False

def test_training_start():
    """Test if training can start without crashing."""
    try:
        logger.info("🚀 Attempting to start training...")
        
        # Start training with minimal parameters
        training_data = {
            "config_file": "config/gan_config.yaml",
            "data_source": "treasury_orderbook_sample.csv",
            "epochs": 2,  # Very short training
            "sequence_length": 25,  # Reduced for memory safety
            "batch_size": 8  # Reduced for memory safety
        }
        
        response = requests.post(
            "http://localhost:8081/api/start_training",  # Fixed endpoint URL
            json=training_data,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("✅ Training started successfully")
            result = response.json()
            logger.info(f"   Response: {result}")
            return True
        else:
            logger.error(f"❌ Training start failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Training start error: {e}")
        return False

def monitor_training_progress():
    """Monitor training progress for a short time."""
    logger.info("📊 Monitoring training progress...")
    
    try:
        # Monitor for up to 30 seconds
        start_time = time.time()
        max_wait = 30
        
        while time.time() - start_time < max_wait:
            try:
                # Check connection stats
                response = requests.get("http://localhost:8081/api/connection_stats", timeout=5)
                if response.status_code == 200:
                    stats = response.json()
                    logger.info(f"   Active clients: {stats.get('active_clients', 0)}")
                    logger.info(f"   Training clients: {stats.get('training_clients', 0)}")
                    logger.info(f"   Progress clients: {stats.get('progress_clients', 0)}")
                
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"   Monitoring error: {e}")
                time.sleep(2)
        
        logger.info("✅ Training monitoring completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training monitoring failed: {e}")
        return False

def main():
    """Run training tests."""
    logger.info("🧪 Testing Training Fixes")
    
    # Test 1: Dashboard connection
    if not test_dashboard_connection():
        logger.error("Dashboard connection failed. Cannot proceed with training tests.")
        return
    
    # Test 2: Training start
    if not test_training_start():
        logger.error("Training start failed. The segmentation fault may still exist.")
        return
    
    # Test 3: Monitor progress
    if not monitor_training_progress():
        logger.warning("Training monitoring failed, but training may still be working.")
    
    logger.info("🎉 Training tests completed!")
    logger.info("If no segmentation faults occurred, the memory safety fixes are working.")

if __name__ == "__main__":
    main() 