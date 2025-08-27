#!/usr/bin/env python3
"""
Test script to verify SSE data flow in the GAN dashboard.
This will send test data to both training and progress channels.
"""

import requests
import json
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training_channel():
    """Test sending training data to the dashboard."""
    url = "http://localhost:8081/training_data"
    
    test_data = {
        "type": "training_update",
        "data": {
            "epoch": 1,
            "total_epochs": 10,
            "generator_loss": 0.75,
            "discriminator_loss": 0.65,
            "real_scores": 0.85,
            "fake_scores": 0.25
        },
        "timestamp": "2024-01-01T12:00:00"
    }
    
    try:
        response = requests.post(url, json=test_data)
        logger.info(f"Training data response: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error sending training data: {e}")
        return False

def test_progress_channel():
    """Test sending progress data to the dashboard."""
    url = "http://localhost:8081/progress_data"
    
    test_data = {
        "type": "progress",
        "epoch": 1,
        "progress_percent": 50.0,
        "timestamp": "2024-01-01T12:00:00"
    }
    
    try:
        response = requests.post(url, json=test_data)
        logger.info(f"Progress data response: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error sending progress data: {e}")
        return False

def test_multiple_updates():
    """Test sending multiple updates to see if there's data duplication."""
    logger.info("Testing multiple updates to identify duplication issues...")
    
    # Send 5 training updates
    for epoch in range(1, 6):
        training_data = {
            "type": "training_update",
            "data": {
                "epoch": epoch,
                "total_epochs": 10,
                "generator_loss": 0.8 - epoch * 0.05,
                "discriminator_loss": 0.7 - epoch * 0.03,
                "real_scores": 0.8 + epoch * 0.01,
                "fake_scores": 0.2 + epoch * 0.02
            },
            "timestamp": f"2024-01-01T12:0{epoch}:00"
        }
        
        progress_data = {
            "type": "progress",
            "epoch": epoch,
            "progress_percent": 100.0,  # Each epoch completes
            "timestamp": f"2024-01-01T12:0{epoch}:30"
        }
        
        # Send training data
        success1 = test_training_channel_with_data(training_data)
        time.sleep(0.5)
        
        # Send progress data
        success2 = test_progress_channel_with_data(progress_data)
        time.sleep(0.5)
        
        if success1 and success2:
            logger.info(f"✅ Epoch {epoch} data sent successfully")
        else:
            logger.error(f"❌ Epoch {epoch} data failed")

def test_training_channel_with_data(data):
    """Test sending specific training data."""
    url = "http://localhost:8081/training_data"
    try:
        response = requests.post(url, json=data)
        logger.info(f"Training data response: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error sending training data: {e}")
        return False

def test_progress_channel_with_data(data):
    """Test sending specific progress data."""
    url = "http://localhost:8081/progress_data"
    try:
        response = requests.post(url, json=data)
        logger.info(f"Progress data response: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error sending progress data: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🧪 Starting SSE data flow test...")
    
    # Test individual channels
    logger.info("1. Testing training channel...")
    training_success = test_training_channel()
    
    logger.info("2. Testing progress channel...")
    progress_success = test_progress_channel()
    
    if training_success and progress_success:
        logger.info("✅ Basic SSE tests passed!")
        
        # Test multiple updates
        logger.info("3. Testing multiple updates...")
        test_multiple_updates()
        
    else:
        logger.error("❌ Basic SSE tests failed!")
        if not training_success:
            logger.error("   - Training channel failed")
        if not progress_success:
            logger.error("   - Progress channel failed")
    
    logger.info("🏁 SSE test completed. Check the dashboard UI for data updates.")

if __name__ == "__main__":
    main() 