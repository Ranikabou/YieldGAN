#!/usr/bin/env python3
"""
Simplified GAN training script with dashboard integration.
This version focuses on demonstrating the dashboard channels without complex training.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
import time
from datetime import datetime
import requests

# Add project root to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardChannelSender:
    """Class to send training data and progress to dashboard channels."""
    
    def __init__(self, dashboard_url=None):
        if dashboard_url is None:
            try:
                from utils.port_manager import get_dashboard_url
                self.dashboard_url = get_dashboard_url()
            except ImportError:
                self.dashboard_url = "http://localhost:8083"  # Updated default
        else:
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

def simulate_training(dashboard_sender, epochs=10):
    """
    Simulate GAN training and send data to dashboard.
    
    Args:
        dashboard_sender: DashboardChannelSender instance
        epochs: Number of epochs to simulate
    """
    logger.info(f"Starting simulated training for {epochs} epochs...")
    
    dashboard_sender.set_total_epochs(epochs)
    
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        
        # Send progress start
        dashboard_sender.send_progress_data(epoch, 0)
        time.sleep(0.1)
        
        # Simulate training progress with granular updates
        for progress in range(1, 100):
            dashboard_sender.send_progress_data(epoch, progress)
            time.sleep(0.02)  # Small delay to simulate realistic progress
        
        # Simulate training completion
        dashboard_sender.send_progress_data(epoch, 100)
        time.sleep(0.1)
        
        # Generate simulated training metrics
        gen_loss = 0.8 - epoch * 0.05 + (epoch % 3) * 0.02
        disc_loss = 0.7 - epoch * 0.03 + (epoch % 2) * 0.01
        real_scores = 0.9 - epoch * 0.01 + (epoch % 4) * 0.005
        fake_scores = 0.1 + epoch * 0.02 - (epoch % 3) * 0.01
        
        # Ensure values are reasonable
        gen_loss = max(0.1, min(1.5, gen_loss))
        disc_loss = max(0.1, min(1.5, disc_loss))
        real_scores = max(0.3, min(1.0, real_scores))
        fake_scores = max(0.0, min(0.7, fake_scores))
        
        # Send training data to dashboard
        dashboard_sender.send_training_data(
            epoch + 1,
            gen_loss,
            disc_loss,
            real_scores,
            fake_scores
        )
        
        logger.info(f"Epoch {epoch + 1} completed - Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
        time.sleep(1.0)  # Wait between epochs
    
    logger.info("Simulated training completed!")

def main():
    """Main function for simplified GAN training simulation."""
    parser = argparse.ArgumentParser(description='Simplified GAN training with dashboard integration')
    parser.add_argument('--config', type=str, default='config/gan_config.yaml',
                       help='Path to GAN configuration file')
    parser.add_argument('--data', type=str, default='treasury_orderbook_sample.csv',
                       help='Data source CSV file to use for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs to simulate')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Initialize dashboard sender
    # Use port manager for automatic dashboard URL detection
    try:
        from utils.port_manager import get_dashboard_url
        dashboard_url = get_dashboard_url()
    except ImportError:
        dashboard_url = config.get('dashboard', {}).get('url', "http://localhost:8083")
    dashboard_sender = DashboardChannelSender(dashboard_url)
    
    logger.info(f"Using data source: {args.data}")
    logger.info(f"Dashboard URL: {dashboard_url}")
    
    # Simulate training with dashboard integration
    simulate_training(dashboard_sender, args.epochs)
    
    logger.info("Simplified GAN training simulation completed successfully!")

if __name__ == "__main__":
    main() 