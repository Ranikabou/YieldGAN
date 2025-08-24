#!/usr/bin/env python3
"""
Test script to verify that train_gan_csv.py can send data to dashboard channels.
This script tests the DashboardChannelSender class independently.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from train_gan_csv import DashboardChannelSender

def test_dashboard_channels():
    """Test the dashboard channel integration."""
    print("🧪 Testing Dashboard Channel Integration")
    print("=" * 50)
    
    # Initialize dashboard sender
    dashboard_sender = DashboardChannelSender("http://localhost:8081")
    dashboard_sender.set_total_epochs(5)
    
    print("✅ Dashboard sender initialized")
    
    # Test progress updates
    print("\n📊 Testing progress updates...")
    for epoch in range(5):
        for progress in [0, 25, 50, 75, 100]:
            success = dashboard_sender.send_progress_data(epoch, progress)
            if success:
                print(f"   ✅ Epoch {epoch}, Progress {progress}% sent")
            else:
                print(f"   ❌ Epoch {epoch}, Progress {progress}% failed")
            time.sleep(0.2)  # Small delay
    
    # Test training data updates
    print("\n🎯 Testing training data updates...")
    for epoch in range(5):
        success = dashboard_sender.send_training_data(
            epoch + 1,
            generator_loss=0.5 + epoch * 0.1,
            discriminator_loss=0.6 + epoch * 0.05,
            real_scores=0.8 - epoch * 0.02,
            fake_scores=0.2 + epoch * 0.03
        )
        if success:
            print(f"   ✅ Epoch {epoch + 1} training data sent")
        else:
            print(f"   ❌ Epoch {epoch + 1} training data failed")
        time.sleep(0.2)  # Small delay
    
    print("\n✅ Dashboard channel integration test completed!")
    print("\n💡 Make sure the dashboard is running on http://localhost:8081")
    print("   You should see the training data and progress updates in real-time!")

if __name__ == "__main__":
    test_dashboard_channels() 