#!/usr/bin/env python3
"""
Minimal test script to verify train_gan_csv.py dashboard integration.
This script runs a very short training session to test the channels.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from train_gan_csv import DashboardChannelSender
import time

def test_minimal_training():
    """Test minimal training with dashboard integration."""
    print("🧪 Testing Minimal Training with Dashboard Integration")
    print("=" * 60)
    
    # Initialize dashboard sender
    dashboard_sender = DashboardChannelSender("http://localhost:8081")
    dashboard_sender.set_total_epochs(3)
    
    print("✅ Dashboard sender initialized")
    print("💡 Make sure the dashboard is running on http://localhost:8081")
    
    # Test sending some training data
    print("\n🎯 Sending test training data...")
    for epoch in range(3):
        # Send progress start
        dashboard_sender.send_progress_data(epoch, 0)
        time.sleep(0.5)
        
        # Send progress updates
        for progress in [25, 50, 75]:
            dashboard_sender.send_progress_data(epoch, progress)
            time.sleep(0.2)
        
        # Send progress completion
        dashboard_sender.send_progress_data(epoch, 100)
        time.sleep(0.5)
        
        # Send training metrics
        dashboard_sender.send_training_data(
            epoch + 1,
            generator_loss=0.8 - epoch * 0.1,
            discriminator_loss=0.7 - epoch * 0.05,
            real_scores=0.9 - epoch * 0.02,
            fake_scores=0.1 + epoch * 0.03
        )
        time.sleep(0.5)
        
        print(f"   ✅ Epoch {epoch + 1} completed")
    
    print("\n✅ Minimal training test completed!")
    print("📊 Check your dashboard to see the real-time updates!")

if __name__ == "__main__":
    test_minimal_training() 