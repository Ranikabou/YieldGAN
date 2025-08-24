#!/usr/bin/env python3
"""
Test script to demonstrate separate SSE channels for training and progress data.
This shows how the two data types are now sent to different endpoints.
"""

import requests
import json
import time
from datetime import datetime
import random

class SeparateChannelTester:
    def __init__(self, server_url="http://localhost:8765"):
        self.server_url = server_url
        self.epoch = 1
        self.total_epochs = 10  # Increased from 3 to 10 epochs
    
    def send_training_data(self):
        """Send training metrics to training channel."""
        training_data = {
            "type": "training_update",
            "data": {
                "epoch": self.epoch,
                "total_epochs": self.total_epochs,
                "generator_loss": round(random.uniform(0.5, 1.2), 4),
                "discriminator_loss": round(random.uniform(0.6, 1.1), 4),
                "real_scores": round(random.uniform(0.4, 0.9), 4),
                "fake_scores": round(random.uniform(0.1, 0.5), 4)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/training_data",
                json=training_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"🎯 Training data sent to training channel")
                print(f"   Epoch: {self.epoch}/{self.total_epochs}")
                print(f"   Generator Loss: {training_data['data']['generator_loss']}")
                print(f"   Discriminator Loss: {training_data['data']['discriminator_loss']}")
                print(f"   Real Scores: {training_data['data']['real_scores']}")
                print(f"   Fake Scores: {training_data['data']['fake_scores']}")
                print(f"   Channel: {result.get('channel', 'unknown')}")
                print(f"   Clients: {result.get('clients', 0)}")
                return True
            else:
                print(f"   ❌ Training data failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Training data error: {e}")
            return False
    
    def send_progress_data(self, progress_percent):
        """Send progress update to progress channel."""
        progress_data = {
            "type": "progress",
            "epoch": self.epoch,
            "progress_percent": progress_percent,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/progress_data",
                json=progress_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"📊 Progress {progress_percent}% sent to progress channel")
                print(f"   Channel: {result.get('channel', 'unknown')}")
                print(f"   Clients: {result.get('clients', 0)}")
                return True
            else:
                print(f"   ❌ Progress {progress_percent}% failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Progress {progress_percent}% error: {e}")
            return False
    
    def run_test(self):
        """Run a complete test showing separate channels."""
        print("🚀 Testing Separate SSE Channels")
        print("=" * 60)
        print("📊 This demonstrates how training and progress data use different channels")
        print("🎯 Training data → /training_data endpoint")
        print("📈 Progress data → /progress_data endpoint")
        print("=" * 60)
        
        # Check if SSE server is running
        try:
            response = requests.get(self.server_url, timeout=5)
            if response.status_code == 200:
                print("✅ SSE server is running")
            else:
                print(f"❌ SSE server returned status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ SSE server not accessible: {e}")
            print("💡 Start the SSE server with: python simple_sse_server.py")
            return False
        
        print(f"\n🔄 Running {self.total_epochs} epochs to demonstrate separate channels...")
        
        for epoch in range(1, self.total_epochs + 1):
            self.epoch = epoch
            print(f"\n📊 Epoch {epoch}/{self.total_epochs}")
            
            # Send progress updates to progress channel
            print("   📈 Sending progress updates to progress channel...")
            for progress in [0, 25, 50, 75, 100]:
                self.send_progress_data(progress)
                time.sleep(1.0)  # Increased from 0.2 to 1.0 seconds
            
            # Send training metrics to training channel
            print("   🎯 Sending training metrics to training channel...")
            self.send_training_data()
            
            time.sleep(3.0)  # Increased from 1.0 to 3.0 seconds between epochs
        
        print("\n✅ Separate channel test completed!")
        print("\n📋 Summary:")
        print("   🎯 Training data uses /training_data endpoint")
        print("   📈 Progress data uses /progress_data endpoint")
        print("   🔌 Each channel has its own SSE connection")
        print("   📊 Data types don't override each other")

def main():
    """Main function."""
    tester = SeparateChannelTester()
    tester.run_test()

if __name__ == "__main__":
    main() 