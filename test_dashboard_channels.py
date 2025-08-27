#!/usr/bin/env python3
"""
Test script to demonstrate the modified GAN dashboard with separate SSE channels and log file reading.
This shows how the dashboard now works similar to test_separate_channels.py but with log file monitoring.
"""

import requests
import json
import time
from datetime import datetime
import random
import os

class DashboardChannelTester:
    def __init__(self, dashboard_url=None):
        if dashboard_url is None:
            try:
                from utils.port_manager import get_dashboard_url
                self.dashboard_url = get_dashboard_url()
            except ImportError:
                self.dashboard_url = "http://localhost:8083"  # Updated default
        else:
            self.dashboard_url = dashboard_url
        self.epoch = 1
        self.total_epochs = 10
    
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
                f"{self.dashboard_url}/training_data",
                json=training_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"🎯 Training data sent to dashboard training channel")
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
                f"{self.dashboard_url}/progress_data",
                json=progress_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"📊 Progress {progress_percent}% sent to dashboard progress channel")
                print(f"   Channel: {result.get('channel', 'unknown')}")
                print(f"   Clients: {result.get('clients', 0)}")
                return True
            else:
                print(f"   ❌ Progress {progress_percent}% failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Progress {progress_percent}% error: {e}")
            return False
    
    def create_sample_log_file(self):
        """Create a sample log file to test log file reading."""
        log_content = f"""
[INFO] Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
[INFO] Epoch 1/10 Generator Loss: 1.2345 Discriminator Loss: 0.8765
[INFO] Progress: 10%
[INFO] Epoch 2/10 Generator Loss: 1.1234 Discriminator Loss: 0.7654
[INFO] Progress: 20%
[INFO] Epoch 3/10 Generator Loss: 1.0123 Discriminator Loss: 0.6543
[INFO] Progress: 30%
[INFO] Epoch 4/10 Generator Loss: 0.9012 Discriminator Loss: 0.5432
[INFO] Progress: 40%
[INFO] Epoch 5/10 Generator Loss: 0.8901 Discriminator Loss: 0.4321
[INFO] Progress: 50%
        """.strip()
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Write sample log file
        with open('logs/sample_training.log', 'w') as f:
            f.write(log_content)
        
        print(f"📝 Created sample log file: logs/sample_training.log")
        return 'logs/sample_training.log'
    
    def append_to_log_file(self, log_file, epoch, gen_loss, disc_loss, progress):
        """Append new training data to the log file."""
        new_log_entry = f"""
[INFO] Epoch {epoch}/10 Generator Loss: {gen_loss:.4f} Discriminator Loss: {disc_loss:.4f}
[INFO] Progress: {progress}%
        """.strip()
        
        with open(log_file, 'a') as f:
            f.write('\n' + new_log_entry)
        
        print(f"📝 Appended to log file: Epoch {epoch}, Progress {progress}%")
    
    def run_test(self):
        """Run a complete test showing separate channels and log file reading."""
        print("🚀 Testing Modified GAN Dashboard with Separate SSE Channels")
        print("=" * 70)
        print("📊 This demonstrates how the dashboard now works with:")
        print("   🎯 Separate training and progress channels")
        print("   📝 Log file reading and monitoring")
        print("   📡 Real-time updates via SSE")
        print("=" * 70)
        
        # Check if dashboard is running
        try:
            response = requests.get(self.dashboard_url, timeout=5)
            if response.status_code == 200:
                print("✅ GAN Dashboard is running")
            else:
                print(f"❌ Dashboard returned status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Dashboard not accessible: {e}")
            print("💡 Start the dashboard with: python gan_dashboard.py")
            return False
        
        # Create sample log file
        print(f"\n📝 Creating sample log file for testing...")
        log_file = self.create_sample_log_file()
        
        print(f"\n🔄 Running {self.total_epochs} epochs to demonstrate separate channels and log reading...")
        
        for epoch in range(1, self.total_epochs + 1):
            self.epoch = epoch
            print(f"\n📊 Epoch {epoch}/{self.total_epochs}")
            
            # Send progress updates to progress channel
            print("   📈 Sending progress updates to progress channel...")
            for progress in [0, 25, 50, 75, 100]:
                self.send_progress_data(progress)
                time.sleep(0.5)
            
            # Send training metrics to training channel
            print("   🎯 Sending training metrics to training channel...")
            self.send_training_data()
            
            # Append to log file to test log file reading
            gen_loss = round(random.uniform(0.5, 1.2), 4)
            disc_loss = round(random.uniform(0.6, 1.1), 4)
            progress = epoch * 10
            self.append_to_log_file(log_file, epoch, gen_loss, disc_loss, progress)
            
            time.sleep(2.0)  # Wait between epochs
        
        print("\n✅ Dashboard channel test completed!")
        print("\n📋 Summary:")
        print("   🎯 Training data uses /training_data endpoint → training clients only")
        print("   📈 Progress data uses /progress_data endpoint → progress clients only")
        print("   📝 Log files are monitored in real-time")
        print("   📡 Each channel has its own SSE connection")
        print("   🔄 Dashboard can read from both live data and log files")
        print("\n💡 Open the dashboard in your browser to see the real-time updates!")

def main():
    """Main function."""
    tester = DashboardChannelTester()
    tester.run_test()

if __name__ == "__main__":
    main() 