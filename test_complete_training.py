#!/usr/bin/env python3
"""
Comprehensive test to simulate a complete training run and verify UI updates.
This will help identify if the UI is updating systematically as training progresses.
"""

import requests
import time
import json
from datetime import datetime

def send_training_data(epoch, gen_loss, disc_loss, real_scores=0.8, fake_scores=0.2):
    """Send training data to dashboard."""
    training_data = {
        "type": "training_update",
        "data": {
            "epoch": epoch,
            "total_epochs": 10,
            "generator_loss": gen_loss,
            "discriminator_loss": disc_loss,
            "real_scores": real_scores,
            "fake_scores": fake_scores
        },
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(
            "http://localhost:8081/training_data",
            json=training_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"🎯 Epoch {epoch}: Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f} - Sent to {result.get('clients', 0)} clients")
            return True
        else:
            print(f"❌ Failed to send training data for epoch {epoch}: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error sending training data for epoch {epoch}: {e}")
        return False

def send_progress_data(epoch, progress_percent):
    """Send progress data to dashboard."""
    progress_data = {
        "type": "progress",
        "epoch": epoch,
        "progress_percent": progress_percent,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(
            "http://localhost:8081/progress_data",
            json=progress_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Epoch {epoch}: Progress {progress_percent}% - Sent to {result.get('clients', 0)} clients")
            return True
        else:
            print(f"❌ Failed to send progress data for epoch {epoch}: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error sending progress data for epoch {epoch}: {e}")
        return False

def simulate_complete_training():
    """Simulate a complete training run with systematic updates."""
    print("🚀 Starting Complete Training Simulation")
    print("=" * 50)
    
    # Test dashboard connectivity
    try:
        response = requests.get("http://localhost:8081/api/training_status", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard is accessible")
        else:
            print(f"❌ Dashboard returned status {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to dashboard: {e}")
        return
    
    print("\n📊 Simulating 10 epochs of training...")
    print("Each epoch will send progress updates and training metrics")
    print("Check the dashboard UI to see if updates are systematic")
    print("-" * 50)
    
    # Simulate 10 epochs
    for epoch in range(1, 11):
        print(f"\n📈 Epoch {epoch}/10")
        
        # Send progress start
        send_progress_data(epoch, 0)
        time.sleep(0.5)
        
        # Send progress updates during training
        for progress in [25, 50, 75]:
            send_progress_data(epoch, progress)
            time.sleep(0.3)
        
        # Calculate realistic training metrics
        # Generator loss should generally decrease over time
        base_gen_loss = 0.8 - epoch * 0.03
        gen_loss = max(0.1, base_gen_loss + (epoch % 3) * 0.02 - (epoch % 2) * 0.01)
        
        # Discriminator loss should also decrease but with some variation
        base_disc_loss = 0.7 - epoch * 0.025
        disc_loss = max(0.1, base_disc_loss + (epoch % 2) * 0.015 - (epoch % 3) * 0.01)
        
        # Scores should improve over time
        real_scores = max(0.5, 0.8 - epoch * 0.01 + (epoch % 2) * 0.005)
        fake_scores = min(0.5, 0.2 + epoch * 0.01 - (epoch % 3) * 0.005)
        
        # Send training metrics
        send_training_data(
            epoch, 
            gen_loss, 
            disc_loss,
            real_scores,
            fake_scores
        )
        time.sleep(0.5)
        
        # Send progress completion
        send_progress_data(epoch, 100)
        time.sleep(0.5)
        
        print(f"   ✅ Epoch {epoch} completed")
        print(f"   📊 Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
        print(f"   🎯 Real Scores: {real_scores:.4f}, Fake Scores: {fake_scores:.4f}")
        
        # Small delay between epochs
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("✅ Training simulation completed!")
    print("\n📋 Expected UI Behavior:")
    print("1. Training Status should show 'Running'")
    print("2. Current Epoch should increment from 1 to 10")
    print("3. Generator Loss should show decreasing values")
    print("4. Discriminator Loss should show decreasing values")
    print("5. Training charts should update with new data points")
    print("6. Progress bar should show completion for each epoch")
    print("\n🔍 Check the dashboard UI to verify these updates")

if __name__ == "__main__":
    simulate_complete_training() 