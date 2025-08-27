#!/usr/bin/env python3
"""
Test script to debug UI update issues in the GAN dashboard.
Simulates training data and checks dashboard response.
"""

import requests
import time
import json
from datetime import datetime

def test_dashboard_connection():
    """Test basic dashboard connectivity."""
    try:
        response = requests.get("http://localhost:8082/api/training_status", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard is accessible")
            return True
        else:
            print(f"❌ Dashboard returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to dashboard: {e}")
        return False

def test_sse_connection():
    """Test SSE connection to training channel."""
    try:
        # This is a simple test - in real usage, the browser would handle this
        print("🔍 Testing SSE endpoint availability...")
        response = requests.get("http://localhost:8082/events/training", timeout=5)
        if response.status_code == 200:
            print("✅ SSE endpoint is accessible")
            return True
        else:
            print(f"❌ SSE endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to SSE endpoint: {e}")
        return False

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
            "http://localhost:8082/training_data",
            json=training_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"🎯 Training data sent for epoch {epoch}: {result}")
            return True
        else:
            print(f"❌ Failed to send training data: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error sending training data: {e}")
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
            "http://localhost:8082/progress_data",
            json=progress_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Progress data sent for epoch {epoch}: {result}")
            return True
        else:
            print(f"❌ Failed to send progress data: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error sending progress data: {e}")
        return False

def simulate_training():
    """Simulate a complete training run."""
    print("🚀 Starting training simulation...")
    
    # Test basic connectivity first
    if not test_dashboard_connection():
        print("❌ Dashboard not accessible, cannot proceed")
        return
    
    if not test_sse_connection():
        print("❌ SSE endpoint not accessible, cannot proceed")
        return
    
    print("✅ All endpoints accessible, starting simulation...")
    
    # Simulate 5 epochs of training
    for epoch in range(1, 6):
        print(f"\n📊 Epoch {epoch}/5")
        
        # Send progress start
        send_progress_data(epoch, 0)
        time.sleep(0.5)
        
        # Send training metrics
        gen_loss = 0.8 - epoch * 0.05 + (epoch % 3) * 0.02
        disc_loss = 0.7 - epoch * 0.03 + (epoch % 2) * 0.01
        
        send_training_data(
            epoch, 
            max(0.1, gen_loss), 
            max(0.1, disc_loss),
            0.9 - epoch * 0.01,
            0.1 + epoch * 0.02
        )
        time.sleep(0.5)
        
        # Send progress completion
        send_progress_data(epoch, 100)
        time.sleep(0.5)
        
        print(f"   Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
    
    print("\n✅ Training simulation completed!")

def check_dashboard_logs():
    """Check dashboard logs for any errors."""
    try:
        response = requests.get("http://localhost:8082/", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard main page is accessible")
            # Check if there are any obvious errors in the HTML
            if "error" in response.text.lower():
                print("⚠️  Dashboard page contains error indicators")
            else:
                print("✅ Dashboard page looks healthy")
        else:
            print(f"❌ Dashboard main page returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot access dashboard main page: {e}")

if __name__ == "__main__":
    print("🔍 GAN Dashboard UI Debug Test")
    print("=" * 50)
    
    # Check dashboard health
    check_dashboard_logs()
    
    # Run training simulation
    simulate_training()
    
    print("\n📋 Debug Summary:")
    print("1. Check browser console for JavaScript errors")
    print("2. Verify SSE connections are established")
    print("3. Check if training data is being received by the UI")
    print("4. Look for any network errors in browser dev tools") 