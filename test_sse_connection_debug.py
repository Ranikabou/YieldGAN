#!/usr/bin/env python3
"""
Debug script to test SSE connections directly using requests.
This will help identify why EventSource connections aren't working.
"""

import requests
import time
import threading
import json

def test_sse_connection(url, channel_name):
    """Test SSE connection using requests stream."""
    print(f"🔍 Testing SSE connection to {url}")
    
    try:
        # Use requests to stream SSE data
        headers = {
            'Accept': 'text/event-stream',
            'Cache-Control': 'no-cache'
        }
        
        response = requests.get(url, headers=headers, stream=True)
        print(f"✅ {channel_name} connection established. Status: {response.status_code}")
        
        # Read the SSE stream
        for line in response.iter_lines(decode_unicode=True):
            if line:
                print(f"📡 {channel_name}: {line}")
                # Stop after receiving a few messages
                if "data:" in line:
                    data_count = getattr(test_sse_connection, f'{channel_name}_count', 0) + 1
                    setattr(test_sse_connection, f'{channel_name}_count', data_count)
                    if data_count >= 3:  # Stop after 3 data messages
                        break
                        
    except Exception as e:
        print(f"❌ Error connecting to {channel_name}: {e}")

def test_connections():
    """Test both SSE channels."""
    print("🧪 Starting SSE connection debug test...")
    
    # Test training channel in a separate thread
    training_thread = threading.Thread(
        target=test_sse_connection, 
        args=("http://localhost:8081/events/training", "Training")
    )
    
    # Test progress channel in a separate thread
    progress_thread = threading.Thread(
        target=test_sse_connection, 
        args=("http://localhost:8081/events/progress", "Progress")
    )
    
    print("🚀 Starting connection threads...")
    training_thread.start()
    progress_thread.start()
    
    # Give connections time to establish
    time.sleep(5)
    
    # Send some test data
    print("📤 Sending test data...")
    send_test_training_data()
    time.sleep(2)
    send_test_progress_data()
    
    # Wait for threads to complete
    print("⏳ Waiting for connections to complete...")
    training_thread.join(timeout=15)
    progress_thread.join(timeout=15)
    
    print("✅ SSE connection test completed")

def send_test_training_data():
    """Send test training data."""
    url = "http://localhost:8081/training_data"
    data = {
        "type": "training_update",
        "data": {
            "epoch": 1,
            "total_epochs": 5,
            "generator_loss": 0.75,
            "discriminator_loss": 0.65,
            "real_scores": 0.85,
            "fake_scores": 0.25
        },
        "timestamp": "2024-01-01T12:00:00"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"📤 Training data sent: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Error sending training data: {e}")

def send_test_progress_data():
    """Send test progress data."""
    url = "http://localhost:8081/progress_data"
    data = {
        "type": "progress",
        "epoch": 1,
        "progress_percent": 50.0,
        "timestamp": "2024-01-01T12:00:00"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"📤 Progress data sent: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Error sending progress data: {e}")

if __name__ == "__main__":
    test_connections() 