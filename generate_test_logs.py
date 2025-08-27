#!/usr/bin/env python3
"""
Generate test log data to verify SSE connections.
"""

import requests
import time
import json
from datetime import datetime

def send_test_log(message, source="test_script"):
    """Send a test log message to the dashboard."""
    try:
        log_data = {
            "type": "log_entry",
            "data": {
                "message": message,
                "source": source,
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        response = requests.post('http://localhost:8082/log_data', json=log_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Log sent: {message}")
            print(f"   Response: {result}")
        else:
            print(f"❌ Failed to send log: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to dashboard. Make sure it's running on port 8082.")
    except Exception as e:
        print(f"❌ Error sending log: {e}")

def main():
    """Main function to generate test logs."""
    print("🚀 Starting test log generation...")
    print("Make sure your dashboard is running and you have SSE connections open!")
    
    # Wait a moment for connections to establish
    print("⏳ Waiting 3 seconds for connections to establish...")
    time.sleep(3)
    
    # Send various types of test logs
    test_logs = [
        "System startup completed successfully",
        "Training session initialized",
        "Data preprocessing started",
        "Epoch 1/10 completed - Generator Loss: 2.45, Discriminator Loss: 1.23",
        "Progress: 25% - Processing batch 250/1000",
        "Model checkpoint saved at epoch 1",
        "Data validation completed",
        "Training metrics updated",
        "Progress: 50% - Processing batch 500/1000",
        "Epoch 2/10 completed - Generator Loss: 2.12, Discriminator Loss: 1.15",
        "Memory usage: 2.3GB / 8GB",
        "Progress: 75% - Processing batch 750/1000",
        "Performance optimization applied",
        "Epoch 3/10 completed - Generator Loss: 1.89, Discriminator Loss: 1.08",
        "Progress: 100% - Processing batch 1000/1000",
        "Training session completed successfully"
    ]
    
    for i, log_message in enumerate(test_logs):
        print(f"\n📝 Sending log {i+1}/{len(test_logs)}...")
        send_test_log(log_message)
        time.sleep(2)  # Wait 2 seconds between logs
    
    print("\n✅ Test log generation completed!")
    print("Check your SSE connections to see if the logs are being received.")

if __name__ == "__main__":
    main() 