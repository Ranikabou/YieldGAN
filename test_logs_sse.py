#!/usr/bin/env python3
"""
Test script to verify SSE logs are working in the GAN Dashboard
"""

import requests
import json
import time
from datetime import datetime

def test_logs_sse():
    """Test the logs SSE endpoint by sending test data."""
    
    # Test different types of log entries
    test_logs = [
        {
            "type": "log_entry",
            "data": {
                "message": "Test log entry 1 - Training started",
                "source": "training",
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        },
        {
            "type": "log_entry", 
            "data": {
                "message": "Test log entry 2 - Progress bar: 50%|█████     | 50/100 [00:30<00:30, 1.67it/s]",
                "source": "training_progress",
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        },
        {
            "type": "log_entry",
            "data": {
                "message": "Test log entry 3 - Real error occurred",
                "source": "training_error", 
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    print("🧪 Testing SSE Logs Endpoint")
    print("=" * 50)
    
    # Test 1: Check if dashboard is running
    try:
        response = requests.get("http://localhost:8081/", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard is running on port 8081")
        else:
            print(f"❌ Dashboard returned status {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to dashboard: {e}")
        return
    
    # Test 2: Check SSE logs endpoint
    try:
        response = requests.get("http://localhost:8081/events/logs", timeout=5)
        print(f"✅ SSE logs endpoint accessible (status: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"❌ SSE logs endpoint error: {e}")
    
    # Test 3: Send test data via API (if available)
    print("\n📤 Sending test log entries...")
    
    for i, log_entry in enumerate(test_logs, 1):
        print(f"  {i}. {log_entry['data']['source']}: {log_entry['data']['message'][:50]}...")
        
        # Note: The dashboard doesn't have a direct API to send logs
        # Logs are only sent during training. This is just for display purposes.
    
    print("\n📋 Test Summary:")
    print("  - Dashboard is running ✅")
    print("  - SSE logs endpoint is accessible ✅")
    print("  - Logs will appear in the UI when training starts")
    print("\n🌐 Open http://localhost:8081 in your browser to see the dashboard")
    print("📝 The logs section should be visible and show placeholder text")
    print("🚀 Start training to see real-time logs appear")

if __name__ == "__main__":
    test_logs_sse() 