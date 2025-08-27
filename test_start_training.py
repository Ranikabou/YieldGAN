#!/usr/bin/env python3
"""
Test Start Training Button Functionality
"""

import requests
import json

def test_start_training():
    """Test the start training API endpoint."""
    base_url = "http://localhost:8082"
    
    print("🧪 Testing Start Training Button Functionality")
    print("=" * 50)
    
    # Test 1: Check if dashboard is accessible
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Dashboard accessible: {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard not accessible: {e}")
        return
    
    # Test 2: Check training status
    try:
        response = requests.get(f"{base_url}/api/training_status")
        status_data = response.json()
        print(f"✅ Training status: {json.dumps(status_data, indent=2)}")
    except Exception as e:
        print(f"❌ Error getting training status: {e}")
    
    # Test 3: Test start training with valid data
    try:
        training_data = {
            "config": "config/gan_config.yaml",
            "data_source": "treasury_orderbook_sample.csv"
        }
        
        print(f"\n🚀 Testing start training with data: {json.dumps(training_data, indent=2)}")
        
        response = requests.post(
            f"{base_url}/api/start_training",
            json=training_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Start training successful: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                print(f"🎯 Training started with PID: {result.get('pid')}")
                print(f"📊 Config: {result.get('config')}")
                print(f"📁 Data source: {result.get('data_source')}")
            else:
                print(f"❌ Training failed to start: {result.get('error')}")
        else:
            print(f"❌ Start training failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing start training: {e}")
    
    # Test 4: Check if training process is running
    try:
        response = requests.get(f"{base_url}/api/training_status")
        status_data = response.json()
        print(f"\n📊 Training status after start attempt: {json.dumps(status_data, indent=2)}")
    except Exception as e:
        print(f"❌ Error getting updated training status: {e}")

if __name__ == "__main__":
    test_start_training() 