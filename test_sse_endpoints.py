#!/usr/bin/env python3
"""
Test SSE Endpoints Accessibility
"""

import requests
import time

def test_sse_endpoints():
    """Test if SSE endpoints are accessible."""
    base_url = "http://localhost:8082"
    
    print("🧪 Testing SSE Endpoints Accessibility")
    print("=" * 50)
    
    # Test 1: Check if dashboard is accessible
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Dashboard accessible: {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard not accessible: {e}")
        return
    
    # Test 2: Check SSE endpoints directly
    endpoints = [
        "/events/training",
        "/events/progress", 
        "/events/logs"
    ]
    
    for endpoint in endpoints:
        try:
            print(f"\n🔌 Testing {endpoint}...")
            response = requests.get(f"{base_url}{endpoint}", stream=True, timeout=5)
            print(f"   Status: {response.status_code}")
            print(f"   Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                print(f"   ✅ {endpoint} is accessible")
                
                # Try to read a few lines to see if it's working
                try:
                    for i, line in enumerate(response.iter_lines()):
                        if i >= 3:  # Only read first 3 lines
                            break
                        if line:
                            print(f"   Data: {line.decode('utf-8')}")
                except Exception as e:
                    print(f"   ⚠️  Error reading stream: {e}")
                    
            else:
                print(f"   ❌ {endpoint} returned status {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error testing {endpoint}: {e}")
    
    # Test 3: Check if any clients are connected
    print(f"\n📊 Checking client connections...")
    
    # Send test data to see client count
    test_data = {
        "type": "training_update",
        "data": {
            "epoch": 1,
            "total_epochs": 1,
            "generator_loss": 0.5,
            "discriminator_loss": 0.6,
            "real_scores": 0.8,
            "fake_scores": 0.2
        },
        "timestamp": "2025-08-25T22:26:00"
    }
    
    try:
        response = requests.post(
            f"{base_url}/training_data",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training data sent: {result}")
            print(f"📊 Training clients: {result.get('clients', 0)}")
        else:
            print(f"❌ Training data failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error sending training data: {e}")
    
    # Test 4: Check progress endpoint
    try:
        response = requests.post(
            f"{base_url}/progress_data",
            json={
                "type": "progress",
                "epoch": 1,
                "progress_percent": 50,
                "timestamp": "2025-08-25T22:26:00"
            },
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Progress data sent: {result}")
            print(f"📊 Progress clients: {result.get('clients', 0)}")
        else:
            print(f"❌ Progress data failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error sending progress data: {e}")

if __name__ == "__main__":
    test_sse_endpoints() 