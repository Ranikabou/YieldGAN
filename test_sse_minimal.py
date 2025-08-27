#!/usr/bin/env python3
"""
Minimal test for SSE endpoint to identify the hanging issue.
"""

import requests
import time

def test_sse_endpoint():
    """Test SSE endpoint with a timeout."""
    print("🔍 Testing SSE endpoint with timeout...")
    
    try:
        # Test with a short timeout to see if it responds
        response = requests.get(
            "http://localhost:8082/events/training", 
            timeout=2,
            stream=True
        )
        
        print(f"✅ Response status: {response.status_code}")
        print(f"✅ Response headers: {dict(response.headers)}")
        
        # Try to read a few lines
        for i, line in enumerate(response.iter_lines()):
            if i >= 3:  # Only read first 3 lines
                break
            if line:
                print(f"📡 Line {i}: {line.decode()}")
        
        response.close()
        
    except requests.exceptions.Timeout:
        print("⏰ SSE endpoint timed out - this indicates the hanging issue")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_simple_endpoint():
    """Test a simple endpoint to ensure dashboard is working."""
    try:
        response = requests.get("http://localhost:8082/api/training_status", timeout=5)
        print(f"✅ Simple endpoint works: {response.status_code}")
        print(f"✅ Response: {response.json()}")
    except Exception as e:
        print(f"❌ Simple endpoint failed: {e}")

if __name__ == "__main__":
    print("🔍 Minimal SSE Test")
    print("=" * 30)
    
    test_simple_endpoint()
    test_sse_endpoint()
    
    print("\n📋 Analysis:")
    print("If SSE times out, the endpoint is hanging in an infinite loop")
    print("This prevents the UI from receiving real-time updates") 