#!/usr/bin/env python3
"""
Robust SSE connection test to debug connection stability issues.
This script will help identify why SSE connections are not staying connected.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class SSEDebugTester:
    def __init__(self, dashboard_url="http://localhost:8081"):
        self.dashboard_url = dashboard_url
        self.session = None
        
    async def setup(self):
        """Set up the HTTP session."""
        self.session = aiohttp.ClientSession()
        
    async def cleanup(self):
        """Clean up the HTTP session."""
        if self.session:
            await self.session.close()
    
    async def test_sse_connection_stability(self):
        """Test SSE connection stability by monitoring connections over time."""
        print("🔍 Testing SSE Connection Stability")
        print(f"📡 Dashboard URL: {self.dashboard_url}")
        print("=" * 60)
        
        # Test 1: Check current client count
        print("\n🎯 Test 1: Checking current client count")
        await self.check_client_count()
        
        # Test 2: Test SSE endpoint directly
        print("\n🎯 Test 2: Testing SSE endpoint directly")
        await self.test_sse_endpoint_directly()
        
        # Test 3: Monitor client count changes
        print("\n🎯 Test 3: Monitoring client count changes")
        await self.monitor_client_count_changes()
        
        return True
    
    async def check_client_count(self):
        """Check the current number of connected clients."""
        try:
            # Send a test message to see current client count
            test_data = {
                'type': 'test_connection',
                'data': {'message': 'Testing connection count'},
                'timestamp': datetime.now().isoformat()
            }
            
            async with self.session.post(
                f"{self.dashboard_url}/training_data",
                json=test_data
            ) as response:
                result = await response.json()
                print(f"   📊 Current training clients: {result.get('clients', 'unknown')}")
                
            async with self.session.post(
                f"{self.dashboard_url}/progress_data",
                json=test_data
            ) as response:
                result = await response.json()
                print(f"   📊 Current progress clients: {result.get('clients', 'unknown')}")
                
        except Exception as e:
            print(f"   ❌ Error checking client count: {e}")
    
    async def test_sse_endpoint_directly(self):
        """Test the SSE endpoint directly to see if it's responding."""
        try:
            print("   🔌 Testing /events/training endpoint...")
            async with self.session.get(f"{self.dashboard_url}/events/training") as response:
                print(f"   📡 Training SSE endpoint status: {response.status}")
                print(f"   📡 Training SSE endpoint headers: {dict(response.headers)}")
                
                if response.status == 200:
                    print("   ✅ Training SSE endpoint is responding correctly")
                else:
                    print(f"   ❌ Training SSE endpoint error: {response.status}")
                    
        except Exception as e:
            print(f"   ❌ Error testing training SSE endpoint: {e}")
        
        try:
            print("   🔌 Testing /events/progress endpoint...")
            async with self.session.get(f"{self.dashboard_url}/events/progress") as response:
                print(f"   📡 Progress SSE endpoint status: {response.status}")
                print(f"   📡 Progress SSE endpoint headers: {dict(response.headers)}")
                
                if response.status == 200:
                    print("   ✅ Progress SSE endpoint is responding correctly")
                else:
                    print(f"   ❌ Progress SSE endpoint error: {response.status}")
                    
        except Exception as e:
            print(f"   ❌ Error testing progress SSE endpoint: {e}")
    
    async def monitor_client_count_changes(self):
        """Monitor client count changes over time."""
        print("   📊 Monitoring client count changes for 30 seconds...")
        
        initial_training = await self.get_client_count('training')
        initial_progress = await self.get_client_count('progress')
        
        print(f"   📊 Initial - Training: {initial_training}, Progress: {initial_progress}")
        
        # Monitor for 30 seconds
        for i in range(6):  # 6 iterations of 5 seconds each
            await asyncio.sleep(5)
            
            current_training = await self.get_client_count('training')
            current_progress = await self.get_client_count('progress')
            
            print(f"   📊 {i*5+5}s - Training: {current_training}, Progress: {current_progress}")
            
            if current_training != initial_training or current_progress != initial_progress:
                print(f"   🔄 Client count changed!")
    
    async def get_client_count(self, channel):
        """Get the current client count for a specific channel."""
        try:
            test_data = {
                'type': 'test_count',
                'data': {'message': 'Getting client count'},
                'timestamp': datetime.now().isoformat()
            }
            
            endpoint = f"{self.dashboard_url}/training_data" if channel == 'training' else f"{self.dashboard_url}/progress_data"
            
            async with self.session.post(endpoint, json=test_data) as response:
                result = await response.json()
                return result.get('clients', 0)
                
        except Exception as e:
            print(f"   ❌ Error getting {channel} client count: {e}")
            return 0

async def main():
    """Main test function."""
    tester = SSEDebugTester()
    
    try:
        await tester.setup()
        await tester.test_sse_connection_stability()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    print("🚀 SSE Connection Stability Debug Test")
    print("This will help identify why SSE connections are not staying stable")
    print("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user") 