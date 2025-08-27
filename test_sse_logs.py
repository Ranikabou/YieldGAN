#!/usr/bin/env python3
"""
Test script to verify SSE log connections are working.
"""

import asyncio
import aiohttp
import json
import time

async def test_log_sse():
    """Test the log SSE endpoint."""
    print("🔌 Testing Log SSE Connection...")
    
    async with aiohttp.ClientSession() as session:
        # Connect to the log SSE endpoint
        async with session.get('http://localhost:8082/events/logs') as response:
            print(f"📡 SSE Response Status: {response.status}")
            print(f"📡 SSE Response Headers: {response.headers}")
            
            if response.status == 200:
                print("✅ Successfully connected to log SSE endpoint")
                
                # Listen for events
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])  # Remove 'data: ' prefix
                            print(f"📝 Received log event: {json.dumps(data, indent=2)}")
                        except json.JSONDecodeError:
                            print(f"📝 Raw data line: {line}")
                    elif line:  # Non-empty line that's not data
                        print(f"📝 Other line: {line}")
            else:
                print(f"❌ Failed to connect to log SSE endpoint: {response.status}")
                print(await response.text())

async def test_log_data_endpoint():
    """Test sending log data to the endpoint."""
    print("\n📤 Testing Log Data Endpoint...")
    
    async with aiohttp.ClientSession() as session:
        # Send a test log entry
        test_log = {
            "type": "log_entry",
            "data": {
                "message": f"Test log message at {time.time()}",
                "source": "test_script",
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        async with session.post('http://localhost:8082/log_data', json=test_log) as response:
            print(f"📡 Log Data Response Status: {response.status}")
            if response.status == 200:
                result = await response.json()
                print(f"✅ Log data sent successfully: {result}")
            else:
                print(f"❌ Failed to send log data: {response.status}")
                print(await response.text())

async def main():
    """Main test function."""
    print("🚀 Starting SSE Log Connection Test...")
    
    try:
        # Test sending log data first
        await test_log_data_endpoint()
        
        # Wait a moment for the data to be processed
        await asyncio.sleep(1)
        
        # Test the SSE connection
        await test_log_sse()
        
    except aiohttp.ClientConnectorError:
        print("❌ Could not connect to dashboard. Make sure it's running on port 8082.")
    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 