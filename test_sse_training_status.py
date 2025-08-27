#!/usr/bin/env python3
"""
Test that training status is only available via SSE (no REST API).
"""

import asyncio
import aiohttp
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_sse_training_status():
    """Test that training status is only available via SSE."""
    
    # Test that REST API endpoint is removed
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://localhost:8082/api/training_status") as response:
                if response.status == 404:
                    logger.info("✅ REST API endpoint properly removed")
                else:
                    logger.error(f"❌ REST API endpoint still exists: {response.status}")
                    return False
        except Exception as e:
            logger.info(f"✅ REST API endpoint not accessible: {e}")
    
    # Test SSE training channel provides status
    try:
        logger.info("🎯 Testing SSE training status channel...")
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8082/events/training") as response:
                if response.status == 200:
                    logger.info("✅ SSE training channel accessible")
                    
                    # Read a few SSE messages
                    line_count = 0
                    async for line in response.content:
                        if line_count >= 3:  # Read first few messages
                            break
                        
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith('data: '):
                            try:
                                data = json.loads(decoded_line[6:])  # Remove 'data: ' prefix
                                logger.info(f"📊 SSE Message: {data.get('type', 'unknown')} - {data}")
                                
                                if data.get('type') == 'status_update':
                                    logger.info(f"✅ Training status via SSE: {data.get('data', {}).get('status', 'unknown')}")
                                
                                line_count += 1
                            except json.JSONDecodeError as e:
                                logger.warning(f"Could not parse SSE data: {decoded_line}")
                    
                    return True
                else:
                    logger.error(f"❌ SSE training channel not accessible: {response.status}")
                    return False
                    
    except Exception as e:
        logger.error(f"❌ SSE test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing SSE-only training status implementation...")
    result = asyncio.run(test_sse_training_status())
    
    if result:
        print("✅ All tests passed - Training status is only available via SSE!")
    else:
        print("❌ Tests failed - Check implementation") 