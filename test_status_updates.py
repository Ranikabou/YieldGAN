#!/usr/bin/env python3
"""
Test script to verify training status updates via SSE
"""

import asyncio
import aiohttp
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_status_updates():
    """Test the SSE status updates."""
    dashboard_url = "http://localhost:8081"
    
    async with aiohttp.ClientSession() as session:
        logger.info("🔍 Testing SSE status updates...")
        
        # Test training channel
        logger.info("🎯 Testing training channel...")
        try:
            async with session.get(f"{dashboard_url}/events/training") as response:
                if response.status == 200:
                    logger.info("✅ Training channel accessible")
                    
                    # Listen for messages
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])  # Remove 'data: ' prefix
                                logger.info(f"📡 Received: {data}")
                                
                                if data.get('type') == 'status_update':
                                    logger.info(f"🎯 Status update: {data['data']['status']}")
                                elif data.get('type') == 'training_complete':
                                    logger.info(f"✅ Training complete: {data['data']['status']}")
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse JSON: {e}")
                        elif line.startswith(': keep-alive'):
                            logger.debug("💓 Keep-alive received")
                else:
                    logger.error(f"❌ Training channel failed: {response.status}")
        except Exception as e:
            logger.error(f"❌ Error testing training channel: {e}")
        
        # Test progress channel
        logger.info("📊 Testing progress channel...")
        try:
            async with session.get(f"{dashboard_url}/events/progress") as response:
                if response.status == 200:
                    logger.info("✅ Progress channel accessible")
                else:
                    logger.error(f"❌ Progress channel failed: {response.status}")
        except Exception as e:
            logger.error(f"❌ Error testing progress channel: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(test_status_updates())
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}") 