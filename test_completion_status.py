#!/usr/bin/env python3
"""
Test script to verify that training completion status is properly displayed.
This script tests both API status and SSE event delivery.
"""

import asyncio
import aiohttp
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_training_completion_status():
    """Test that training completion status is properly communicated."""
    
    dashboard_url = "http://localhost:8085"
    
    # Test 1: Check API status
    logger.info("🧪 Test 1: Checking training status via API...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{dashboard_url}/api/training_status") as response:
                if response.status == 200:
                    status_data = await response.json()
                    logger.info(f"✅ API Status: {status_data}")
                    
                    if status_data.get('status') == 'completed':
                        logger.info("✅ API correctly reports training as completed")
                    else:
                        logger.warning(f"⚠️ API reports status as: {status_data.get('status')}")
                else:
                    logger.error(f"❌ API request failed with status: {response.status}")
    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
    
    # Test 2: Test SSE connection and check for completion message
    logger.info("🧪 Test 2: Testing SSE training events channel...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{dashboard_url}/events/training") as response:
                if response.status == 200:
                    logger.info("✅ SSE connection established")
                    
                    # Read a few SSE messages
                    message_count = 0
                    completion_received = False
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])  # Remove 'data: ' prefix
                                logger.info(f"📨 SSE Message: {data.get('type', 'unknown')}")
                                
                                if data.get('type') == 'training_complete':
                                    logger.info(f"✅ Received training_complete event: {data}")
                                    completion_received = True
                                    break
                                elif data.get('type') == 'status_update':
                                    status = data.get('data', {}).get('status', 'unknown')
                                    logger.info(f"📊 Status update: {status}")
                                    if status == 'completed':
                                        logger.info("✅ Status update shows training as completed")
                                        completion_received = True
                                        break
                                        
                                message_count += 1
                                if message_count >= 5:  # Limit messages to avoid hanging
                                    break
                                    
                            except json.JSONDecodeError as e:
                                logger.debug(f"Non-JSON line: {line}")
                    
                    if completion_received:
                        logger.info("✅ SSE successfully delivered completion status")
                    else:
                        logger.warning("⚠️ No completion message received via SSE")
                        
                else:
                    logger.error(f"❌ SSE connection failed with status: {response.status}")
    except Exception as e:
        logger.error(f"❌ SSE test failed: {e}")
    
    # Test 3: Simulate a new client connecting (to test pending message delivery)
    logger.info("🧪 Test 3: Testing new client connection (pending message delivery)...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{dashboard_url}/events/training") as response:
                if response.status == 200:
                    logger.info("✅ New SSE connection established")
                    
                    # Check first few messages for any pending completion
                    message_count = 0
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                message_type = data.get('type', 'unknown')
                                logger.info(f"📨 New client message: {message_type}")
                                
                                if message_type == 'training_complete':
                                    logger.info("✅ Pending completion message delivered to new client")
                                    break
                                elif message_type == 'status_update':
                                    status = data.get('data', {}).get('status', 'unknown')
                                    if status == 'completed':
                                        logger.info("✅ New client received completed status")
                                        break
                                        
                                message_count += 1
                                if message_count >= 3:  # Just check first few messages
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                                
    except Exception as e:
        logger.error(f"❌ New client test failed: {e}")
    
    logger.info("🏁 Test completed")

if __name__ == "__main__":
    asyncio.run(test_training_completion_status()) 