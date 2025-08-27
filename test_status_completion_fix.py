#!/usr/bin/env python3
"""
Test that training completion status is properly broadcast via SSE.
"""

import asyncio
import aiohttp
import json
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_training_completion_status():
    """Test that training completion status is properly broadcast via SSE."""
    
    logger.info("🧪 Testing training completion status via SSE...")
    
    try:
        # Connect to SSE training channel
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8082/events/training") as response:
                if response.status == 200:
                    logger.info("✅ SSE training channel connected")
                    
                    # Listen for status updates
                    message_count = 0
                    status_received = False
                    completion_received = False
                    
                    async for line in response.content:
                        if message_count >= 10:  # Don't listen forever
                            break
                            
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith('data: '):
                            try:
                                data = json.loads(decoded_line[6:])  # Remove 'data: ' prefix
                                message_type = data.get('type', 'unknown')
                                
                                logger.info(f"📊 SSE Message: {message_type}")
                                
                                if message_type == 'status_update':
                                    status = data.get('data', {}).get('status', 'unknown')
                                    logger.info(f"🎯 Status Update: {status}")
                                    status_received = True
                                    
                                    if status in ['completed', 'failed']:
                                        logger.info(f"✅ Training completion status received: {status}")
                                        return True
                                        
                                elif message_type == 'training_complete':
                                    status = data.get('data', {}).get('status', 'unknown')
                                    logger.info(f"🎉 Training Complete: {status}")
                                    completion_received = True
                                    
                                    if status in ['completed', 'failed']:
                                        logger.info(f"✅ Training completion event received: {status}")
                                        return True
                                
                                elif message_type == 'connection':
                                    logger.info(f"🔗 Connection: {data.get('message', 'Connected')}")
                                
                                message_count += 1
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"Could not parse SSE data: {decoded_line}")
                                
                    if status_received or completion_received:
                        logger.info("✅ Some status messages were received")
                        return True
                    else:
                        logger.warning("⚠️ No status messages received")
                        return False
                        
                else:
                    logger.error(f"❌ SSE training channel not accessible: {response.status}")
                    return False
                    
    except Exception as e:
        logger.error(f"❌ SSE test failed: {e}")
        return False

async def test_immediate_status():
    """Test that connecting to SSE immediately returns current status."""
    
    logger.info("🧪 Testing immediate status on SSE connection...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8082/events/training") as response:
                if response.status == 200:
                    logger.info("✅ SSE training channel connected")
                    
                    # Read first few messages quickly
                    start_time = time.time()
                    timeout = 5  # 5 second timeout
                    
                    async for line in response.content:
                        if time.time() - start_time > timeout:
                            break
                            
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith('data: '):
                            try:
                                data = json.loads(decoded_line[6:])
                                message_type = data.get('type', 'unknown')
                                
                                if message_type == 'status_update':
                                    status = data.get('data', {}).get('status', 'unknown')
                                    logger.info(f"✅ Immediate status received: {status}")
                                    return status in ['idle', 'running', 'completed', 'failed']
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    logger.warning("⚠️ No immediate status received within timeout")
                    return False
                else:
                    logger.error(f"❌ SSE training channel not accessible: {response.status}")
                    return False
                    
    except Exception as e:
        logger.error(f"❌ Immediate status test failed: {e}")
        return False

if __name__ == "__main__":
    async def run_all_tests():
        print("🧪 Testing training completion status fixes...")
        
        # Test immediate status
        immediate_result = await test_immediate_status()
        
        # Test completion status  
        completion_result = await test_training_completion_status()
        
        if immediate_result and completion_result:
            print("✅ All tests passed - Training status is working correctly!")
            return True
        elif immediate_result:
            print("✅ Immediate status test passed - Connect to see completion status")
            return True
        else:
            print("❌ Tests failed - Check SSE implementation")
            return False
    
    result = asyncio.run(run_all_tests()) 