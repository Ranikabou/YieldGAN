#!/usr/bin/env python3
"""
SSE UI Fix Test Script
Tests if the dashboard UI properly receives and displays SSE updates.
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SSEUITester:
    def __init__(self, dashboard_url=None):
        if dashboard_url is None:
            try:
                from utils.port_manager import get_dashboard_url
                self.dashboard_url = get_dashboard_url()
            except ImportError:
                self.dashboard_url = "http://localhost:8083"
        else:
            self.dashboard_url = dashboard_url
        
    async def test_sse_communication(self):
        """Test SSE communication by sending training and progress data."""
        logger.info("🚀 Starting SSE UI communication test...")
        
        # Test 1: Send training data
        training_data = {
            "type": "training_update",
            "data": {
                "epoch": 1,
                "total_epochs": 5,
                "generator_loss": 0.7500,
                "discriminator_loss": 1.2000,
                "real_scores": 0.8500,
                "fake_scores": 0.2500
            },
            "timestamp": datetime.now().isoformat()
        }
        
        success = await self.send_training_data(training_data)
        if success:
            logger.info("✅ Training data sent successfully")
        else:
            logger.error("❌ Failed to send training data")
            
        # Wait a moment
        await asyncio.sleep(1)
        
        # Test 2: Send progress data
        progress_data = {
            "type": "progress",
            "epoch": 1,
            "progress_percent": 75,
            "timestamp": datetime.now().isoformat()
        }
        
        success = await self.send_progress_data(progress_data)
        if success:
            logger.info("✅ Progress data sent successfully")
        else:
            logger.error("❌ Failed to send progress data")
            
        # Test 3: Run a simulation of multiple epochs
        logger.info("🎯 Running training simulation...")
        await self.simulate_training_epochs()
        
    async def send_training_data(self, data):
        """Send training data to dashboard."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.dashboard_url}/training_data",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"📊 Training data response: {result}")
                        return True
                    else:
                        logger.error(f"❌ Training data failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Training data error: {e}")
            return False
            
    async def send_progress_data(self, data):
        """Send progress data to dashboard."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.dashboard_url}/progress_data",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"📈 Progress data response: {result}")
                        return True
                    else:
                        logger.error(f"❌ Progress data failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Progress data error: {e}")
            return False
            
    async def simulate_training_epochs(self):
        """Simulate a training session with multiple epochs."""
        total_epochs = 3
        
        for epoch in range(1, total_epochs + 1):
            logger.info(f"🔄 Simulating epoch {epoch}/{total_epochs}")
            
            # Send progress updates throughout the epoch
            for progress in [25, 50, 75, 100]:
                progress_data = {
                    "type": "progress",
                    "epoch": epoch,
                    "progress_percent": progress,
                    "timestamp": datetime.now().isoformat()
                }
                await self.send_progress_data(progress_data)
                await asyncio.sleep(0.5)  # Brief pause between progress updates
                
            # Send training metrics at end of epoch
            gen_loss = 1.0 - (epoch * 0.15)  # Decreasing loss
            disc_loss = 0.8 + (epoch * 0.05)  # Slightly increasing loss
            
            training_data = {
                "type": "training_update", 
                "data": {
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "generator_loss": round(gen_loss, 4),
                    "discriminator_loss": round(disc_loss, 4),
                    "real_scores": round(0.85 - epoch * 0.02, 4),
                    "fake_scores": round(0.15 + epoch * 0.03, 4)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await self.send_training_data(training_data)
            await asyncio.sleep(1)  # Pause between epochs
            
        logger.info("✅ Training simulation completed")
        
    async def test_dashboard_connectivity(self):
        """Test if dashboard is reachable."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.dashboard_url}/",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        logger.info(f"✅ Dashboard reachable at {self.dashboard_url}")
                        return True
                    else:
                        logger.error(f"❌ Dashboard returned status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Dashboard connectivity test failed: {e}")
            return False

async def main():
    """Main test function."""
    logger.info("🧪 SSE UI Fix Test Starting...")
    
    # Try to use port manager first
    try:
        from utils.port_manager import get_port_manager
        pm = get_port_manager()
        dashboard_url = pm.get_dashboard_url()
        
        # Test if this URL is accessible
        tester = SSEUITester(dashboard_url)
        if await tester.test_dashboard_connectivity():
            logger.info(f"🎯 Using dashboard from port manager: {dashboard_url}")
        else:
            dashboard_url = None
    except ImportError:
        dashboard_url = None
    
    # Fallback to manual detection if port manager fails
    if not dashboard_url:
        possible_ports = [8083, 8081, 8082, 8084]
        
        for port in possible_ports:
            test_url = f"http://localhost:{port}"
            tester = SSEUITester(test_url)
            
            logger.info(f"🔍 Testing dashboard at {test_url}...")
            if await tester.test_dashboard_connectivity():
                dashboard_url = test_url
                break
                
        if not dashboard_url:
            logger.error("❌ No accessible dashboard found on common ports")
            logger.info("💡 Make sure the dashboard is running: python gan_dashboard.py")
            return
            
        logger.info(f"🎯 Found dashboard at {dashboard_url}")
        
        # Create tester with found URL
        tester = SSEUITester(dashboard_url)
    else:
        # Already created tester above
        pass
    
    # Run SSE communication tests
    await tester.test_sse_communication()
    
    logger.info("🏁 SSE UI Fix Test Completed")
    logger.info("📋 Check your dashboard UI to see if the data appeared!")
    logger.info(f"📱 Dashboard URL: {dashboard_url}")

if __name__ == "__main__":
    asyncio.run(main()) 