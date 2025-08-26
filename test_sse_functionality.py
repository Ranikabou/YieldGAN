#!/usr/bin/env python3
"""
Simple test script to verify SSE functionality in the dashboard.
This script sends test training data to verify that the UI updates correctly.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class SSETester:
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
    
    async def send_training_data(self, epoch, generator_loss, discriminator_loss, real_scores, fake_scores):
        """Send training data to the dashboard."""
        training_data = {
            'type': 'training_update',
            'data': {
                'epoch': epoch,
                'total_epochs': 100,
                'generator_loss': generator_loss,
                'discriminator_loss': discriminator_loss,
                'real_scores': real_scores,
                'fake_scores': fake_scores
            },
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            async with self.session.post(
                f"{self.dashboard_url}/training_data",
                json=training_data
            ) as response:
                result = await response.json()
                print(f"✅ Training data sent successfully: {result}")
                return True
        except Exception as e:
            print(f"❌ Error sending training data: {e}")
            return False
    
    async def send_progress_data(self, epoch, progress_percent):
        """Send progress data to the dashboard."""
        progress_data = {
            'type': 'progress',
            'epoch': epoch,
            'progress_percent': progress_percent,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            async with self.session.post(
                f"{self.dashboard_url}/progress_data",
                json=progress_data
            ) as response:
                result = await response.json()
                print(f"✅ Progress data sent successfully: {result}")
                return True
        except Exception as e:
            print(f"❌ Error sending progress data: {e}")
            return False
    
    async def test_sse_functionality(self):
        """Test the SSE functionality by sending multiple data points."""
        print("🧪 Testing SSE functionality...")
        print(f"📡 Dashboard URL: {self.dashboard_url}")
        print("=" * 50)
        
        # Test 1: Send initial training data
        print("\n🎯 Test 1: Sending initial training data")
        success = await self.send_training_data(
            epoch=1,
            generator_loss=0.8,
            discriminator_loss=0.6,
            real_scores=0.7,
            fake_scores=0.3
        )
        
        if not success:
            print("❌ Failed to send initial training data")
            return False
        
        # Test 2: Send progress data
        print("\n📊 Test 2: Sending progress data")
        success = await self.send_progress_data(epoch=1, progress_percent=10)
        
        if not success:
            print("❌ Failed to send progress data")
            return False
        
        # Test 3: Send multiple training updates
        print("\n🔄 Test 3: Sending multiple training updates")
        for epoch in range(2, 6):
            await self.send_training_data(
                epoch=epoch,
                generator_loss=0.8 - (epoch * 0.1),
                discriminator_loss=0.6 + (epoch * 0.05),
                real_scores=0.7 + (epoch * 0.02),
                fake_scores=0.3 - (epoch * 0.02)
            )
            await asyncio.sleep(0.5)  # Small delay between updates
        
        # Test 4: Send final progress
        print("\n🏁 Test 4: Sending final progress")
        await self.send_progress_data(epoch=5, progress_percent=50)
        
        print("\n✅ SSE functionality test completed!")
        print("📱 Check your dashboard UI to see if the data is being displayed correctly")
        print("🎯 You should see:")
        print("   - Epoch updates")
        print("   - Generator Loss changes")
        print("   - Discriminator Loss changes")
        print("   - Real vs Synthetic Scores")
        print("   - Training Progress percentage")
        
        return True

async def main():
    """Main test function."""
    tester = SSETester()
    
    try:
        await tester.setup()
        await tester.test_sse_functionality()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    print("🚀 SSE Functionality Test")
    print("Make sure your dashboard is running on http://localhost:8081")
    print("Press Ctrl+C to stop the test")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user") 