#!/usr/bin/env python3
"""
Test SSE Data Flow - Simulates training data to test systematic updates
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class SSEDataFlowTester:
    def __init__(self, base_url="http://localhost:8082"):
        self.base_url = base_url
        
    async def send_training_data(self, data):
        """Send training data to the dashboard."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/training_data",
                    json=data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ Training data sent: {result}")
                        return True
                    else:
                        print(f"❌ Training data failed: {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Training data error: {e}")
            return False
    
    async def send_progress_data(self, data):
        """Send progress data to the dashboard."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/progress_data",
                    json=data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ Progress data sent: {result}")
                        return True
                    else:
                        print(f"❌ Progress data failed: {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Progress data error: {e}")
            return False
    
    async def simulate_training_session(self):
        """Simulate a complete training session with systematic updates."""
        print("🚀 Simulating training session with systematic updates...")
        print("=" * 60)
        
        # Simulate 3 epochs with progress updates
        for epoch in range(1, 4):
            print(f"\n📊 EPOCH {epoch}/3")
            print("-" * 30)
            
            # Send progress updates (0% to 100%)
            for progress in range(0, 101, 25):
                progress_data = {
                    "type": "progress",
                    "epoch": epoch,
                    "progress_percent": progress,
                    "timestamp": datetime.now().isoformat()
                }
                
                success = await self.send_progress_data(progress_data)
                if success:
                    print(f"   📈 Progress: {progress}% - Epoch {epoch}")
                
                await asyncio.sleep(0.5)  # Small delay between progress updates
            
            # Send training metrics after progress completes
            training_data = {
                "type": "training_update",
                "data": {
                    "epoch": epoch,
                    "total_epochs": 3,
                    "generator_loss": 0.8 - epoch * 0.1 + (epoch % 2) * 0.05,
                    "discriminator_loss": 0.7 - epoch * 0.08 + (epoch % 3) * 0.03,
                    "real_scores": 0.9 - epoch * 0.02,
                    "fake_scores": 0.1 + epoch * 0.03
                },
                "timestamp": datetime.now().isoformat()
            }
            
            success = await self.send_training_data(training_data)
            if success:
                print(f"   🎯 Training metrics sent for epoch {epoch}")
                print(f"      Gen Loss: {training_data['data']['generator_loss']:.4f}")
                print(f"      Disc Loss: {training_data['data']['discriminator_loss']:.4f}")
            
            await asyncio.sleep(1)  # Delay between epochs
        
        print("\n" + "=" * 60)
        print("✅ Training session simulation completed!")
        print("=" * 60)
    
    async def test_rapid_updates(self):
        """Test rapid data updates to check systematic behavior."""
        print("\n⚡ Testing rapid updates for systematic behavior...")
        print("=" * 60)
        
        # Send rapid training updates
        for i in range(10):
            training_data = {
                "type": "training_update",
                "data": {
                    "epoch": 1,
                    "total_epochs": 10,
                    "generator_loss": 0.5 + (i * 0.02),
                    "discriminator_loss": 0.6 - (i * 0.01),
                    "real_scores": 0.8 + (i * 0.01),
                    "fake_scores": 0.2 - (i * 0.01)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            success = await self.send_training_data(training_data)
            if success:
                print(f"   🚀 Rapid update {i+1}/10 sent")
            
            await asyncio.sleep(0.2)  # Very rapid updates
        
        print("✅ Rapid updates test completed!")
    
    async def test_connection_stability(self):
        """Test connection stability during data flow."""
        print("\n🔌 Testing connection stability...")
        print("=" * 60)
        
        # Send data over a longer period to test stability
        start_time = time.time()
        message_count = 0
        
        while time.time() - start_time < 10:  # Test for 10 seconds
            # Send training data
            training_data = {
                "type": "training_update",
                "data": {
                    "epoch": 1,
                    "total_epochs": 1,
                    "generator_loss": 0.5 + (message_count * 0.01),
                    "discriminator_loss": 0.6 - (message_count * 0.01),
                    "real_scores": 0.8,
                    "fake_scores": 0.2
                },
                "timestamp": datetime.now().isoformat()
            }
            
            success = await self.send_training_data(training_data)
            if success:
                message_count += 1
                print(f"   📡 Stable connection test: {message_count} messages sent")
            
            await asyncio.sleep(0.5)
        
        print(f"✅ Connection stability test completed! {message_count} messages sent")

async def main():
    """Main test function."""
    tester = SSEDataFlowTester()
    
    print("🧪 SSE Data Flow Test Suite")
    print("Testing systematic updates on SSE endpoints")
    print()
    
    try:
        # Test 1: Simulate training session
        await tester.simulate_training_session()
        
        # Test 2: Test rapid updates
        await tester.test_rapid_updates()
        
        # Test 3: Test connection stability
        await tester.test_connection_stability()
        
        print("\n🎯 All tests completed successfully!")
        print("Check the dashboard UI to see if updates are systematic")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    
    print("\n🏁 Data flow test completed")

if __name__ == "__main__":
    asyncio.run(main()) 