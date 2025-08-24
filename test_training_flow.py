#!/usr/bin/env python3
"""
Comprehensive test script to verify the complete training flow:
1. Start Training button functionality
2. SSE pipeline data flow
3. Dashboard UI updates
4. Real-time metrics display
"""

import asyncio
import aiohttp
import json
import time
import requests
from datetime import datetime

class TrainingFlowTester:
    def __init__(self, base_url="http://localhost:8081"):
        self.base_url = base_url
        self.session = None
        
    async def setup(self):
        """Setup aiohttp session."""
        self.session = aiohttp.ClientSession()
        
    async def cleanup(self):
        """Cleanup aiohttp session."""
        if self.session:
            await self.session.close()
            
    async def test_start_training(self):
        """Test the Start Training API endpoint."""
        print("🧪 Testing Start Training API...")
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/start_training",
                json={
                    "config": "config/gan_config.yaml",
                    "data_source": "csv"
                }
            ) as response:
                result = await response.json()
                print(f"✅ Start Training Response: {result}")
                return result.get('success', False)
        except Exception as e:
            print(f"❌ Start Training Error: {e}")
            return False
            
    async def test_training_events_sse(self):
        """Test the Training Events SSE endpoint."""
        print("🧪 Testing Training Events SSE...")
        
        try:
            async with self.session.get(f"{self.base_url}/events/training") as response:
                if response.status == 200:
                    print("✅ Training Events SSE connected successfully")
                    
                    # Read a few events to verify data flow
                    event_count = 0
                    async for line in response.content:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            try:
                                data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                                if data.get('type') == 'training_update':
                                    print(f"📊 Training Update: Epoch {data['data']['epoch']}, "
                                          f"Gen Loss: {data['data']['generator_loss']:.4f}, "
                                          f"Disc Loss: {data['data']['discriminator_loss']:.4f}")
                                    event_count += 1
                                    if event_count >= 5:  # Read 5 events
                                        break
                            except json.JSONDecodeError:
                                continue
                    
                    print(f"✅ Received {event_count} training events")
                    return True
                else:
                    print(f"❌ Training Events SSE failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Training Events SSE Error: {e}")
            return False
            
    async def test_progress_events_sse(self):
        """Test the Progress Events SSE endpoint."""
        print("🧪 Testing Progress Events SSE...")
        
        try:
            async with self.session.get(f"{self.base_url}/events/progress") as response:
                if response.status == 200:
                    print("✅ Progress Events SSE connected successfully")
                    
                    # Read a few events to verify data flow
                    event_count = 0
                    async for line in response.content:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            try:
                                data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                                if data.get('type') == 'progress':
                                    print(f"📈 Progress Update: Epoch {data['epoch']}, "
                                          f"Progress: {data['progress_percent']}%")
                                    event_count += 1
                                    if event_count >= 3:  # Read 3 events
                                        break
                            except json.JSONDecodeError:
                                continue
                    
                    print(f"✅ Received {event_count} progress events")
                    return True
                else:
                    print(f"❌ Progress Events SSE failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Progress Events SSE Error: {e}")
            return False
            
    async def test_training_status(self):
        """Test the Training Status API endpoint."""
        print("🧪 Testing Training Status API...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/training_status") as response:
                result = await response.json()
                print(f"✅ Training Status: {result}")
                return True
        except Exception as e:
            print(f"❌ Training Status Error: {e}")
            return False
            
    async def test_stop_training(self):
        """Test the Stop Training API endpoint."""
        print("🧪 Testing Stop Training API...")
        
        try:
            async with self.session.post(f"{self.base_url}/api/stop_training") as response:
                result = await response.json()
                print(f"✅ Stop Training Response: {result}")
                return result.get('success', False)
        except Exception as e:
            print(f"❌ Stop Training Error: {e}")
            return False
            
    async def test_csv_preview(self):
        """Test CSV preview functionality."""
        print("🧪 Testing CSV Preview...")
        
        try:
            async with self.session.get(
                f"{self.base_url}/api/preview_csv?filename=treasury_orderbook_sample.csv"
            ) as response:
                result = await response.json()
                if result.get('success'):
                    data_info = result['data_info']
                    print(f"✅ CSV Preview: {data_info['shape'][0]} rows × {data_info['shape'][1]} columns")
                    print(f"   Data Type: {data_info['data_type']}")
                    print(f"   Numeric Columns: {len(data_info['numeric_columns'])}")
                    return True
                else:
                    print(f"❌ CSV Preview failed: {result.get('error')}")
                    return False
        except Exception as e:
            print(f"❌ CSV Preview Error: {e}")
            return False

async def main():
    """Main test function."""
    print("🚀 Starting Comprehensive Training Flow Test")
    print("=" * 60)
    
    tester = TrainingFlowTester()
    await tester.setup()
    
    try:
        # Test CSV preview first
        csv_success = await tester.test_csv_preview()
        
        # Test Start Training
        start_success = await tester.test_start_training()
        
        if start_success:
            # Wait a moment for training to start
            print("⏳ Waiting for training to start...")
            await asyncio.sleep(3)
            
            # Test Training Status
            status_success = await tester.test_training_status()
            
            # Test SSE endpoints
            training_sse_success = await tester.test_training_events_sse()
            progress_sse_success = await tester.test_progress_events_sse()
            
            # Test Stop Training
            stop_success = await tester.test_stop_training()
            
            # Summary
            print("\n" + "=" * 60)
            print("📊 TEST RESULTS SUMMARY")
            print("=" * 60)
            print(f"CSV Preview: {'✅ PASS' if csv_success else '❌ FAIL'}")
            print(f"Start Training: {'✅ PASS' if start_success else '❌ FAIL'}")
            print(f"Training Status: {'✅ PASS' if status_success else '❌ FAIL'}")
            print(f"Training SSE: {'✅ PASS' if training_sse_success else '❌ FAIL'}")
            print(f"Progress SSE: {'✅ PASS' if progress_sse_success else '❌ FAIL'}")
            print(f"Stop Training: {'✅ PASS' if stop_success else '❌ FAIL'}")
            
            # Overall result
            all_tests = [csv_success, start_success, status_success, 
                        training_sse_success, progress_sse_success, stop_success]
            overall_success = all(all_tests)
            
            print(f"\n🎯 OVERALL RESULT: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
            
        else:
            print("❌ Start Training failed, skipping other tests")
            
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 