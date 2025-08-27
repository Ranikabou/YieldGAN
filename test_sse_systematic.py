#!/usr/bin/env python3
"""
Systematic SSE Endpoint Test Script
Tests all SSE channels to ensure they're updating systematically on the UI
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class SSESystematicTester:
    def __init__(self, base_url="http://localhost:8082"):
        self.base_url = base_url
        self.results = {
            'training': {'connected': False, 'messages': [], 'last_update': None},
            'progress': {'connected': False, 'messages': [], 'last_update': None},
            'logs': {'connected': False, 'messages': [], 'last_update': None}
        }
        
    async def test_training_channel(self):
        """Test the training SSE channel systematically."""
        print("🎯 Testing Training SSE Channel...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/events/training") as response:
                    if response.status == 200:
                        print("✅ Training channel connected successfully")
                        self.results['training']['connected'] = True
                        
                        # Monitor for messages
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])
                                    self.results['training']['messages'].append(data)
                                    self.results['training']['last_update'] = datetime.now()
                                    
                                    print(f"🎯 Training message: {data.get('type', 'unknown')} - {datetime.now().strftime('%H:%M:%S')}")
                                    
                                    # Test systematic updates
                                    if data.get('type') == 'training_update':
                                        print(f"   📊 Epoch: {data.get('data', {}).get('epoch', 'N/A')}")
                                        print(f"   📈 Gen Loss: {data.get('data', {}).get('generator_loss', 'N/A')}")
                                        print(f"   📉 Disc Loss: {data.get('data', {}).get('discriminator_loss', 'N/A')}")
                                    
                                except json.JSONDecodeError as e:
                                    print(f"❌ JSON decode error: {e}")
                                    
                                # Limit message history
                                if len(self.results['training']['messages']) > 50:
                                    self.results['training']['messages'] = self.results['training']['messages'][-50:]
                                    
                    else:
                        print(f"❌ Training channel failed: {response.status}")
                        
        except Exception as e:
            print(f"❌ Training channel error: {e}")
    
    async def test_progress_channel(self):
        """Test the progress SSE channel systematically."""
        print("📊 Testing Progress SSE Channel...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/events/progress") as response:
                    if response.status == 200:
                        print("✅ Progress channel connected successfully")
                        self.results['progress']['connected'] = True
                        
                        # Monitor for messages
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])
                                    self.results['progress']['messages'].append(data)
                                    self.results['progress']['last_update'] = datetime.now()
                                    
                                    print(f"📊 Progress message: {data.get('type', 'unknown')} - {datetime.now().strftime('%H:%M:%S')}")
                                    
                                    # Test systematic updates
                                    if data.get('type') == 'progress':
                                        print(f"   📈 Progress: {data.get('progress_percent', 'N/A')}%")
                                        print(f"   ⏰ Epoch: {data.get('epoch', 'N/A')}")
                                    
                                except json.JSONDecodeError as e:
                                    print(f"❌ JSON decode error: {e}")
                                    
                                # Limit message history
                                if len(self.results['progress']['messages']) > 50:
                                    self.results['progress']['messages'] = self.results['progress']['messages'][-50:]
                                    
                    else:
                        print(f"❌ Progress channel failed: {response.status}")
                        
        except Exception as e:
            print(f"❌ Progress channel error: {e}")
    
    async def test_logs_channel(self):
        """Test the logs SSE channel systematically."""
        print("📝 Testing Logs SSE Channel...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/events/logs") as response:
                    if response.status == 200:
                        print("✅ Logs channel connected successfully")
                        self.results['logs']['connected'] = True
                        
                        # Monitor for messages
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])
                                    self.results['logs']['messages'].append(data)
                                    self.results['logs']['last_update'] = datetime.now()
                                    
                                    print(f"📝 Log message: {data.get('type', 'unknown')} - {datetime.now().strftime('%H:%M:%S')}")
                                    
                                    # Test systematic updates
                                    if data.get('type') == 'log_entry':
                                        print(f"   📋 Source: {data.get('data', {}).get('source', 'N/A')}")
                                        print(f"   💬 Message: {data.get('data', {}).get('message', 'N/A')[:100]}...")
                                    
                                except json.JSONDecodeError as e:
                                    print(f"❌ JSON decode error: {e}")
                                    
                                # Limit message history
                                if len(self.results['logs']['messages']) > 50:
                                    self.results['logs']['messages'] = self.results['logs']['messages'][-50:]
                                    
                    else:
                        print(f"❌ Logs channel failed: {response.status}")
                        
        except Exception as e:
            print(f"❌ Logs channel error: {e}")
    
    async def test_all_channels_concurrently(self, duration=60):
        """Test all SSE channels concurrently for systematic updates."""
        print(f"🚀 Starting systematic SSE test for {duration} seconds...")
        print(f"🌐 Dashboard URL: {self.base_url}")
        print("=" * 60)
        
        # Start all channels concurrently
        tasks = [
            asyncio.create_task(self.test_training_channel()),
            asyncio.create_task(self.test_progress_channel()),
            asyncio.create_task(self.test_logs_channel())
        ]
        
        # Run for specified duration
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=duration)
        except asyncio.TimeoutError:
            print(f"\n⏰ Test completed after {duration} seconds")
        
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Print systematic analysis
        self.print_systematic_analysis()
    
    def print_systematic_analysis(self):
        """Print systematic analysis of SSE endpoint performance."""
        print("\n" + "=" * 60)
        print("📊 SYSTEMATIC SSE ANALYSIS")
        print("=" * 60)
        
        for channel, data in self.results.items():
            print(f"\n🔍 {channel.upper()} CHANNEL:")
            print(f"   ✅ Connected: {data['connected']}")
            print(f"   📨 Messages Received: {len(data['messages'])}")
            print(f"   🕐 Last Update: {data['last_update']}")
            
            if data['messages']:
                # Analyze message types
                message_types = {}
                for msg in data['messages']:
                    msg_type = msg.get('type', 'unknown')
                    message_types[msg_type] = message_types.get(msg_type, 0) + 1
                
                print(f"   📊 Message Types: {message_types}")
                
                # Check for systematic updates
                if len(data['messages']) > 1:
                    timestamps = [msg.get('timestamp') for msg in data['messages'] if msg.get('timestamp')]
                    if timestamps:
                        print(f"   ⏱️  Timestamps: {len(timestamps)} valid timestamps")
                        
                        # Check for regular intervals (heartbeats)
                        heartbeat_count = sum(1 for msg in data['messages'] if msg.get('type') == 'heartbeat')
                        print(f"   💓 Heartbeats: {heartbeat_count}")
                
                # Check for data consistency
                if channel == 'training' and any(msg.get('type') == 'training_update' for msg in data['messages']):
                    training_updates = [msg for msg in data['messages'] if msg.get('type') == 'training_update']
                    print(f"   🎯 Training Updates: {len(training_updates)}")
                    
                    # Check epoch progression
                    epochs = [msg.get('data', {}).get('epoch') for msg in training_updates if msg.get('data', {}).get('epoch')]
                    if epochs:
                        print(f"   📈 Epochs: {epochs}")
                
                elif channel == 'progress' and any(msg.get('type') == 'progress' for msg in data['messages']):
                    progress_updates = [msg for msg in data['messages'] if msg.get('type') == 'progress']
                    print(f"   📊 Progress Updates: {len(progress_updates)}")
                    
                    # Check progress percentages
                    percentages = [msg.get('progress_percent') for msg in progress_updates if msg.get('progress_percent')]
                    if percentages:
                        print(f"   📈 Progress Range: {min(percentages)}% - {max(percentages)}%")
        
        print("\n" + "=" * 60)
        print("🎯 SYSTEMATIC UPDATE ASSESSMENT:")
        
        # Overall assessment
        total_messages = sum(len(data['messages']) for data in self.results.values())
        active_channels = sum(1 for data in self.results.values() if data['connected'])
        
        if total_messages > 0 and active_channels == 3:
            print("✅ EXCELLENT: All channels active with systematic updates")
        elif total_messages > 0 and active_channels > 0:
            print("⚠️  PARTIAL: Some channels active but not all")
        else:
            print("❌ POOR: No systematic updates detected")
        
        print(f"   📊 Total Messages: {total_messages}")
        print(f"   🔌 Active Channels: {active_channels}/3")
        print("=" * 60)

async def main():
    """Main test function."""
    tester = SSESystematicTester()
    
    print("🧪 SSE Systematic Test Suite")
    print("Testing if SSE endpoints update systematically on the UI")
    print()
    
    try:
        # Test all channels concurrently
        await tester.test_all_channels_concurrently(duration=30)
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    
    print("\n🏁 Test completed")

if __name__ == "__main__":
    asyncio.run(main()) 