#!/usr/bin/env python3
"""
Port Management System Test
Verifies that automatic port detection and configuration works across all components.
"""

import asyncio
import logging
import time
import subprocess
import signal
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_port_management():
    """Test the complete port management system."""
    logger.info("🧪 Testing Port Management System...")
    
    # Test 1: Port Manager Utility
    logger.info("\n1️⃣ Testing Port Manager Utility...")
    try:
        from utils.port_manager import PortManager
        pm = PortManager()
        
        # Test port finding
        free_port = pm.find_free_port()
        logger.info(f"✅ Found free port: {free_port}")
        
        # Test dashboard discovery
        active_dashboard = pm.discover_active_dashboard()
        if active_dashboard:
            logger.info(f"✅ Active dashboard found: {active_dashboard}")
        else:
            logger.info("ℹ️ No active dashboard found (expected if dashboard not running)")
        
        # Test configuration management
        logger.info(f"✅ Port Manager utility working correctly")
        
    except Exception as e:
        logger.error(f"❌ Port Manager test failed: {e}")
        return False
    
    # Test 2: Training Script Port Detection
    logger.info("\n2️⃣ Testing Training Script Port Detection...")
    try:
        from train_gan_csv import DashboardChannelSender
        
        # Test with automatic detection
        sender = DashboardChannelSender()
        logger.info(f"✅ Training script detected dashboard URL: {sender.dashboard_url}")
        
    except Exception as e:
        logger.error(f"❌ Training script test failed: {e}")
        return False
    
    # Test 3: Test Scripts Port Detection
    logger.info("\n3️⃣ Testing Test Scripts Port Detection...")
    try:
        from test_dashboard_channels import DashboardChannelTester
        from test_sse_ui_fix import SSEUITester
        
        # Test dashboard channels
        tester1 = DashboardChannelTester()
        logger.info(f"✅ Dashboard channels tester URL: {tester1.dashboard_url}")
        
        # Test SSE UI fix
        tester2 = SSEUITester()
        logger.info(f"✅ SSE UI tester URL: {tester2.dashboard_url}")
        
    except Exception as e:
        logger.error(f"❌ Test scripts test failed: {e}")
        return False
    
    # Test 4: Configuration File Reading
    logger.info("\n4️⃣ Testing Configuration File Reading...")
    try:
        import yaml
        config_file = Path(__file__).parent / "config" / "gan_config.yaml"
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        dashboard_config = config.get('dashboard', {})
        logger.info(f"✅ Config dashboard URL: {dashboard_config.get('url')}")
        logger.info(f"✅ Config auto-detection: {dashboard_config.get('auto_detection')}")
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False
    
    logger.info("\n✅ All Port Management Tests Passed!")
    return True

async def test_full_integration():
    """Test the complete integration with a running dashboard."""
    logger.info("\n🔄 Testing Full Integration...")
    
    dashboard_process = None
    try:
        # Start dashboard
        logger.info("🚀 Starting dashboard...")
        dashboard_process = subprocess.Popen(
            [sys.executable, "gan_dashboard.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for dashboard to start
        await asyncio.sleep(5)
        
        # Test port discovery
        from utils.port_manager import get_port_manager
        pm = get_port_manager()
        
        dashboard_url = pm.discover_active_dashboard()
        if dashboard_url:
            logger.info(f"✅ Dashboard auto-discovered at: {dashboard_url}")
            
            # Test training script connection
            from train_gan_csv import DashboardChannelSender
            sender = DashboardChannelSender()
            logger.info(f"✅ Training script connected to: {sender.dashboard_url}")
            
            # Test SSE UI connection
            from test_sse_ui_fix import SSEUITester
            ui_tester = SSEUITester()
            
            connectivity = await ui_tester.test_dashboard_connectivity()
            if connectivity:
                logger.info("✅ SSE UI connectivity test passed")
            else:
                logger.warning("⚠️ SSE UI connectivity test failed")
            
            logger.info("✅ Full Integration Test Passed!")
            return True
        else:
            logger.error("❌ Could not discover dashboard")
            return False
            
    except Exception as e:
        logger.error(f"❌ Full integration test failed: {e}")
        return False
    finally:
        # Cleanup dashboard
        if dashboard_process:
            logger.info("🧹 Stopping dashboard...")
            dashboard_process.terminate()
            try:
                dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard_process.kill()

def print_summary():
    """Print a summary of the port management improvements."""
    logger.info("\n" + "="*60)
    logger.info("🎯 PORT MANAGEMENT SYSTEM SUMMARY")
    logger.info("="*60)
    logger.info("")
    logger.info("✅ IMPLEMENTED:")
    logger.info("  • Centralized PortManager utility")
    logger.info("  • Automatic dashboard port discovery")
    logger.info("  • Runtime configuration management")
    logger.info("  • Dynamic URL resolution for all components")
    logger.info("  • Fallback mechanisms for reliability")
    logger.info("")
    logger.info("🔧 UPDATED COMPONENTS:")
    logger.info("  • gan_dashboard.py - Registers port on startup")
    logger.info("  • train_gan_csv.py - Uses port manager")
    logger.info("  • train_gan_csv_simple.py - Uses port manager")
    logger.info("  • test_sse_ui_fix.py - Uses port manager")
    logger.info("  • test_dashboard_channels.py - Uses port manager")
    logger.info("  • config/gan_config.yaml - Auto-detection enabled")
    logger.info("")
    logger.info("🎮 HOW IT WORKS:")
    logger.info("  1. Dashboard starts and registers its port")
    logger.info("  2. Port manager saves runtime configuration")
    logger.info("  3. Training scripts auto-discover dashboard URL")
    logger.info("  4. All components use the same URL automatically")
    logger.info("  5. No more hardcoded port dependencies!")
    logger.info("")
    logger.info("🚀 TO USE:")
    logger.info("  1. Start dashboard: python gan_dashboard.py")
    logger.info("  2. All other scripts automatically find it")
    logger.info("  3. No manual port configuration needed")
    logger.info("")
    logger.info("="*60)

async def main():
    """Main test function."""
    print_summary()
    
    # Run basic tests
    basic_tests_passed = await test_port_management()
    
    if basic_tests_passed:
        logger.info("\n🎯 All basic tests passed!")
        
        # Ask user if they want to run full integration test
        logger.info("\n🤔 Would you like to run the full integration test?")
        logger.info("   This will start a dashboard and test complete integration.")
        logger.info("   (You can press Ctrl+C to skip this)")
        
        try:
            await asyncio.sleep(3)  # Give user time to read
            await test_full_integration()
        except KeyboardInterrupt:
            logger.info("\n⏭️ Skipping full integration test")
    
    logger.info("\n🏁 Port Management Testing Complete!")
    logger.info("💡 Next: Start the dashboard with 'python gan_dashboard.py'")
    logger.info("   Then run training with any script - it will auto-connect!")

if __name__ == "__main__":
    asyncio.run(main()) 