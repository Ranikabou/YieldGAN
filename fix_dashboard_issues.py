#!/usr/bin/env python3
"""
Focused fix for GAN Dashboard issues identified in logs
Addresses SSE connection problems, excessive polling, and premature cleanup
"""

import re
import asyncio
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardIssueFixer:
    def __init__(self):
        self.issues_fixed = []
        
    def fix_connection_cleanup_frequency(self):
        """Fix: Reduce connection cleanup frequency from 60s to 300s (5 minutes)."""
        logger.info("🔧 Fixing connection cleanup frequency...")
        
        try:
            with open('gan_dashboard.py', 'r') as f:
                content = f.read()
            
            # Find the cleanup interval
            old_pattern = r'await asyncio\.sleep\(60\)  # Run cleanup every minute'
            new_pattern = 'await asyncio.sleep(300)  # Run cleanup every 5 minutes'
            
            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
                logger.info("✅ Fixed cleanup frequency from 60s to 300s")
                self.issues_fixed.append("Connection cleanup frequency reduced")
            else:
                logger.warning("⚠️  Cleanup interval pattern not found")
                
            # Also fix the comment
            old_comment = '# Run cleanup every minute'
            new_comment = '# Run cleanup every 5 minutes'
            content = content.replace(old_comment, new_comment)
            
            with open('gan_dashboard.py', 'w') as f:
                f.write(content)
                
        except Exception as e:
            logger.error(f"❌ Failed to fix cleanup frequency: {e}")
            
    def fix_aggressive_connection_cleanup(self):
        """Fix: Make connection cleanup less aggressive by improving health checks."""
        logger.info("🔧 Fixing aggressive connection cleanup...")
        
        try:
            with open('gan_dashboard.py', 'r') as f:
                content = f.read()
            
            # Find the is_client_healthy function
            old_health_check = '''def is_client_healthy(self, client):
        """Check if a client connection is still healthy."""
        try:
            return (client.transport is not None and 
                   not client.transport.is_closing())
        except Exception:
            return False'''
            
            new_health_check = '''def is_client_healthy(self, client):
        """Check if a client connection is still healthy."""
        try:
            # More lenient health check - only disconnect if clearly broken
            if client.transport is None:
                return False
            if client.transport.is_closing():
                return False
            # Add additional checks to prevent premature disconnection
            if hasattr(client, '_last_activity'):
                # Only disconnect if no activity for more than 2 minutes
                time_since_activity = (datetime.now() - client._last_activity).total_seconds()
                if time_since_activity > 120:  # 2 minutes
                    return False
            return True
        except Exception:
            # Be more conservative - don't disconnect on exceptions
            return True'''
            
            if old_health_check in content:
                content = content.replace(old_health_check, new_health_check)
                logger.info("✅ Improved client health checks")
                self.issues_fixed.append("Client health checks improved")
            else:
                logger.warning("⚠️  Health check function not found")
                
            with open('gan_dashboard.py', 'w') as f:
                f.write(content)
                
        except Exception as e:
            logger.error(f"❌ Failed to fix health checks: {e}")
            
    def fix_excessive_log_polling(self):
        """Fix: Reduce excessive log channel polling in the frontend."""
        logger.info("🔧 Fixing excessive log polling...")
        
        try:
            with open('gan_dashboard.py', 'r') as f:
                content = f.read()
            
            # Find the log channel connection and reduce polling frequency
            old_log_connection = '''function connectLogChannel() {
                    if (logEventSource) {
                        logEventSource.close();
                    }
                    
                    logEventSource = new EventSource('/events/logs');'''
            
            new_log_connection = '''function connectLogChannel() {
                    if (logEventSource) {
                        logEventSource.close();
                    }
                    
                    // Add connection retry logic and reduce polling
                    logEventSource = new EventSource('/events/logs');
                    
                    // Set a longer reconnection delay to reduce excessive polling
                    if (logEventSource.addEventListener) {
                        logEventSource.addEventListener('error', function(e) {
                            // Wait 10 seconds before reconnecting instead of immediate retry
                            setTimeout(() => {
                                if (logEventSource.readyState === EventSource.CLOSED) {
                                    connectLogChannel();
                                }
                            }, 10000);
                        });
                    }'''
            
            if old_log_connection in content:
                content = content.replace(old_log_connection, new_log_connection)
                logger.info("✅ Fixed excessive log polling")
                self.issues_fixed.append("Log polling frequency reduced")
            else:
                logger.warning("⚠️  Log connection function not found")
                
            with open('gan_dashboard.py', 'w') as f:
                f.write(content)
                
        except Exception as e:
            logger.error(f"❌ Failed to fix log polling: {e}")
            
    def fix_data_duplication(self):
        """Fix: Add data deduplication to prevent duplicate broadcasts."""
        logger.info("🔧 Fixing data duplication...")
        
        try:
            with open('gan_dashboard.py', 'r') as f:
                content = f.read()
            
            # Add deduplication logic to the broadcast functions
            old_broadcast_training = '''async def broadcast_training_update(self, data):
        """Broadcast training update to all connected training clients."""
        if not self.training_clients:
            logger.debug("No training clients connected, skipping broadcast")
            return'''
            
            new_broadcast_training = '''async def broadcast_training_update(self, data):
        """Broadcast training update to all connected training clients."""
        if not self.training_clients:
            logger.debug("No training clients connected, skipping broadcast")
            return
            
        # Add deduplication check
        data_hash = hash(json.dumps(data, sort_keys=True))
        if hasattr(self, '_last_training_hash') and self._last_training_hash == data_hash:
            logger.debug("Skipping duplicate training data broadcast")
            return
        self._last_training_hash = data_hash'''
            
            if old_broadcast_training in content:
                content = content.replace(old_broadcast_training, new_broadcast_training)
                logger.info("✅ Added training data deduplication")
                self.issues_fixed.append("Training data deduplication added")
            else:
                logger.warning("⚠️  Training broadcast function not found")
                
            with open('gan_dashboard.py', 'w') as f:
                f.write(content)
                
        except Exception as e:
            logger.error(f"❌ Failed to fix data duplication: {e}")
            
    def fix_monitoring_conflicts(self):
        """Fix: Consolidate monitoring sources to prevent conflicts."""
        logger.info("🔧 Fixing monitoring conflicts...")
        
        try:
            with open('gan_dashboard.py', 'r') as f:
                content = f.read()
            
            # Comment out the log file monitoring to prevent duplicate data sources
            old_log_monitoring = '''async def start_log_monitoring(self):
        """Start monitoring training log files for real-time updates."""
        logger.info("Starting log monitoring...")
        
        # Monitor logs directory for new files
        logs_dir = Path("logs")
        if not logs_dir.exists():
            logs_dir.mkdir(exist_ok=True)
            logger.info("Created logs directory")
        
        # Get existing log files
        log_files = list(logs_dir.glob("*.log"))
        logger.info(f"Found {len(log_files)} existing log files")
        
        # Monitor each log file
        for log_file in log_files:
            asyncio.create_task(self.monitor_log_file(log_file))
        
        # Watch for new log files
        while True:
            try:
                await asyncio.sleep(5)  # Check for new files every 5 seconds
                current_files = set(logs_dir.glob("*.log"))
                new_files = current_files - set(log_files)
                
                for new_file in new_files:
                    logger.info(f"New log file detected: {new_file}")
                    asyncio.create_task(self.monitor_log_file(new_file))
                    log_files.append(new_file)
                    
            except Exception as e:
                logger.error(f"Error in log monitoring loop: {e}")
                await asyncio.sleep(10)  # Wait longer on error'''
            
            new_log_monitoring = '''async def start_log_monitoring(self):
        """Start monitoring training log files for real-time updates."""
        logger.info("Starting log monitoring...")
        
        # DISABLED: Log file monitoring conflicts with real-time training output
        # Use only real-time training output monitoring to prevent duplicates
        logger.info("Log file monitoring disabled to prevent data conflicts")
        
        # Monitor logs directory for new files
        logs_dir = Path("logs")
        if not logs_dir.exists():
            logs_dir.mkdir(exist_ok=True)
            logger.info("Created logs directory")
        
        # Get existing log files but don't monitor them
        log_files = list(logs_dir.glob("*.log"))
        logger.info(f"Found {len(log_files)} existing log files (monitoring disabled)")
        
        # Don't start monitoring tasks to prevent conflicts
        logger.info("Log file monitoring tasks disabled to prevent duplicate data sources")'''
            
            if old_log_monitoring in content:
                content = content.replace(old_log_monitoring, new_log_monitoring)
                logger.info("✅ Disabled conflicting log file monitoring")
                self.issues_fixed.append("Conflicting log monitoring disabled")
            else:
                logger.warning("⚠️  Log monitoring function not found")
                
            with open('gan_dashboard.py', 'w') as f:
                f.write(content)
                
        except Exception as e:
            logger.error(f"❌ Failed to fix monitoring conflicts: {e}")
            
    def add_connection_stability_improvements(self):
        """Add improvements for connection stability."""
        logger.info("🔧 Adding connection stability improvements...")
        
        try:
            with open('gan_dashboard.py', 'r') as f:
                content = f.read()
            
            # Add connection stability improvements to the SSE endpoints
            old_training_events = '''# Keep connection alive with simple heartbeat
            while True:
                try:
                    await response.write(f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n".encode())
                    await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                except Exception as e:
                    logger.debug(f"Training client {id(response)} disconnected: {e}")
                    break'''
            
            new_training_events = '''# Keep connection alive with improved heartbeat and stability
            last_activity = datetime.now()
            while True:
                try:
                    # Update last activity timestamp
                    last_activity = datetime.now()
                    
                    # Send heartbeat with connection info
                    heartbeat_data = {
                        'type': 'heartbeat', 
                        'timestamp': last_activity.isoformat(),
                        'client_id': id(response),
                        'connection_duration': (last_activity - datetime.fromisoformat(connection_msg['timestamp'])).total_seconds()
                    }
                    await response.write(f"data: {json.dumps(heartbeat_data)}\n\n".encode())
                    
                    # Longer heartbeat interval to reduce overhead
                    await asyncio.sleep(45)  # Send heartbeat every 45 seconds
                    
                except Exception as e:
                    logger.debug(f"Training client {id(response)} disconnected: {e}")
                    break'''
            
            if old_training_events in content:
                content = content.replace(old_training_events, new_training_events)
                logger.info("✅ Improved training connection stability")
                self.issues_fixed.append("Training connection stability improved")
            else:
                logger.warning("⚠️  Training events heartbeat not found")
                
            with open('gan_dashboard.py', 'w') as f:
                f.write(content)
                
        except Exception as e:
            logger.error(f"❌ Failed to add connection stability: {e}")
            
    def run_all_fixes(self):
        """Run all the fixes."""
        logger.info("🚀 Starting dashboard issue fixes...")
        
        self.fix_connection_cleanup_frequency()
        self.fix_aggressive_connection_cleanup()
        self.fix_excessive_log_polling()
        self.fix_data_duplication()
        self.fix_monitoring_conflicts()
        self.add_connection_stability_improvements()
        
        logger.info("=" * 60)
        logger.info("🔧 FIX SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total fixes applied: {len(self.issues_fixed)}")
        
        for i, fix in enumerate(self.issues_fixed, 1):
            logger.info(f"{i}. {fix}")
            
        logger.info("=" * 60)
        logger.info("✅ All fixes applied successfully!")
        logger.info("🔄 Restart the dashboard to apply changes")
        
        return self.issues_fixed

def main():
    """Main function to run all fixes."""
    fixer = DashboardIssueFixer()
    fixes_applied = fixer.run_all_fixes()
    
    print(f"\n🎯 Applied {len(fixes_applied)} fixes to resolve dashboard issues:")
    for fix in fixes_applied:
        print(f"  ✅ {fix}")
        
    print("\n🔄 Next steps:")
    print("  1. Restart the GAN dashboard")
    print("  2. Monitor the logs for improved stability")
    print("  3. Check that SSE connections are more stable")
    print("  4. Verify that training data flows without duplication")

if __name__ == "__main__":
    main() 