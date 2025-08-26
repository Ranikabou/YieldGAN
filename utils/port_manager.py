#!/usr/bin/env python3
"""
Centralized Port Management Utility
Handles dynamic port discovery, configuration updates, and URL resolution for dashboard components.
"""

import socket
import yaml
import json
import logging
import os
import time
import requests
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class PortManager:
    """Centralized port management for the GAN dashboard system."""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.config_file = self.project_root / "config" / "gan_config.yaml"
        self.runtime_config_file = self.project_root / ".dashboard_runtime.json"
        
        # Default port preferences
        self.preferred_ports = [8081, 8082, 8083, 8084, 8085]
        self.current_dashboard_port = None
        self.current_dashboard_url = None
        
    def find_free_port(self, start_port: int = 8081) -> Optional[int]:
        """Find a free port starting from start_port."""
        # Always try preferred ports first
        for port in self.preferred_ports:
            if port >= start_port:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('localhost', port))
                        return port
                except OSError:
                    continue
        
        # If no preferred ports available, try sequential search
        for port in range(start_port, start_port + 100):
            if port in self.preferred_ports:
                continue  # Already tried
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        
        return None
    
    def discover_active_dashboard(self) -> Optional[str]:
        """Discover if there's already a dashboard running and return its URL."""
        for port in self.preferred_ports:
            url = f"http://localhost:{port}"
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    # Check if it's actually our dashboard by looking for specific content
                    if 'Treasury GAN Dashboard' in response.text or 'GAN Dashboard' in response.text:
                        logger.info(f"🎯 Found active dashboard at {url}")
                        self.current_dashboard_port = port
                        self.current_dashboard_url = url
                        self.save_runtime_config(port, url)
                        return url
            except (requests.RequestException, requests.ConnectionError):
                continue
        
        return None
    
    def save_runtime_config(self, port: int, url: str):
        """Save runtime configuration to file for other processes to read."""
        runtime_config = {
            "dashboard_port": port,
            "dashboard_url": url,
            "updated_at": time.time(),
            "status": "active"
        }
        
        try:
            with open(self.runtime_config_file, 'w') as f:
                json.dump(runtime_config, f, indent=2)
            logger.info(f"💾 Saved runtime config: {url}")
        except Exception as e:
            logger.warning(f"Failed to save runtime config: {e}")
    
    def load_runtime_config(self) -> Optional[Dict]:
        """Load runtime configuration from file."""
        if not self.runtime_config_file.exists():
            return None
        
        try:
            with open(self.runtime_config_file, 'r') as f:
                config = json.load(f)
            
            # Check if config is recent (within last 30 minutes)
            if time.time() - config.get('updated_at', 0) > 1800:
                logger.debug("Runtime config is stale, ignoring")
                return None
            
            # Verify the dashboard is still active
            url = config.get('dashboard_url')
            if url:
                try:
                    response = requests.get(url, timeout=2)
                    if response.status_code == 200:
                        return config
                except requests.RequestException:
                    pass
            
            return None
        except Exception as e:
            logger.debug(f"Failed to load runtime config: {e}")
            return None
    
    def update_main_config(self, port: int, url: str):
        """Update the main configuration file with the current dashboard URL."""
        try:
            # Load existing config
            config = {}
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f) or {}
            
            # Update dashboard section
            if 'dashboard' not in config:
                config['dashboard'] = {}
            
            config['dashboard']['url'] = url
            config['dashboard']['port'] = port
            config['dashboard']['auto_detected'] = True
            
            # Save updated config
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"🔧 Updated main config with dashboard URL: {url}")
            
        except Exception as e:
            logger.error(f"Failed to update main config: {e}")
    
    def get_dashboard_url(self, fallback_detection: bool = True) -> str:
        """Get the current dashboard URL with automatic detection."""
        # 1. Check runtime config first
        runtime_config = self.load_runtime_config()
        if runtime_config:
            url = runtime_config['dashboard_url']
            logger.info(f"📋 Using dashboard URL from runtime config: {url}")
            return url
        
        # 2. Try to discover active dashboard
        if fallback_detection:
            discovered_url = self.discover_active_dashboard()
            if discovered_url:
                return discovered_url
        
        # 3. Check main config file
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f) or {}
                
                url = config.get('dashboard', {}).get('url')
                if url:
                    # Verify this URL is actually accessible
                    try:
                        response = requests.get(url, timeout=2)
                        if response.status_code == 200:
                            logger.info(f"📝 Using dashboard URL from config: {url}")
                            return url
                    except requests.RequestException:
                        logger.warning(f"Config URL {url} not accessible")
        except Exception as e:
            logger.debug(f"Failed to read main config: {e}")
        
        # 4. Default fallback
        default_url = "http://localhost:8081"
        logger.warning(f"⚠️ Using default dashboard URL: {default_url}")
        return default_url
    
    def register_dashboard_startup(self, port: int):
        """Register that a dashboard has started on the given port."""
        url = f"http://localhost:{port}"
        self.current_dashboard_port = port
        self.current_dashboard_url = url
        
        # Save to both configs
        self.save_runtime_config(port, url)
        self.update_main_config(port, url)
        
        logger.info(f"✅ Dashboard registered at {url}")
        return url
    
    def cleanup_runtime_config(self):
        """Clean up runtime configuration on dashboard shutdown."""
        try:
            if self.runtime_config_file.exists():
                self.runtime_config_file.unlink()
            logger.info("🧹 Cleaned up runtime configuration")
        except Exception as e:
            logger.debug(f"Failed to cleanup runtime config: {e}")
    
    def wait_for_dashboard(self, timeout: int = 30) -> Optional[str]:
        """Wait for a dashboard to become available."""
        logger.info(f"⏳ Waiting for dashboard (timeout: {timeout}s)...")
        
        for _ in range(timeout):
            url = self.discover_active_dashboard()
            if url:
                return url
            time.sleep(1)
        
        logger.warning("⏰ Timeout waiting for dashboard")
        return None
    
    def get_all_dashboard_urls(self) -> List[str]:
        """Get all possible dashboard URLs for testing purposes."""
        urls = []
        for port in self.preferred_ports:
            urls.append(f"http://localhost:{port}")
        return urls
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is currently in use."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return False
        except OSError:
            return True

# Global instance for easy access
_port_manager = None

def get_port_manager() -> PortManager:
    """Get the global port manager instance."""
    global _port_manager
    if _port_manager is None:
        _port_manager = PortManager()
    return _port_manager

def get_dashboard_url() -> str:
    """Convenient function to get the current dashboard URL."""
    return get_port_manager().get_dashboard_url()

def register_dashboard(port: int) -> str:
    """Convenient function to register a dashboard startup."""
    return get_port_manager().register_dashboard_startup(port)

def cleanup_dashboard() -> None:
    """Convenient function to cleanup dashboard configuration."""
    get_port_manager().cleanup_runtime_config()

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    pm = PortManager()
    
    print("🔍 Testing Port Manager...")
    print(f"Current dashboard URL: {pm.get_dashboard_url()}")
    print(f"Available ports: {pm.get_all_dashboard_urls()}")
    
    # Test port discovery
    free_port = pm.find_free_port()
    if free_port:
        print(f"Next free port: {free_port}")
    
    # Test dashboard discovery
    active_dashboard = pm.discover_active_dashboard()
    if active_dashboard:
        print(f"Active dashboard found: {active_dashboard}")
    else:
        print("No active dashboard found") 