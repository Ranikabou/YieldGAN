#!/usr/bin/env python3
"""
Simple startup script for the Treasury GAN Dashboard.
This script handles setup and launches the dashboard.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'torch', 'numpy', 'pandas', 'sklearn', 'matplotlib', 
        'seaborn', 'scipy', 'aiohttp', 'pyyaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please install dependencies with: pip install -r requirements.txt")
        return False
    
    logger.info("All required dependencies are installed")
    return True

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data/csv',
        'data/processed', 
        'checkpoints',
        'results',
        'results/plots',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_config_files():
    """Check if configuration files exist."""
    config_files = [
        'config/gan_config.yaml',
        'config/csv_config.yaml'
    ]
    
    missing_configs = []
    for config_file in config_files:
        if not Path(config_file).exists():
            missing_configs.append(config_file)
    
    if missing_configs:
        logger.warning(f"Missing configuration files: {', '.join(missing_configs)}")
        logger.info("Some features may not work without proper configuration")
        return False
    
    logger.info("Configuration files found")
    return True

def start_dashboard():
    """Start the GAN dashboard."""
    try:
        logger.info("Starting Treasury GAN Dashboard...")
        
        # Import and start dashboard
        from gan_dashboard import GANDashboard
        import asyncio
        
        async def main():
            dashboard = GANDashboard()
            await dashboard.start()
            
            # Keep the server running
            while True:
                await asyncio.sleep(1)
        
        asyncio.run(main())
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        return False
    
    return True

def main():
    """Main startup function."""
    logger.info("🚀 Treasury GAN Dashboard Startup")
    logger.info("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check config files
    check_config_files()
    
    logger.info("✅ System ready!")
    logger.info("🌐 Dashboard will be available at: http://localhost:8080")
    logger.info("📖 Press Ctrl+C to stop the dashboard")
    logger.info("=" * 50)
    
    # Start dashboard
    if not start_dashboard():
        sys.exit(1)

if __name__ == "__main__":
    main() 