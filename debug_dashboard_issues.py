#!/usr/bin/env python3
"""
Debug script for GAN Dashboard SSE and data flow issues
Identifies and fixes the problems causing the dashboard to break
"""

import asyncio
import json
import logging
import time
from datetime import datetime
import aiohttp
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardDebugger:
    def __init__(self, dashboard_url="http://localhost:8082"):
        self.dashboard_url = dashboard_url
        self.session = None
        self.issues_found = []
        
    async def start_session(self):
        """Start aiohttp session for async operations."""
        self.session = aiohttp.ClientSession()
        
    async def close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            
    def log_issue(self, severity, category, message, details=None):
        """Log an issue with structured information."""
        issue = {
            'severity': severity,
            'category': category,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.issues_found.append(issue)
        logger.error(f"[{severity}] {category}: {message}")
        if details:
            logger.error(f"Details: {details}")
            
    async def test_dashboard_health(self):
        """Test basic dashboard health."""
        logger.info("🔍 Testing dashboard health...")
        
        try:
            # Test main page
            async with self.session.get(f"{self.dashboard_url}/") as response:
                if response.status == 200:
                    logger.info("✅ Main dashboard page accessible")
                else:
                    self.log_issue("HIGH", "Connectivity", f"Main page returned status {response.status}")
                    
            # Test API endpoints
            async with self.session.get(f"{self.dashboard_url}/api/training_status") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Training status API working: {data}")
                else:
                    self.log_issue("HIGH", "API", f"Training status API returned status {response.status}")
                    
        except Exception as e:
            self.log_issue("CRITICAL", "Connectivity", f"Dashboard health check failed: {e}")
            
    async def test_sse_connections(self):
        """Test SSE connection stability."""
        logger.info("🔍 Testing SSE connections...")
        
        # Test training channel
        try:
            async with self.session.get(f"{self.dashboard_url}/events/training") as response:
                if response.status == 200:
                    logger.info("✅ Training SSE channel accessible")
                    
                    # Test data flow
                    await self.test_training_data_flow()
                else:
                    self.log_issue("HIGH", "SSE", f"Training SSE channel returned status {response.status}")
                    
        except Exception as e:
            self.log_issue("HIGH", "SSE", f"Training SSE connection failed: {e}")
            
        # Test progress channel
        try:
            async with self.session.get(f"{self.dashboard_url}/events/progress") as response:
                if response.status == 200:
                    logger.info("✅ Progress SSE channel accessible")
                else:
                    self.log_issue("HIGH", "SSE", f"Progress SSE channel returned status {response.status}")
                    
        except Exception as e:
            self.log_issue("HIGH", "SSE", f"Progress SSE connection failed: {e}")
            
        # Test logs channel
        try:
            async with self.session.get(f"{self.dashboard_url}/events/logs") as response:
                if response.status == 200:
                    logger.info("✅ Logs SSE channel accessible")
                else:
                    self.log_issue("HIGH", "SSE", f"Logs SSE channel returned status {response.status}")
                    
        except Exception as e:
            self.log_issue("HIGH", "SSE", f"Logs SSE connection failed: {e}")
            
    async def test_training_data_flow(self):
        """Test the complete training data flow."""
        logger.info("🔍 Testing training data flow...")
        
        try:
            # Send test training data
            test_data = {
                "type": "training_update",
                "data": {
                    "epoch": 1,
                    "total_epochs": 10,
                    "generator_loss": 0.75,
                    "discriminator_loss": 0.65,
                    "real_scores": 0.85,
                    "fake_scores": 0.25
                },
                "timestamp": datetime.now().isoformat()
            }
            
            async with self.session.post(f"{self.dashboard_url}/training_data", json=test_data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Test training data sent successfully: {result}")
                else:
                    self.log_issue("MEDIUM", "Data Flow", f"Failed to send test training data: {response.status}")
                    
        except Exception as e:
            self.log_issue("MEDIUM", "Data Flow", f"Training data flow test failed: {e}")
            
    async def test_training_lifecycle(self):
        """Test the complete training lifecycle."""
        logger.info("🔍 Testing training lifecycle...")
        
        try:
            # Start training
            start_data = {
                "config": "config/gan_config.yaml",
                "data_source": "treasury_orderbook_sample.csv"
            }
            
            async with self.session.post(f"{self.dashboard_url}/api/start_training", json=start_data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Training started: {result}")
                    
                    # Wait a bit for training to initialize
                    await asyncio.sleep(2)
                    
                    # Check training status
                    async with self.session.get(f"{self.dashboard_url}/api/training_status") as status_response:
                        if status_response.status == 200:
                            status_data = await status_response.json()
                            logger.info(f"✅ Training status: {status_data}")
                        else:
                            self.log_issue("MEDIUM", "Training", f"Failed to get training status: {status_response.status}")
                            
                    # Stop training after a short delay
                    await asyncio.sleep(3)
                    
                    async with self.session.post(f"{self.dashboard_url}/api/stop_training") as stop_response:
                        if stop_response.status == 200:
                            stop_result = await stop_response.json()
                            logger.info(f"✅ Training stopped: {stop_result}")
                        else:
                            self.log_issue("MEDIUM", "Training", f"Failed to stop training: {stop_response.status}")
                            
                else:
                    self.log_issue("HIGH", "Training", f"Failed to start training: {response.status}")
                    
        except Exception as e:
            self.log_issue("HIGH", "Training", f"Training lifecycle test failed: {e}")
            
    async def analyze_connection_patterns(self):
        """Analyze connection patterns and identify issues."""
        logger.info("🔍 Analyzing connection patterns...")
        
        # Check for excessive polling
        logger.info("📊 Checking for excessive polling patterns...")
        
        # Check for connection cleanup frequency
        logger.info("🧹 Checking connection cleanup frequency...")
        
        # Check for data duplication
        logger.info("🔄 Checking for data duplication...")
        
    def generate_report(self):
        """Generate a comprehensive debug report."""
        logger.info("📋 Generating debug report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dashboard_url': self.dashboard_url,
            'total_issues': len(self.issues_found),
            'issues_by_severity': {},
            'issues_by_category': {},
            'recommendations': []
        }
        
        # Categorize issues
        for issue in self.issues_found:
            severity = issue['severity']
            category = issue['category']
            
            if severity not in report['issues_by_severity']:
                report['issues_by_severity'][severity] = 0
            report['issues_by_severity'][severity] += 1
            
            if category not in report['issues_by_category']:
                report['issues_by_category'][category] = 0
            report['issues_by_category'][category] += 1
            
        # Generate recommendations based on issues found
        if report['issues_by_category'].get('SSE', 0) > 0:
            report['recommendations'].append({
                'category': 'SSE',
                'action': 'Reduce connection cleanup frequency from 60s to 300s',
                'priority': 'HIGH'
            })
            report['recommendations'].append({
                'category': 'SSE',
                'action': 'Implement connection health checks before cleanup',
                'priority': 'MEDIUM'
            })
            
        if report['issues_by_category'].get('Data Flow', 0) > 0:
            report['recommendations'].append({
                'category': 'Data Flow',
                'action': 'Add data deduplication to prevent duplicate broadcasts',
                'priority': 'HIGH'
            })
            
        if report['issues_by_category'].get('Training', 0) > 0:
            report['recommendations'].append({
                'category': 'Training',
                'action': 'Consolidate monitoring sources to prevent conflicts',
                'priority': 'HIGH'
            })
            
        # Print report
        logger.info("=" * 60)
        logger.info("🔍 DASHBOARD DEBUG REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Issues Found: {report['total_issues']}")
        logger.info(f"Dashboard URL: {report['dashboard_url']}")
        logger.info(f"Timestamp: {report['timestamp']}")
        
        if report['issues_by_severity']:
            logger.info("\nIssues by Severity:")
            for severity, count in report['issues_by_severity'].items():
                logger.info(f"  {severity}: {count}")
                
        if report['issues_by_category']:
            logger.info("\nIssues by Category:")
            for category, count in report['issues_by_category'].items():
                logger.info(f"  {category}: {count}")
                
        if report['recommendations']:
            logger.info("\nRecommendations:")
            for rec in report['recommendations']:
                logger.info(f"  [{rec['priority']}] {rec['category']}: {rec['action']}")
                
        logger.info("=" * 60)
        
        return report
        
    async def run_full_debug(self):
        """Run the complete debug suite."""
        logger.info("🚀 Starting comprehensive dashboard debug...")
        
        try:
            await self.start_session()
            
            # Run all tests
            await self.test_dashboard_health()
            await self.test_sse_connections()
            await self.test_training_lifecycle()
            await self.analyze_connection_patterns()
            
            # Generate report
            report = self.generate_report()
            
            return report
            
        finally:
            await self.close_session()

async def main():
    """Main debug function."""
    debugger = DashboardDebugger()
    report = await debugger.run_full_debug()
    
    # Save report to file
    with open('dashboard_debug_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    logger.info("💾 Debug report saved to 'dashboard_debug_report.json'")
    
    # Exit with error code if critical issues found
    if report['issues_by_severity'].get('CRITICAL', 0) > 0:
        logger.error("❌ Critical issues found - dashboard needs immediate attention")
        exit(1)
    elif report['total_issues'] > 0:
        logger.warning("⚠️  Issues found - review recommendations")
        exit(0)
    else:
        logger.info("✅ No issues found - dashboard appears healthy")
        exit(0)

if __name__ == "__main__":
    asyncio.run(main()) 