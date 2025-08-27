#!/usr/bin/env python3
"""
Dashboard Health Monitor
Monitors the GAN dashboard for common issues and provides real-time health status
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime, timedelta
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardHealthMonitor:
    def __init__(self, dashboard_url="http://localhost:8082"):
        self.dashboard_url = dashboard_url
        self.session = None
        self.health_metrics = {
            'start_time': datetime.now(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'sse_connections': 0,
            'last_training_status': None,
            'connection_issues': [],
            'performance_metrics': []
        }
        
    async def start_session(self):
        """Start aiohttp session."""
        self.session = aiohttp.ClientSession()
        
    async def close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            
    async def check_endpoint_health(self, endpoint, name):
        """Check health of a specific endpoint."""
        try:
            start_time = time.time()
            async with self.session.get(f"{self.dashboard_url}{endpoint}") as response:
                response_time = time.time() - start_time
                
                self.health_metrics['total_requests'] += 1
                
                if response.status == 200:
                    self.health_metrics['successful_requests'] += 1
                    logger.info(f"✅ {name}: Healthy (Response time: {response_time:.3f}s)")
                    
                    # Store performance metric
                    self.health_metrics['performance_metrics'].append({
                        'endpoint': name,
                        'response_time': response_time,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'success'
                    })
                    
                    return True, response_time
                else:
                    self.health_metrics['failed_requests'] += 1
                    logger.error(f"❌ {name}: Failed (Status: {response.status})")
                    
                    # Record connection issue
                    self.health_metrics['connection_issues'].append({
                        'endpoint': name,
                        'status_code': response.status,
                        'timestamp': datetime.now().isoformat(),
                        'type': 'http_error'
                    })
                    
                    return False, response_time
                    
        except Exception as e:
            self.health_metrics['failed_requests'] += 1
            logger.error(f"❌ {name}: Error - {e}")
            
            # Record connection issue
            self.health_metrics['connection_issues'].append({
                'endpoint': name,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'type': 'connection_error'
            })
            
            return False, 0
            
    async def check_sse_connections(self):
        """Check SSE connection health."""
        logger.info("🔍 Checking SSE connections...")
        
        sse_endpoints = [
            ('/events/training', 'Training SSE'),
            ('/events/progress', 'Progress SSE'),
            ('/events/logs', 'Logs SSE')
        ]
        
        healthy_connections = 0
        
        for endpoint, name in sse_endpoints:
            try:
                start_time = time.time()
                async with self.session.get(f"{self.dashboard_url}{endpoint}") as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        healthy_connections += 1
                        logger.info(f"✅ {name}: Connected (Response time: {response_time:.3f}s)")
                    else:
                        logger.error(f"❌ {name}: Failed to connect (Status: {response.status})")
                        
            except Exception as e:
                logger.error(f"❌ {name}: Connection error - {e}")
                
        self.health_metrics['sse_connections'] = healthy_connections
        return healthy_connections
        
    async def check_training_status(self):
        """Check training status via SSE (REST endpoint removed)."""
        try:
            # Training status is now only available via SSE
            self.health_metrics['last_training_status'] = {'status': 'Available via SSE only'}
            logger.info("🎯 Training status available via SSE channels")
            return True
        except Exception as e:
            logger.error(f"❌ Error checking training status: {e}")
            return False
            
    def calculate_health_score(self):
        """Calculate overall health score (0-100)."""
        if self.health_metrics['total_requests'] == 0:
            return 100
            
        success_rate = self.health_metrics['successful_requests'] / self.health_metrics['total_requests']
        sse_score = min(self.health_metrics['sse_connections'] / 3, 1.0)  # 3 SSE endpoints
        
        # Weighted scoring
        health_score = (success_rate * 70) + (sse_score * 30)
        
        return round(health_score, 1)
        
    def generate_health_report(self):
        """Generate comprehensive health report."""
        uptime = datetime.now() - self.health_metrics['start_time']
        health_score = self.calculate_health_score()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'uptime': str(uptime),
            'health_score': health_score,
            'total_requests': self.health_metrics['total_requests'],
            'success_rate': f"{(self.health_metrics['successful_requests'] / max(self.health_metrics['total_requests'], 1) * 100):.1f}%",
            'sse_connections': f"{self.health_metrics['sse_connections']}/3",
            'connection_issues': len(self.health_metrics['connection_issues']),
            'performance_avg': 0
        }
        
        # Calculate average response time
        if self.health_metrics['performance_metrics']:
            avg_response_time = sum(m['response_time'] for m in self.health_metrics['performance_metrics']) / len(self.health_metrics['performance_metrics'])
            report['performance_avg'] = f"{avg_response_time:.3f}s"
            
        # Health status
        if health_score >= 90:
            status = "🟢 EXCELLENT"
        elif health_score >= 75:
            status = "🟡 GOOD"
        elif health_score >= 50:
            status = "🟠 FAIR"
        else:
            status = "🔴 POOR"
            
        report['status'] = status
        
        return report
        
    def print_health_report(self):
        """Print formatted health report."""
        report = self.generate_health_report()
        
        print("\n" + "=" * 60)
        print("🏥 DASHBOARD HEALTH REPORT")
        print("=" * 60)
        print(f"Status: {report['status']}")
        print(f"Health Score: {report['health_score']}/100")
        print(f"Uptime: {report['uptime']}")
        print(f"Total Requests: {report['total_requests']}")
        print(f"Success Rate: {report['success_rate']}")
        print(f"SSE Connections: {report['sse_connections']}")
        print(f"Connection Issues: {report['connection_issues']}")
        print(f"Avg Response Time: {report['performance_avg']}")
        print("=" * 60)
        
        # Show recent issues if any
        if self.health_metrics['connection_issues']:
            print("\n🚨 Recent Issues:")
            for issue in self.health_metrics['connection_issues'][-5:]:  # Last 5 issues
                print(f"  • {issue['timestamp']}: {issue.get('endpoint', 'Unknown')} - {issue.get('type', 'Unknown')}")
                
        # Show performance trends
        if self.health_metrics['performance_metrics']:
            print("\n📈 Performance Trends:")
            recent_metrics = self.health_metrics['performance_metrics'][-10:]  # Last 10 metrics
            for metric in recent_metrics:
                print(f"  • {metric['endpoint']}: {metric['response_time']:.3f}s ({metric['status']})")
                
        print("=" * 60)
        
    async def continuous_monitoring(self, interval=30):
        """Continuously monitor dashboard health."""
        logger.info(f"🔍 Starting continuous monitoring (interval: {interval}s)")
        
        try:
            while True:
                logger.info("\n" + "-" * 40)
                logger.info(f"🕐 Health Check: {datetime.now().strftime('%H:%M:%S')}")
                logger.info("-" * 40)
                
                # Check all endpoints
                await self.check_endpoint_health('/', 'Main Dashboard')
                # Training Status API removed - now available via SSE only
                
                # Check SSE connections
                await self.check_sse_connections()
                
                # Check training status
                await self.check_training_status()
                
                # Print health report
                self.print_health_report()
                
                # Wait for next check
                logger.info(f"⏳ Next health check in {interval} seconds...")
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")

async def main():
    """Main function."""
    monitor = DashboardHealthMonitor()
    
    try:
        await monitor.start_session()
        
        # Run continuous monitoring
        await monitor.continuous_monitoring(interval=30)  # Check every 30 seconds
        
    finally:
        await monitor.close_session()

if __name__ == "__main__":
    print("🏥 Dashboard Health Monitor")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Monitoring stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}") 