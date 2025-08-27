#!/usr/bin/env python3
"""
Modern Web Dashboard for Treasury GAN Training
Integrates with the existing GAN training pipeline and provides real-time monitoring.
Now includes log file reading and separate SSE channels for training and progress data.
"""

import asyncio
import json
import aiohttp
from aiohttp import web
import logging
import os
import sys
from pathlib import Path
import threading
import time
from datetime import datetime, timedelta
import subprocess
import signal
import psutil
import re
import glob
import socket
import yaml
import pandas as pd
import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import port management utility
from utils.port_manager import get_port_manager, register_dashboard, cleanup_dashboard

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_free_port(start_port=8081):
    """Find a free port starting from start_port."""
    # Always try 8081 first
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 8081))
            return 8081
    except OSError:
        pass
    
    # If 8081 is not available, try other ports
    for port in range(start_port, start_port + 100):
        if port == 8081:  # Skip 8081 since we already tried it
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

class GANDashboard:
    def __init__(self, host='localhost', port=None):
        self.host = host
        self.port_manager = get_port_manager()
        
        # Use port manager for intelligent port selection
        if port:
            self.port = port
        else:
            self.port = self.port_manager.find_free_port()
            
        if not self.port:
            raise RuntimeError("No free ports available")
        
        self.app = web.Application()
        self.runner = None
        
        # Training state
        self.training_process = None
        self.training_status = "idle"
        self.training_metrics = {}
        self.training_logs = []
        
        # Log file monitoring
        self.log_files = []
        self.last_log_positions = {}
        self.log_monitoring_active = False
        
        # SSE clients - separate channels like test_separate_channels.py
        self.training_clients = set()
        self.progress_clients = set()
        self.log_clients = set()
        
        # Pending messages for when SSE broadcast fails
        self.pending_completion_message = None
        self.pending_status_message = None
        
        # Set up routes
        self.setup_routes()
        
        # Start background task to broadcast pending messages
        asyncio.create_task(self.broadcast_pending_messages_periodically())
        
        # Start background tasks
        self.start_background_tasks()
        
        logger.info(f"Dashboard initialized on port {self.port}")
    
    def setup_routes(self):
        """Set up web routes."""
        # Static files (commented out for now)
        # self.app.router.add_static('/static', 'static')
        
        # Main pages
        self.app.router.add_get('/', self.dashboard)
        self.app.router.add_get('/training', self.training_page)
        self.app.router.add_get('/evaluation', self.evaluation_page)
        self.app.router.add_get('/models', self.models_page)
        self.app.router.add_get('/test_sse_debug', self.test_sse_debug)
        self.app.router.add_get('/test_minimal_sse', self.test_minimal_sse)
        self.app.router.add_get('/test_main_dashboard_sse', self.test_main_dashboard_sse)
        self.app.router.add_get('/test_ui_updates', self.test_ui_updates)
        
        # API endpoints
        self.app.router.add_post('/api/start_training', self.start_training)
        self.app.router.add_post('/api/stop_training', self.stop_training)
        self.app.router.add_get('/api/models', self.get_models)
        self.app.router.add_post('/api/generate_sample', self.generate_sample)
        self.app.router.add_post('/api/upload_csv', self.upload_csv)
        self.app.router.add_get('/api/preview_csv', self.preview_csv)
        self.app.router.add_post('/api/test_training_complete', self.test_training_complete_event)
        
        # New endpoints for separate channels like test_separate_channels.py
        self.app.router.add_post('/training_data', self.receive_training_data)
        self.app.router.add_post('/progress_data', self.receive_progress_data)
        
        # SSE endpoints - separate channels
        self.app.router.add_get('/events/training', self.training_events)
        self.app.router.add_get('/events/progress', self.progress_events)
        self.app.router.add_get('/events/logs', self.log_events)
        
        # Debug and test pages
        self.app.router.add_get('/debug_sse_connection', self.debug_sse_connection)
        self.app.router.add_get('/minimal_sse_test', self.minimal_sse_test)
        self.app.router.add_get('/test_sse_connection_debug', self.test_sse_connection_debug)
        self.app.router.add_get('/test_main_dashboard_simple', self.test_main_dashboard_simple)
        self.app.router.add_get('/test_sse_connection_simple', self.test_sse_connection_simple)
        self.app.router.add_get('/debug_sse_connections', self.debug_sse_connections)
        self.app.router.add_get('/test_sse_simple', self.test_sse_simple)
        self.app.router.add_get('/test_status_fix', self.test_status_fix)
        self.app.router.add_get('/test_training_completion', self.test_training_completion)
    
    def start_background_tasks(self):
        """Start background monitoring tasks."""
        async def monitor_training():
            while True:
                # Only perform basic health checks, don't interfere with training completion logic
                # The actual training completion is handled by the monitor_output thread in start_training()
                if self.training_process and self.training_process.poll() is None:
                    # Process is still running - just wait
                    await asyncio.sleep(5)
                else:
                    # Process is not running, but don't automatically set to completed
                    # Let the training monitor handle completion status properly
                    await asyncio.sleep(5)
        
        asyncio.create_task(monitor_training())
        
        # Start log monitoring
        async def monitor_logs():
            while True:
                if self.log_monitoring_active:
                    await self.check_log_files()
                await asyncio.sleep(2)
        
        asyncio.create_task(monitor_logs())
    
    async def start_log_monitoring(self):
        """Start monitoring training log files for real-time updates."""
        self.log_monitoring_active = True
        
        while self.log_monitoring_active:
            try:
                # Find all log files
                log_patterns = [
                    "logs/*.log",
                    "logs/*.txt", 
                    "*.log",
                    "training_*.log"
                ]
                
                current_log_files = []
                for pattern in log_patterns:
                    current_log_files.extend(glob.glob(pattern))
                
                # Update log files list
                if current_log_files != self.log_files:
                    self.log_files = current_log_files
                    logger.info(f"Found log files: {self.log_files}")
                
                # Monitor each log file
                for log_file in self.log_files:
                    if os.path.exists(log_file):
                        await self.monitor_log_file(log_file)
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in log monitoring: {e}")
                await asyncio.sleep(5)
    
    async def monitor_log_file(self, log_file):
        """Monitor a single log file for new content."""
        try:
            if log_file not in self.last_log_positions:
                self.last_log_positions[log_file] = 0
            
            current_size = os.path.getsize(log_file)
            last_position = self.last_log_positions[log_file]
            
            if current_size > last_position:
                # Read new content
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_position)
                    new_content = f.read()
                    self.last_log_positions[log_file] = f.tell()
                
                # Parse new content for training metrics
                if new_content.strip():
                    await self.parse_log_content(new_content, log_file)
                    
        except Exception as e:
            logger.error(f"Error monitoring log file {log_file}: {e}")
    
    async def parse_log_content(self, content, log_file):
        """Parse log content for training metrics and progress."""
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse training metrics (similar to the original parsing logic)
            if "Epoch" in line and ("Generator Loss" in line or "generator_loss" in line):
                try:
                    # Try different log formats
                    metrics = self.extract_training_metrics(line)
                    if metrics:
                        await self.broadcast_training_update(metrics)
                except Exception as e:
                    logger.debug(f"Could not parse training line: {line}, error: {e}")
            
            # Parse progress information
            elif "progress" in line.lower() or "%" in line:
                try:
                    progress = self.extract_progress_info(line)
                    if progress:
                        await self.broadcast_progress_update(progress)
                except Exception as e:
                    logger.debug(f"Could not parse progress line: {line}, error: {e}")
            
            # Parse general log information
            elif any(keyword in line.lower() for keyword in ["error", "warning", "info", "debug"]):
                log_entry = {
                    "type": "log_entry",
                    "data": {
                        "message": line,
                        "source": log_file,
                        "timestamp": datetime.now().isoformat()
                    },
                    "timestamp": datetime.now().isoformat()
                }
                await self.broadcast_log_update(log_entry)
    
    def extract_training_metrics(self, line):
        """Extract training metrics from real GAN training logs."""
        try:
            # Pattern 1: Real GAN training metrics from trainer.py
            epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', line)
            gen_loss_match = re.search(r'Generator Loss:\s*([\d.]+)', line)
            disc_loss_match = re.search(r'Discriminator Loss:\s*([\d.]+)', line)
            real_scores_match = re.search(r'Real Scores:\s*([\d.]+)', line)
            fake_scores_match = re.search(r'Fake Scores:\s*([\d.]+)', line)
            
            if epoch_match and gen_loss_match and disc_loss_match:
                epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                gen_loss = float(gen_loss_match.group(1))
                disc_loss = float(disc_loss_match.group(1))
                
                # Extract real/fake scores if available
                real_scores = float(real_scores_match.group(1)) if real_scores_match else 0.8
                fake_scores = float(fake_scores_match.group(1)) if fake_scores_match else 0.2
                
                return {
                    "type": "training_update",
                    "data": {
                        "epoch": epoch,
                        "total_epochs": total_epochs,
                        "generator_loss": gen_loss,
                        "discriminator_loss": disc_loss,
                        "real_scores": real_scores,
                        "fake_scores": fake_scores
                    },
                    "timestamp": datetime.now().isoformat()
                }
            
            # Pattern 2: Dashboard channel data (already formatted)
            if "🎯 Training data sent to dashboard:" in line:
                # Extract metrics from dashboard log line
                metrics_match = re.search(r'Epoch (\d+), Gen Loss: ([\d.]+), Disc Loss: ([\d.]+)', line)
                if metrics_match:
                    epoch = int(metrics_match.group(1))
                    gen_loss = float(metrics_match.group(2))
                    disc_loss = float(metrics_match.group(3))
                    
                    return {
                        "type": "training_update",
                        "data": {
                            "epoch": epoch,
                            "total_epochs": 10,  # Default from config
                            "generator_loss": gen_loss,
                            "discriminator_loss": disc_loss,
                            "real_scores": 0.8,  # Default values
                            "fake_scores": 0.2
                        },
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Pattern 3: JSON-like format
            if "{" in line and "}" in line:
                try:
                    # Try to extract JSON from the line
                    json_start = line.find("{")
                    json_end = line.rfind("}") + 1
                    json_str = line[json_start:json_end]
                    data = json.loads(json_str)
                    
                    if "epoch" in data and "generator_loss" in data:
                        return {
                            "type": "training_update",
                            "data": data,
                            "timestamp": datetime.now().isoformat()
                        }
                except:
                    pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting training metrics: {e}")
            return None
    
    def extract_progress_info(self, line):
        """Extract progress information from real GAN training logs."""
        try:
            # Pattern 1: Dashboard progress data
            if "📊 Progress" in line and "sent to dashboard" in line:
                progress_match = re.search(r'Progress ([\d.]+)% sent to dashboard for epoch (\d+)', line)
                if progress_match:
                    progress_percent = float(progress_match.group(1))
                    epoch = int(progress_match.group(2))
                    
                    return {
                        "type": "progress",
                        "epoch": epoch,
                        "progress_percent": progress_percent,
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Pattern 2: Standard percentage patterns
            percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
            if percent_match:
                progress_percent = float(percent_match.group(1))
                
                # Try to extract epoch info
                epoch_match = re.search(r'epoch\s+(\d+)', line.lower())
                epoch = int(epoch_match.group(1)) if epoch_match else 1
                
                return {
                    "type": "progress",
                    "epoch": epoch,
                    "progress_percent": progress_percent,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Pattern 3: Epoch completion indicators
            if "Epoch" in line and "completed" in line:
                epoch_match = re.search(r'Epoch (\d+)', line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    return {
                        "type": "progress",
                        "epoch": epoch,
                        "progress_percent": 100.0,
                        "timestamp": datetime.now().isoformat()
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting progress info: {e}")
            return None
    
    def load_historical_logs(self):
        """Load and parse historical training logs."""
        historical_data = []
        
        for log_file in self.log_files:
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for line in lines:
                            metrics = self.extract_training_metrics(line)
                            if metrics:
                                historical_data.append(metrics)
                            
                            progress = self.extract_progress_info(line)
                            if progress:
                                historical_data.append(progress)
                                
                except Exception as e:
                    logger.error(f"Error reading historical log {log_file}: {e}")
        
        return historical_data
    
    async def test_sse_debug(self, request):
        """Test page for debugging SSE connections."""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SSE Debug Test</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .connected { background-color: #d4edda; color: #155724; }
                .disconnected { background-color: #f8d7da; color: #721c24; }
                .data { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
                pre { white-space: pre-wrap; word-wrap: break-word; }
            </style>
        </head>
        <body>
            <h1>SSE Connection Debug Test</h1>
            
            <div id="training-status" class="status disconnected">Training Channel: Disconnected</div>
            <div id="progress-status" class="status disconnected">Progress Channel: Disconnected</div>
            
            <h2>Training Data Received:</h2>
            <div id="training-data" class="data">No data yet...</div>
            
            <h2>Progress Data Received:</h2>
            <div id="progress-data" class="data">No data yet...</div>
            
            <h2>Console Log:</h2>
            <div id="console-log" class="data" style="height: 300px; overflow-y-auto; font-family: monospace; font-size: 12px;"></div>
            
            <script>
                let trainingEventSource = null;
                let progressEventSource = null;
                
                function log(message) {
                    const consoleLog = document.getElementById('console-log');
                    const timestamp = new Date().toLocaleTimeString();
                    consoleLog.innerHTML += `[${timestamp}] ${message}\\n`;
                    consoleLog.scrollTop = consoleLog.scrollHeight;
                    console.log(message);
                }
                
                function connectTrainingChannel() {
                    if (trainingEventSource) {
                        trainingEventSource.close();
                    }
                    
                    log('🎯 Connecting to training channel...');
                    trainingEventSource = new EventSource('/events/training');
                    
                    trainingEventSource.onopen = function() {
                        log('🎯 Training channel connected');
                        document.getElementById('training-status').textContent = 'Training Channel: Connected';
                        document.getElementById('training-status').className = 'status connected';
                    };
                    
                    trainingEventSource.onmessage = function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            log(`🎯 Training data received: ${JSON.stringify(data, null, 2)}`);
                            
                            if (data.type === 'training_update') {
                                // Update the main dashboard UI elements
                                updateDashboard(data);
                                
                                // Also update the training data display
                                document.getElementById('training-data').innerHTML = `
                                    <h3>Training Update:</h3>
                                    <pre>${JSON.stringify(data, null, 2)}</pre>
                                `;
                            } else if (data.type === 'connection') {
                                log(`🎯 Training channel info: ${data.message}`);
                            }
                        } catch (error) {
                            log(`🎯 Error parsing training data: ${error}`);
                        }
                    };
                    
                    trainingEventSource.onerror = function(error) {
                        log(`🎯 Training channel error: ${error}`);
                        document.getElementById('training-status').textContent = 'Training Channel: Error';
                        document.getElementById('training-status').className = 'status disconnected';
                    };
                }
                
                function connectProgressChannel() {
                    if (progressEventSource) {
                        progressEventSource.close();
                    }
                    
                    log('📊 Connecting to progress channel...');
                    progressEventSource = new EventSource('/events/progress');
                    
                    progressEventSource.onopen = function() {
                        log('📊 Progress channel connected');
                        document.getElementById('progress-status').textContent = 'Progress Channel: Connected';
                        document.getElementById('progress-status').className = 'status connected';
                    };
                    
                    progressEventSource.onmessage = function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            log(`📊 Progress data received: ${JSON.stringify(data, null, 2)}`);
                            
                            if (data.type === 'progress') {
                                document.getElementById('progress-data').innerHTML = `
                                    <h3>Progress Update:</h3>
                                    <pre>${JSON.stringify(data, null, 2)}</pre>
                                `;
                            } else if (data.type === 'connection') {
                                log(`📊 Progress channel info: ${data.message}`);
                            }
                        } catch (error) {
                            log(`📊 Error parsing progress data: ${error}`);
                        }
                    };
                    
                    progressEventSource.onerror = function(error) {
                        log(`📊 Progress channel error: ${error}`);
                        document.getElementById('progress-status').textContent = 'Progress Channel: Error';
                        document.getElementById('progress-status').className = 'status disconnected';
                    };
                }
                
                // Connect on page load
                window.addEventListener('load', function() {
                    log('🚀 Page loaded, connecting to SSE channels...');
                    connectTrainingChannel();
                    connectProgressChannel();
                });
                
                // Cleanup on page unload
                window.addEventListener('beforeunload', function() {
                    if (trainingEventSource) {
                        trainingEventSource.close();
                    }
                    if (progressEventSource) {
                        progressEventSource.close();
                    }
                });
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def test_minimal_sse(self, request):
        """Minimal SSE test page."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Minimal SSE Test</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .connected { background-color: #d4edda; color: #155724; }
                .disconnected { background-color: #f8d7da; color: #721c24; }
                .data { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Minimal SSE Test</h1>
            
            <div id="training-status" class="status disconnected">Training: Disconnected</div>
            <div id="progress-status" class="status disconnected">Progress: Disconnected</div>
            
            <div id="training-data" class="data">No training data</div>
            <div id="progress-data" class="data">No progress data</div>
            
            <div id="console-log" style="height: 200px; overflow-y: auto; background: #f0f0f0; padding: 10px; font-family: monospace; font-size: 12px;"></div>
            
            <script>
                function log(message) {
                    const consoleLog = document.getElementById('console-log');
                    const timestamp = new Date().toLocaleTimeString();
                    consoleLog.innerHTML += `[${timestamp}] ${message}<br>`;
                    consoleLog.scrollTop = consoleLog.scrollHeight;
                    console.log(message);
                }
                
                // Test SSE connection
                log('🚀 Starting SSE test...');
                
                // Training channel
                const trainingEventSource = new EventSource('/events/training');
                
                trainingEventSource.onopen = function() {
                    log('🎯 Training channel connected');
                    document.getElementById('training-status').textContent = 'Training: Connected';
                    document.getElementById('training-status').className = 'status connected';
                };
                
                trainingEventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        log(`🎯 Training data: ${JSON.stringify(data)}`);
                        
                        if (data.type === 'training_update') {
                            document.getElementById('training-data').innerHTML = `
                                <h3>Training Update:</h3>
                                <p>Epoch: ${data.data.epoch}</p>
                                <p>Gen Loss: ${data.data.generator_loss}</p>
                                <p>Disc Loss: ${data.data.discriminator_loss}</p>
                            </p>
                        }
                    } catch (error) {
                        log(`🎯 Error: ${error}`);
                    }
                };
                
                trainingEventSource.onerror = function(error) {
                    log(`🎯 Training error: ${error}`);
                    document.getElementById('training-status').textContent = 'Training: Error';
                    document.getElementById('training-status').className = 'status disconnected';
                };
                
                // Progress channel
                const progressEventSource = new EventSource('/events/progress');
                
                progressEventSource.onopen = function() {
                    log('📊 Progress channel connected');
                    document.getElementById('progress-status').textContent = 'Progress: Connected';
                    document.getElementById('progress-status').className = 'status connected';
                };
                
                progressEventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        log(`📊 Progress data: ${JSON.stringify(data)}`);
                        
                        if (data.type === 'progress') {
                            document.getElementById('progress-data').innerHTML = `
                                <h3>Progress Update:</h3>
                                <p>Epoch: ${data.epoch}</p>
                                <p>Progress: ${data.progress_percent}%</p>
                            </p>
                        }
                    } catch (error) {
                        log(`📊 Progress error: ${error}`);
                    }
                };
                
                progressEventSource.onerror = function(error) {
                    log(`📊 Progress error: ${error}`);
                    document.getElementById('progress-status').textContent = 'Progress: Error';
                    document.getElementById('progress-status').className = 'status disconnected';
                };
                
                log('✅ SSE channels initialized');
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def test_main_dashboard_sse(self, request):
        """Test page that simulates the main dashboard's SSE functionality."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Main Dashboard SSE Test</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .connected { background-color: #d4edda; color: #155724; }
                .disconnected { background-color: #f8d7da; color: #721c24; }
                .data { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body>
            <h1>Main Dashboard SSE Test</h1>
            
            <!-- Training Status Cards -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div class="card">
                    <div class="flex items-center">
                        <div class="p-2 rounded-full bg-blue-100 text-blue-600">📊</div>
                        <div class="ml-3">
                            <h3 class="text-sm font-semibold text-gray-700">Training Status</h3>
                            <p id="status-text" class="text-xl font-bold text-blue-600">Idle</p>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="flex items-center">
                        <div class="p-2 rounded-full bg-green-100 text-green-600">📈</div>
                        <div class="ml-3">
                            <h3 class="text-sm font-semibold text-gray-700">Generator Loss</h3>
                            <p id="gen-loss" class="text-xl font-bold text-green-600">-</p>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="flex items-center">
                        <div class="p-2 rounded-full bg-red-100 text-red-600">📉</div>
                        <div class="ml-3">
                            <h3 class="text-sm font-semibold text-gray-700">Discriminator Loss</h3>
                            <p id="disc-loss" class="text-xl font-bold text-red-600">-</p>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="flex items-center">
                        <div class="p-2 rounded-full bg-purple-100 text-purple-600">⏰</div>
                        <div class="ml-3">
                            <h3 class="text-sm font-semibold text-gray-700">Epoch</h3>
                            <p id="current-epoch" class="text-xl font-bold text-purple-600">-</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="console-log" style="height: 300px; overflow-y: auto; background: #f0f0f0; padding: 10px; font-family: monospace; font-size: 12px;"></div>
            
            <script>
                function log(message) {
                    const consoleLog = document.getElementById('console-log');
                    const timestamp = new Date().toLocaleTimeString();
                    consoleLog.innerHTML += `[${timestamp}] ${message}<br>`;
                    consoleLog.scrollTop = consoleLog.scrollHeight;
                    console.log(message);
                }
                
                // Simulate the main dashboard's SSE functionality
                let trainingEventSource = null;
                let progressEventSource = null;
                
                function updateDashboard(data) {
                    log('🎯 Updating dashboard with: ' + JSON.stringify(data));
                    
                    if (data.type === 'training_update') {
                        log('🎯 Processing training update: ' + JSON.stringify(data.data));
                        
                        // Update training status to Running when we receive training data
                        const statusElement = document.getElementById('status-text');
                        log('🎯 Looking for status-text element: ' + statusElement);
                        if (statusElement) {
                            statusElement.textContent = 'Running';
                            statusElement.className = 'text-xl font-bold text-green-600';
                            log('✅ Updated training status to Running');
                        } else {
                            log('❌ status-text element not found');
                        }
                        
                        // Update status cards
                        const genLossElement = document.getElementById('gen-loss');
                        const discLossElement = document.getElementById('disc-loss');
                        const currentEpochElement = document.getElementById('current-epoch');
                        
                        log('🎯 Found elements: ' + JSON.stringify({
                            genLoss: genLossElement,
                            discLoss: discLossElement,
                            currentEpoch: currentEpochElement
                        }));
                        
                        if (genLossElement) {
                            genLossElement.textContent = data.data.generator_loss.toFixed(4);
                            log('✅ Updated generator loss: ' + data.data.generator_loss);
                        } else {
                            log('❌ gen-loss element not found');
                        }
                        
                        if (discLossElement) {
                            discLossElement.textContent = data.data.discriminator_loss.toFixed(4);
                            log('✅ Updated discriminator loss: ' + data.data.discriminator_loss);
                        } else {
                            log('❌ disc-loss element not found');
                        }
                        
                        if (currentEpochElement) {
                            currentEpochElement.textContent = data.data.epoch;
                            log('✅ Updated current epoch: ' + data.data.epoch);
                        } else {
                            log('❌ current-epoch element not found');
                        }
                    }
                }
                
                function updateProgress(data) {
                    log('📊 Progress update: ' + JSON.stringify(data));
                    
                    if (data.type === 'progress') {
                        log('📊 Processing progress: ' + data.progress_percent + '% for epoch ' + data.epoch);
                    }
                }
                
                function connectTrainingChannel() {
                    if (trainingEventSource) {
                        trainingEventSource.close();
                    }
                    
                    log('🎯 Connecting to training channel...');
                    trainingEventSource = new EventSource('/events/training');
                    
                    trainingEventSource.onopen = function() {
                        log('🎯 Connected to Training SSE Channel');
                    };
                    
                    trainingEventSource.onmessage = function(event) {
                        try {
                            log('🎯 Raw training event received: ' + event.data);
                            const data = JSON.parse(event.data);
                            log('🎯 Training data received: ' + JSON.stringify(data));
                            
                            if (data.type === 'training_update') {
                                log('🎯 Updating dashboard with training data: ' + JSON.stringify(data));
                                updateDashboard(data);
                            } else if (data.type === 'connection') {
                                log('🎯 Training channel connected: ' + data.message);
                            } else {
                                log('🎯 Unknown training data type: ' + data.type);
                            }
                        } catch (error) {
                            log('🎯 Error parsing training data: ' + error);
                        }
                    };
                    
                    trainingEventSource.onerror = function(error) {
                        log('🎯 Training channel connection error: ' + error);
                    };
                }
                
                function connectProgressChannel() {
                    if (progressEventSource) {
                        progressEventSource.close();
                    }
                    
                    log('📊 Connecting to progress channel...');
                    progressEventSource = new EventSource('/events/progress');
                    
                    progressEventSource.onopen = function() {
                        log('📊 Connected to Progress SSE Channel');
                    };
                    
                    progressEventSource.onmessage = function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            log('📊 Progress data received: ' + JSON.stringify(data));
                            
                            if (data.type === 'progress') {
                                log('📊 Updating progress with data: ' + JSON.stringify(data));
                                updateProgress(data);
                            } else if (data.type === 'connection') {
                                log('📊 Progress channel connected: ' + data.message);
                            } else {
                                log('📊 Unknown progress data type: ' + data.type);
                            }
                        } catch (error) {
                            log('📊 Progress channel connection error: ' + error);
                        }
                    };
                    
                    progressEventSource.onerror = function(error) {
                        log('📊 Progress channel connection error: ' + error);
                    };
                }
                
                // Connect on page load
                window.addEventListener('load', function() {
                    log('🚀 Page loaded, setting up SSE channels');
                    connectTrainingChannel();
                    connectProgressChannel();
                });
                
                log('✅ Script loaded');
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def test_ui_updates(self, request):
        """Test page for monitoring dashboard UI updates."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard UI Updates Test</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .connected { background-color: #d4edda; color: #155724; }
                .disconnected { background-color: #f8d7da; color: #721c24; }
                .data { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
                .metric { text-align: center; }
                .metric-value { font-size: 2em; font-weight: bold; margin: 10px 0; }
                .gen-loss { color: #28a745; }
                .disc-loss { color: #dc3545; }
                .epoch { color: #6f42c1; }
                .progress { color: #007bff; }
            </style>
        </head>
        <body>
            <h1>Dashboard UI Updates Test</h1>
            
            <!-- Connection Status -->
            <div class="grid">
                <div class="card">
                    <h3>Training Channel</h3>
                    <div id="training-status" class="status disconnected">Disconnected</div>
                    <div id="training-clients">Clients: 0</div>
                </div>
                
                <div class="card">
                    <h3>Progress Channel</h3>
                    <div id="progress-status" class="status disconnected">Disconnected</div>
                    <div id="progress-clients">Clients: 0</div>
                </div>
            </div>
            
            <!-- Training Metrics -->
            <div class="grid">
                <div class="card metric">
                    <h3>Current Epoch</h3>
                    <div id="current-epoch" class="metric-value epoch">-</div>
                </div>
                
                <div class="card metric">
                    <h3>Generator Loss</h3>
                    <div id="gen-loss" class="metric-value gen-loss">-</div>
                </div>
                
                <div class="card metric">
                    <h3>Discriminator Loss</h3>
                    <div id="disc-loss" class="metric-value disc-loss">-</div>
                </div>
                
                <div class="card metric">
                    <h3>Progress</h3>
                    <div id="progress-percent" class="metric-value progress">-</div>
                </div>
            </div>
            
            <!-- Raw Data Display -->
            <div class="card">
                <h3>Latest Training Update</h3>
                <div id="training-data" class="data">No training data received</div>
            </div>
            
            <div class="card">
                <h3>Latest Progress Update</h3>
                <div id="progress-data" class="data">No progress data received</div>
            </div>
            
            <!-- Console Log -->
            <div class="card">
                <h3>Console Log</h3>
                <div id="console-log" style="height: 300px; overflow-y: auto; background: #f0f0f0; padding: 10px; font-family: monospace; font-size: 12px;"></div>
            </div>
            
            <!-- Test Buttons -->
            <div class="card">
                <h3>Test Actions</h3>
                <button onclick="testTrainingData()">Send Test Training Data</button>
                <button onclick="testProgressData()">Send Test Progress Data</button>
                <button onclick="runTrainingSimulation()">Run Training Simulation</button>
            </div>
            
            <script>
                let trainingEventSource = null;
                let progressEventSource = null;
                
                function log(message) {
                    const consoleLog = document.getElementById('console-log');
                    const timestamp = new Date().toLocaleTimeString();
                    consoleLog.innerHTML += `[${timestamp}] ${message}<br>`;
                    consoleLog.scrollTop = consoleLog.scrollHeight;
                    console.log(message);
                }
                
                function connectTrainingChannel() {
                    if (trainingEventSource) {
                        trainingEventSource.close();
                    }
                    
                    log('🎯 Connecting to training channel...');
                    trainingEventSource = new EventSource('/events/training');
                    
                    trainingEventSource.onopen = function() {
                        log('🎯 Training channel connected');
                        document.getElementById('training-status').textContent = 'Connected';
                        document.getElementById('training-status').className = 'status connected';
                    };
                    
                    trainingEventSource.onmessage = function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            log(`🎯 Training data received: ${JSON.stringify(data, null, 2)}`);
                            
                            if (data.type === 'training_update') {
                                // Update UI metrics
                                document.getElementById('current-epoch').textContent = data.data.epoch;
                                document.getElementById('gen-loss').textContent = data.data.generator_loss.toFixed(4);
                                document.getElementById('disc-loss').textContent = data.data.discriminator_loss.toFixed(4);
                                
                                // Update raw data display
                                document.getElementById('training-data').innerHTML = `
                                    <h4>Training Update:</h4>
                                    <p><strong>Epoch:</strong> ${data.data.epoch}/${data.data.total_epochs}</p>
                                    <p><strong>Generator Loss:</strong> ${data.data.generator_loss.toFixed(4)}</p>
                                    <p><strong>Discriminator Loss:</strong> ${data.data.discriminator_loss.toFixed(4)}</p>
                                    <p><strong>Real Scores:</strong> ${data.data.real_scores.toFixed(4)}</p>
                                    <p><strong>Fake Scores:</strong> ${data.data.fake_scores.toFixed(4)}</p>
                                    <p><strong>Timestamp:</strong> ${data.timestamp}</p>
                                `;
                            } else if (data.type === 'connection') {
                                document.getElementById('training-clients').textContent = `Clients: ${data.client_id || 'Unknown'}`;
                            }
                        } catch (error) {
                            log(`🎯 Error parsing training data: ${error}`);
                        }
                    };
                    
                    trainingEventSource.onerror = function(error) {
                        log(`🎯 Training channel error: ${error}`);
                        document.getElementById('training-status').textContent = 'Error';
                        document.getElementById('training-status').className = 'status disconnected';
                    };
                }
                
                function connectProgressChannel() {
                    if (progressEventSource) {
                        progressEventSource.close();
                    }
                    
                    log('📊 Connecting to progress channel...');
                    progressEventSource = new EventSource('/events/progress');
                    
                    progressEventSource.onopen = function() {
                        log('📊 Progress channel connected');
                        document.getElementById('progress-status').textContent = 'Connected';
                        document.getElementById('progress-status').className = 'status connected';
                    };
                    
                    progressEventSource.onmessage = function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            log(`📊 Progress data received: ${JSON.stringify(data, null, 2)}`);
                            
                            if (data.type === 'progress') {
                                // Update the main dashboard progress element
                                const progressElement = document.getElementById('progress-percent');
                                if (progressElement) {
                                    progressElement.textContent = data.progress_percent + '%';
                                    log(`✅ Updated progress to ${data.progress_percent}%`);
                                } else {
                                    log(`❌ progress-percent element not found`);
                                }
                                
                                // Update raw data display
                                document.getElementById('progress-data').innerHTML = `
                                    <h4>Progress Update:</h4>
                                    <p><strong>Epoch:</strong> ${data.epoch}</p>
                                    <p><strong>Progress:</strong> ${data.progress_percent}%</p>
                                    <p><strong>Timestamp:</strong> ${data.timestamp}</p>
                                `;
                            } else if (data.type === 'connection') {
                                document.getElementById('progress-clients').textContent = `Clients: ${data.client_count || 'Unknown'}`;
                            }
                        } catch (error) {
                            log(`📊 Progress channel error: ${error}`);
                        }
                    };
                    
                    progressEventSource.onerror = function(error) {
                        log(`📊 Progress channel error: ${error}`);
                        document.getElementById('progress-status').textContent = 'Error';
                        document.getElementById('progress-status').className = 'status disconnected';
                    };
                }
                
                function testTrainingData() {
                    log('🧪 Sending test training data...');
                    fetch('/training_data', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            type: 'training_update',
                            data: {
                                epoch: 1,
                                total_epochs: 10,
                                generator_loss: 0.75,
                                discriminator_loss: 0.65,
                                real_scores: 0.85,
                                fake_scores: 0.25
                            },
                            timestamp: new Date().toISOString()
                        })
                    })
                    .then(response => response.json())
                    .then(data => log(`✅ Test training data sent: ${JSON.stringify(data)}`))
                    .catch(error => log(`❌ Error sending test training data: ${error}`));
                }
                
                function testProgressData() {
                    log('🧪 Sending test progress data...');
                    fetch('/progress_data', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            type: 'progress',
                            epoch: 1,
                            progress_percent: 75,
                            timestamp: new Date().toISOString()
                        })
                    })
                    .then(response => response.json())
                    .then(data => log(`✅ Test progress data sent: ${JSON.stringify(data)}`))
                    .catch(error => log(`❌ Error sending test progress data: ${error}`));
                }
                
                function runTrainingSimulation() {
                    log('🚀 Starting training simulation...');
                    
                    // Simulate training over 5 epochs
                    for (let epoch = 1; epoch <= 5; epoch++) {
                        setTimeout(() => {
                            // Send progress updates
                            for (let progress = 0; progress <= 100; progress += 25) {
                                setTimeout(() => {
                                    fetch('/progress_data', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({
                                            type: 'progress',
                                            epoch: epoch,
                                            progress_percent: progress,
                                            timestamp: new Date().toISOString()
                                        })
                                    });
                                }, progress * 10);
                            }
                            
                            // Send training metrics after progress completes
                            setTimeout(() => {
                                const genLoss = 0.8 - epoch * 0.05 + (epoch % 3) * 0.02;
                                const discLoss = 0.7 - epoch * 0.03 + (epoch % 2) * 0.01;
                                
                                fetch('/training_data', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({
                                        type: 'training_update',
                                        data: {
                                            epoch: epoch,
                                            total_epochs: 5,
                                            generator_loss: Math.max(0.1, genLoss),
                                            discriminator_loss: Math.max(0.1, discLoss),
                                            real_scores: 0.9 - epoch * 0.01,
                                            fake_scores: 0.1 + epoch * 0.02
                                        },
                                        timestamp: new Date().toISOString()
                                    })
                                });
                            }, 1000);
                        }, epoch * 2000);
                    }
                    
                    log('✅ Training simulation scheduled');
                }
                
                // Connect on page load
                window.addEventListener('load', function() {
                    log('🚀 Page loaded, connecting to SSE channels...');
                    connectTrainingChannel();
                    connectProgressChannel();
                });
                
                // Cleanup on page unload
                window.addEventListener('beforeunload', function() {
                    if (trainingEventSource) {
                        trainingEventSource.close();
                    }
                    if (progressEventSource) {
                        progressEventSource.close();
                    }
                });
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def dashboard(self, request):
        """Main dashboard page."""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Treasury GAN Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://unpkg.com/feather-icons"></script>
            <style>
                /* Enhanced button state styling */
                .training-btn {
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }
                
                .training-btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                    transform: scale(0.95);
                }
                
                .training-btn:not(:disabled):hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                }
                
                .training-btn:not(:disabled):active {
                    transform: translateY(0);
                }
                
                /* Completion notification animation */
                .completion-notification {
                    animation: slideInRight 0.5s ease-out;
                }
                
                @keyframes slideInRight {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
                
                /* Progress bar enhancement */
                #training-progress-bar {
                    transition: width 0.5s ease-out;
                    background: linear-gradient(90deg, #3b82f6, #1d4ed8);
                }
                
                /* Status text enhancement */
                #status-text {
                    transition: all 0.3s ease;
                }
                
                /* Comprehensive data option enhancement */
                #comprehensive-option {
                    background: linear-gradient(135deg, #dbeafe 0%, #e9d5ff 100%);
                    border: 2px solid #3b82f6;
                    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15);
                }
                
                #comprehensive-option:hover {
                    background: linear-gradient(135deg, #bfdbfe 0%, #ddd6fe 100%);
                    border-color: #2563eb;
                    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
                    transform: translateY(-1px);
                }
                
                /* Enhanced table styling for treasury data */
                .treasury-table th {
                    background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
                    border-bottom: 2px solid #d1d5db;
                }
                
                .treasury-table td {
                    border-bottom: 1px solid #f3f4f6;
                }
                
                .treasury-table tr:hover {
                    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                }
            </style>
        </head>
        <body class="bg-gray-100">
            <nav class="bg-blue-600 text-white shadow-lg">
                <div class="max-w-7xl mx-auto px-4">
                    <div class="flex justify-between items-center py-4">
                        <h1 class="text-2xl font-bold">🏦 Treasury GAN Dashboard</h1>
                        <div class="flex space-x-4">
                            <a href="/" class="hover:text-blue-200">Dashboard</a>
                            <a href="/training" class="hover:text-blue-200">Training</a>
                            <a href="/evaluation" class="hover:text-blue-200">Evaluation</a>
                            <a href="/models" class="hover:text-blue-200">Models</a>
                            <a href="/test_sse_debug" class="hover:text-blue-200">SSE Debug</a>
                        </div>
                    </div>
                </div>
            </nav>
            
            <div class="max-w-7xl mx-auto px-4 py-8">
                <!-- Quick Actions and Training Data Preview - Side by Side -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                    <!-- Quick Actions Section - Left Side -->
                    <div class="bg-white rounded-lg shadow-md p-4">
                        <h3 class="text-lg font-semibold text-gray-700 mb-3">Quick Actions</h3>
                        
                        <!-- Data Source Selection -->
                        <div class="mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
                            <h4 class="text-md font-medium text-gray-700 mb-2">📊 Select Training Data Source</h4>
                            
                            <!-- Available Data Sources - Compact Grid -->
                            <div class="mb-3">
                                <div class="grid grid-cols-2 gap-2">
                                    <div class="p-2 bg-white rounded border border-gray-200 hover:border-blue-300 cursor-pointer transition-colors text-center" onclick="window.selectDataSource('treasury_orderbook_sample.csv', 'orderbook')">
                                        <div class="w-2 h-2 bg-blue-500 rounded-full mx-auto mb-1"></div>
                                        <span class="text-xs font-medium">Treasury Orderbook</span>
                                    </div>
                                    <div class="p-2 bg-white rounded border border-gray-200 hover:border-blue-300 cursor-pointer transition-colors text-center" onclick="window.selectDataSource('sample_timeseries.csv', 'timeseries')">
                                        <div class="w-2 h-2 bg-green-500 rounded-full mx-auto mb-1"></div>
                                        <span class="text-xs font-medium">Yield Curve</span>
                                    </div>
                                </div>
                                <div class="mt-2">
                                    <div class="p-2 bg-gradient-to-r from-gray-50 to-gray-100 rounded border border-gray-300 cursor-not-allowed transition-all text-center opacity-60" id="comprehensive-option">
                                        <div class="w-3 h-3 bg-gradient-to-r from-gray-400 to-gray-500 rounded-full mx-auto mb-1"></div>
                                        <span class="text-xs font-medium text-gray-500">🎯 Treasury Orderbook Data</span>
                                        <div class="text-xs text-gray-400 mt-1">Click "Generate Sample" to create 5-level orderbook</div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Custom Upload - Compact Row -->
                            <div class="flex items-center space-x-3">
                                <div class="flex-1">
                                    <input type="file" id="csv-file-input" accept=".csv" class="hidden" />
                                    <button id="upload-csv" class="bg-purple-600 text-white px-4 py-2 rounded text-sm hover:bg-purple-700 transition-colors">
                                        📁 Upload CSV
                                    </button>
                                    <span id="selected-file-name" class="ml-2 text-xs text-gray-600"></span>
                                </div>
                                <button id="generate-sample" class="bg-green-600 text-white px-4 py-2 rounded text-sm hover:bg-green-700 transition-colors">
                                    🔄 Generate Sample
                                </button>
                            </div>
                            
                            <!-- Selected Data Preview - Compact -->
                            <div id="selected-data-preview" class="hidden mt-2">
                                <div id="data-preview-content" class="bg-white rounded border border-gray-200 p-2 text-xs">
                                    <!-- Data preview content will be populated here -->
                                </div>
                            </div>
                        </div>
                        
                        <!-- Training Controls - Compact -->
                        <div class="p-3 bg-blue-50 rounded-lg border border-blue-200">
                            <h4 class="text-md font-medium text-blue-700 mb-2">🚀 Training Controls</h4>
                            <div class="flex items-center justify-between">
                                <div class="flex items-center space-x-3">
                                    <div class="flex items-center space-x-2">
                                        <label for="epochs-input" class="text-sm font-medium text-blue-700">Epochs:</label>
                                        <input type="number" id="epochs-input" value="5" min="1" max="1000" 
                                               class="w-16 px-2 py-1 text-sm border border-blue-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                                    </div>
                                    <button id="start-training" class="training-btn bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 font-medium text-sm">
                                        ▶️ Start Training
                                    </button>
                                    <button id="stop-training" class="training-btn bg-red-600 text-white px-6 py-2 rounded hover:bg-red-700 font-medium text-sm" disabled>
                                        ⏹️ Stop Training
                                    </button>
                                </div>
                                <div id="data-source-indicator" class="text-xs text-gray-600">
                                    <span class="font-medium">Status:</span> 
                                    <span id="data-source-status">No data source selected</span>
                                    <div id="training-status-indicator" class="hidden mt-1">
                                        <div class="flex items-center space-x-2">
                                            <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                                            <span class="text-green-600 font-medium">Training Active</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Sample Data Section - Right Side -->
                    <div id="data-preview-section" class="bg-white rounded-lg shadow-md p-4" style="display: none;">
                        <h3 class="text-lg font-semibold text-gray-700 mb-3">📊 Training Data Preview</h3>
                        
                        <!-- Sample Data Table -->
                        <div>
                            <h4 class="text-md font-medium text-gray-700 mb-2">Sample Data (First 5 rows)</h4>
                            <div class="overflow-x-auto">
                                <div id="data-table-preview" class="bg-gray-50 p-3 rounded-lg">
                                    <div class="text-center text-gray-500 py-8">
                                        <div class="text-2xl mb-2">📊</div>
                                        <div class="text-sm">No data selected</div>
                                        <div class="text-xs text-gray-400 mt-1">Upload a CSV file or generate sample data to preview</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Time Series Visualization - Full Width Below Both Sections -->
                <div class="mb-6">
                    <div class="bg-white rounded-lg shadow-md p-4">
                        <h3 class="text-lg font-semibold text-gray-700 mb-3">📈 Time Series Visualization</h3>
                        <div class="bg-gray-50 p-4 rounded-lg">
                            <canvas id="dataPreviewChart" width="800" height="300"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- Training Status Cards -->
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                    <div class="bg-white rounded-lg shadow-md p-4">
                        <div class="flex items-center">
                            <div class="p-2 rounded-full bg-blue-100 text-blue-600">
                                <i data-feather="activity" class="w-5 h-5"></i>
                            </div>
                            <div class="ml-3">
                                <h3 class="text-sm font-semibold text-gray-700">Training Status</h3>
                                <p id="status-text" class="text-xl font-bold text-blue-600">Idle</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-4">
                        <div class="flex items-center">
                            <div class="p-2 rounded-full bg-green-100 text-green-600">
                                <i data-feather="trending-up" class="w-5 h-5"></i>
                            </div>
                            <div class="ml-3">
                                <h3 class="text-sm font-semibold text-gray-700">Generator Loss</h3>
                                <p id="gen-loss" class="text-xl font-bold text-green-600">-</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-4">
                        <div class="flex items-center">
                            <div class="p-2 rounded-full bg-red-100 text-red-600">
                                <i data-feather="trending-down" class="w-5 h-5"></i>
                            </div>
                            <div class="ml-3">
                                <h3 class="text-sm font-semibold text-gray-700">Discriminator Loss</h3>
                                <p id="disc-loss" class="text-xl font-bold text-red-600">-</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-4">
                        <div class="flex items-center">
                            <div class="p-2 rounded-full bg-purple-100 text-purple-600">
                                <i data-feather="clock" class="w-5 h-5"></i>
                            </div>
                            <div class="ml-3">
                                <h3 class="text-sm font-semibold text-gray-700">Training Progress</h3>
                                <p id="training-progress-display" class="text-xl font-bold text-purple-600">0.0%</p>
                                <p id="training-epoch-display" class="text-sm text-gray-600">Epoch -</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Training Charts -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                    <div class="bg-white rounded-lg shadow-md p-4">
                        <h3 class="text-lg font-semibold text-gray-700 mb-3">Training Progress</h3>
                        <canvas id="trainingChart" width="400" height="200"></canvas>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-4">
                        <h3 class="text-lg font-semibold text-gray-700 mb-3">Real vs Synthetic Scores</h3>
                        <canvas id="scoresChart" width="400" height="200"></canvas>
                    </div>
                </div>
                


            </div>
            
            <script>
                // Initialize charts
                const trainingCtx = document.getElementById('trainingChart').getContext('2d');
                const scoresCtx = document.getElementById('scoresChart').getContext('2d');
                
                // Initialize chart reference as null to prevent destroy() errors
                window.dataPreviewChart = null;
                
                // Helper function to safely destroy chart
                function safeDestroyChart(chart) {
                    if (chart && typeof chart.destroy === 'function') {
                        try {
                            chart.destroy();
                            return true;
                        } catch (error) {
                            console.error('Error destroying chart:', error);
                            return false;
                        }
                    }
                    return false;
                }
                
                // Initialize data preview section
                function initializeDataPreview() {
                    console.log('🔍 initializeDataPreview: Starting...');
                    const dataPreviewSection = document.getElementById('data-preview-section');
                    if (dataPreviewSection) {
                        // Ensure the section is hidden initially
                        dataPreviewSection.style.display = 'none';
                        console.log('🔍 initializeDataPreview: Data preview section hidden');
                        
                        // Initialize chart container
                        const chartElement = document.getElementById('dataPreviewChart');
                        
                        if (chartElement) {
                            console.log('🔍 initializeDataPreview: Data preview chart element found and ready');
                        } else {
                            console.error('🔍 initializeDataPreview: Data preview chart element not found during initialization');
                        }
                    } else {
                        console.error('🔍 initializeDataPreview: Data preview section element not found');
                    }
                    console.log('🔍 initializeDataPreview: Completed');
                }
                
                // Helper function to safely clear chart area
                function clearChartArea() {
                    const chartElement = document.getElementById('dataPreviewChart');
                    if (chartElement && chartElement.parentElement) {
                        const chartContainer = chartElement.parentElement;
                        chartContainer.innerHTML = `
                            <canvas id="dataPreviewChart" width="800" height="300"></canvas>
                        `;
                        // Re-initialize the chart element reference
                        window.dataPreviewChart = null;
                    }
                }
                
                // Helper function to show loading state
                function showChartLoading() {
                    const chartElement = document.getElementById('dataPreviewChart');
                    if (chartElement && chartElement.parentElement) {
                        const chartContainer = chartElement.parentElement;
                        chartContainer.innerHTML = `
                            <div class="text-center text-blue-500 py-8">
                                <div class="text-2xl mb-2">⏳</div>
                                <div class="text-sm">Loading chart...</div>
                            </div>
                        `;
                    }
                }
                
                // Helper function to safely update chart
                function safeUpdateChart(plotData) {
                    try {
                        const chartElement = document.getElementById('dataPreviewChart');
                        
                        if (!chartElement) {
                            console.error('Chart element not found for update');
                            return false;
                        }
                        
                        // Use validation helper function
                        const validation = validateChartData(plotData);
                        return validation.valid;
                    } catch (error) {
                        console.error('Error in safeUpdateChart:', error);
                        return false;
                    }
                }
                
                // Helper function to safely handle chart errors
                function handleChartError(error, context = 'chart operation') {
                    console.error(`Error in ${context}:`, error);
                    
                    const chartElement = document.getElementById('dataPreviewChart');
                    if (chartElement && chartElement.parentElement) {
                        const chartContainer = chartElement.parentElement;
                        chartContainer.innerHTML = `
                            <div class="text-center text-red-500 py-8">
                                <div class="text-2xl mb-2">❌</div>
                                <div class="text-sm">Chart error occurred</div>
                                <div class="text-xs text-gray-500 mt-1">${error.message || 'Unknown error'}</div>
                                <button onclick="clearChartArea()" class="mt-2 px-3 py-1 bg-red-100 text-red-700 rounded text-xs hover:bg-red-200">
                                    Reset Chart
                                </button>
                            </div>
                        `;
                    }
                    
                    // Reset chart reference
                    window.dataPreviewChart = null;
                }
                
                // Helper function to safely validate chart data
                function validateChartData(plotData) {
                    if (!plotData || typeof plotData !== 'object') {
                        return { valid: false, error: 'Invalid data format' };
                    }
                    
                    if (Object.keys(plotData).length === 0) {
                        return { valid: false, error: 'Empty data object' };
                    }
                    
                    const validColumns = Object.keys(plotData).filter(key => 
                        key !== 'index' && plotData[key] && Array.isArray(plotData[key])
                    );
                    
                    if (validColumns.length === 0) {
                        return { valid: false, error: 'No numeric data available' };
                    }
                    
                    return { valid: true, columns: validColumns };
                }
                
                window.trainingChart = new Chart(trainingCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Generator Loss',
                            data: [],
                            borderColor: 'rgb(59, 130, 246)',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            tension: 0.1
                        }, {
                            label: 'Discriminator Loss',
                            data: [],
                            borderColor: 'rgb(239, 68, 68)',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
                
                window.scoresChart = new Chart(scoresCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Real Scores',
                            data: [],
                            borderColor: 'rgb(34, 197, 94)',
                            backgroundColor: 'rgba(34, 197, 94, 0.1)',
                            tension: 0.1
                        }, {
                            label: 'Fake Scores',
                            data: [],
                            borderColor: 'rgb(168, 85, 247)',
                            backgroundColor: 'rgba(168, 85, 247, 0.1)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1
                            }
                        }
                    }
                });
                
                // Initialize data preview when page loads
                document.addEventListener('DOMContentLoaded', function() {
                    console.log('🔍 DOMContentLoaded: Initializing data preview...');
                    initializeDataPreview();
                    
                    // Add global error handler for chart operations
                    window.addEventListener('error', function(event) {
                        if (event.message && event.message.includes('dataPreviewChart')) {
                            console.error('Chart error detected:', event.error);
                            // Try to recover by reinitializing
                            setTimeout(() => {
                                clearChartArea();
                                window.dataPreviewChart = null;
                            }, 100);
                        }
                    });
                });
                
                // Connect to separate SSE channels like test_separate_channels.py
                let trainingEventSource = null;
                let progressEventSource = null;
                let logEventSource = null;
                
                function connectTrainingChannel() {
                    if (trainingEventSource) {
                        trainingEventSource.close();
                    }
                    
                    console.log('🎯 Connecting to training channel...');
                    
                    try {
                        trainingEventSource = new EventSource('/events/training');
                        
                        trainingEventSource.onopen = function() {
                            console.log('🎯 Connected to Training SSE Channel');
                        };
                        
                        trainingEventSource.onmessage = function(event) {
                            try {
                                console.log('🎯 Raw training event received:', event);
                                const data = JSON.parse(event.data);
                                console.log('🎯 Training data received:', data);
                                
                                if (data.type === 'training_update') {
                                    console.log('🎯 Updating dashboard with training data:', data);
                                    updateDashboard(data);
                                } else if (data.type === 'training_complete') {
                                    console.log('🎯 Training completed:', data);
                                    console.log('🎯 Calling updateDashboard with training_complete event');
                                    updateDashboard(data);
                                    console.log('🎯 updateDashboard completed for training_complete');
                                } else if (data.type === 'training_start') {
                                    console.log('🎯 Training started:', data);
                                    updateDashboard(data);
                                } else if (data.type === 'connection') {
                                    console.log('🎯 Training channel connected:', data.message);

                                } else {
                                    console.log('🎯 Unknown training data type:', data.type, data);
                                }
                            } catch (error) {
                                console.error('🎯 Error parsing training data:', error, event.data);
                            }
                        };
                        
                        trainingEventSource.onerror = function(error) {
                            console.error('🎯 Training channel connection error:', error);
                            console.log('🎯 Training channel readyState:', trainingEventSource.readyState);
                            
                            // Try to reconnect if the connection was closed
                            if (trainingEventSource.readyState === EventSource.CLOSED) {
                                console.log('🔄 Training channel closed, attempting to reconnect...');
                                setTimeout(() => {
                                    connectTrainingChannel();
                                }, 2000);
                            }
                        };
                        
                    } catch (error) {
                        console.error('🎯 Error creating training EventSource:', error);
                    }
                }
                
                function connectProgressChannel() {
                    if (progressEventSource) {
                        progressEventSource.close();
                    }
                    
                    console.log('📊 Connecting to progress channel...');
                    
                    try {
                        progressEventSource = new EventSource('/events/progress');
                        
                        progressEventSource.onopen = function() {
                            console.log('📊 Connected to Progress SSE Channel');
                        };
                        
                        progressEventSource.onmessage = function(event) {
                            try {
                                const data = JSON.parse(event.data);
                                console.log('📊 Progress data received:', data);
                                
                                if (data.type === 'progress') {
                                    console.log('📊 Updating progress with data:', data);
                                    updateProgress(data);
                                } else if (data.type === 'connection') {
                                    console.log('📊 Progress channel connected:', data.message);

                                } else {
                                    console.log('📊 Unknown progress data type:', data.type, data);
                                }
                            } catch (error) {
                                console.error('📊 Error parsing progress data:', error, event.data);
                            }
                        };
                        
                        progressEventSource.onerror = function(error) {
                            console.error('📊 Progress channel connection error:', error);
                            console.log('📊 Progress channel readyState:', progressEventSource.readyState);
                            
                            // Try to reconnect if the connection was closed
                            if (progressEventSource.readyState === EventSource.CLOSED) {
                                console.log('🔄 Progress channel closed, attempting to reconnect...');
                                setTimeout(() => {
                                    connectProgressChannel();
                                }, 2000);
                            }
                        };
                        
                    } catch (error) {
                        console.error('📊 Error creating progress EventSource:', error);
                    }
                }
                
                function connectLogChannel() {
                    // DISABLED: Excessive log polling was causing UI refresh issues
                    // The logs channel was making requests every 3-4 seconds causing SSE interference
                    console.log('📝 Log channel disabled to prevent SSE interference');
                    // if (logEventSource) {
                    //     logEventSource.close();
                    // }
                    // 
                    // logEventSource = new EventSource('/events/logs');
                    // 
                    // logEventSource.onopen = function() {
                    //     console.log('📝 Connected to Log SSE Channel');
                    //     document.getElementById('log-status').textContent = 'Connected';
                    // };
                    // 
                    // logEventSource.onmessage = function(event) {
                    //     const data = JSON.parse(event.data);
                    //     console.log('📝 Log data received:', data);
                    //     
                    //     if (data.type === 'log_entry') {
                    //         updateLogs(data);
                    //     } else if (data.type === 'connection') {
                    //         console.log('📝 Log channel connected:', data.message);
                    //     }
                    // };
                    // 
                    // logEventSource.onerror = function() {
                    //     console.error('📝 Log channel connection error');
                    //     document.getElementById('log-status').textContent = 'Disconnected';
                    // };
                }
                
                function disconnectAllChannels() {
                    if (trainingEventSource) {
                        trainingEventSource.close();
                        trainingEventSource = null;
                    }
                    if (progressEventSource) {
                        progressEventSource.close();
                        progressEventSource = null;
                    }
                    if (logEventSource) {
                        logEventSource.close();
                        logEventSource = null;
                    }
                    console.log('🔌 All SSE channels disconnected');
                }
                

                
                function updateDashboard(data) {
                    console.log('🎯 Updating dashboard with:', data);
                    console.log('🎯 Data type:', data.type);
                    console.log('🎯 Data content:', JSON.stringify(data, null, 2));
                    
                    // Debug: Check if key elements exist
                    console.log('🔍 DOM Elements check:');
                    console.log('  - status-text:', document.getElementById('status-text'));
                    console.log('  - gen-loss:', document.getElementById('gen-loss'));
                    console.log('  - training controls:', document.querySelector('.p-4.bg-blue-50'));
                    console.log('  - main container:', document.querySelector('.max-w-7xl'));
                    
                    if (data.type === 'training_update') {
                        console.log('🎯 Processing training update:', data.data);
                        
                        // Update training status to Running when we receive training data
                        const statusElement = document.getElementById('status-text');
                        console.log('🎯 Looking for status-text element:', statusElement);
                        if (statusElement) {
                            statusElement.textContent = 'Running';
                            statusElement.className = 'text-xl font-bold text-green-600';
                            console.log('✅ Updated training status to Running');
                        } else {
                            console.error('❌ status-text element not found');
                        }
                        
                        // Update status cards
                        const genLossElement = document.getElementById('gen-loss');
                        const discLossElement = document.getElementById('disc-loss');
                        
                        console.log('🎯 Found elements:', {
                            genLoss: genLossElement,
                            discLoss: discLossElement
                        });
                        
                        if (genLossElement) {
                            genLossElement.textContent = data.data.generator_loss.toFixed(4);
                            console.log('✅ Updated generator loss:', data.data.generator_loss);
                        } else {
                            console.error('❌ gen-loss element not found');
                        }
                        
                        if (discLossElement) {
                            discLossElement.textContent = data.data.discriminator_loss.toFixed(4);
                            console.log('✅ Updated discriminator loss:', data.data.discriminator_loss);
                        } else {
                            console.error('❌ disc-loss element not found');
                        }
                        
                        // Reset training progress display when new training data comes in
                        const trainingProgressElement = document.getElementById('training-progress-display');
                        const trainingEpochElement = document.getElementById('training-epoch-display');
                        if (trainingProgressElement) {
                            trainingProgressElement.textContent = '0.0%';
                            console.log('✅ Reset training progress to 0.0%');
                        }
                        if (trainingEpochElement) {
                            trainingEpochElement.textContent = `Epoch ${data.data.epoch || 0}`;
                            console.log('✅ Updated training epoch:', data.data.epoch);
                        }
                        
                        // Update total epochs in overall progress section
                        const totalEpochsElement = document.getElementById('total-epochs');
                        if (totalEpochsElement && data.data.total_epochs) {
                            totalEpochsElement.textContent = data.data.total_epochs;
                            console.log('✅ Updated total epochs:', data.data.total_epochs);
                        }
                        
                        // Update charts
                        if (window.trainingChart) {
                            window.trainingChart.data.labels.push(data.data.epoch);
                            window.trainingChart.data.datasets[0].data.push(data.data.generator_loss);
                            window.trainingChart.data.datasets[1].data.push(data.data.discriminator_loss);
                            window.trainingChart.update();
                            console.log('✅ Updated training chart');
                        } else {
                            console.error('❌ trainingChart not found');
                        }
                        
                        if (window.scoresChart) {
                            window.scoresChart.data.labels.push(data.data.epoch);
                            window.scoresChart.data.datasets[0].data.push(data.data.real_scores);
                            window.scoresChart.data.datasets[1].data.push(data.data.fake_scores);
                            window.scoresChart.update();
                            console.log('✅ Updated scores chart');
                        } else {
                            console.error('❌ scoresChart not found');
                        }
                        
                    } else if (data.type === 'training_start') {
                        console.log('🚀 Training started:', data.data);
                        document.getElementById('status-text').textContent = 'Running';
                        document.getElementById('status-text').className = 'text-2xl font-bold text-green-600';
                        
                        // Update training controls
                        document.getElementById('start-training').disabled = true;
                        document.getElementById('start-training').classList.add('opacity-50', 'cursor-not-allowed');
                        document.getElementById('stop-training').disabled = false;
                        document.getElementById('stop-training').classList.remove('opacity-50', 'cursor-not-allowed');
                        
                        // Show training status indicator
                        const trainingStatusIndicator = document.getElementById('training-status-indicator');
                        if (trainingStatusIndicator) {
                            trainingStatusIndicator.classList.remove('hidden');
                        }
                        
                        // Show and initialize overall training progress section
                        const overallProgressSection = document.getElementById('training-progress-section');
                        const totalEpochsElement = document.getElementById('total-epochs');
                        if (overallProgressSection && totalEpochsElement) {
                            overallProgressSection.style.display = 'block';
                            // Reset progress to 0 when training starts
                            const overallProgressBar = document.getElementById('overall-progress-bar');
                            const overallProgressText = document.getElementById('overall-progress-text');
                            if (overallProgressBar && overallProgressText) {
                                overallProgressBar.style.width = '0%';
                                overallProgressText.textContent = 'Overall: 0% complete';
                            }
                        }
                        
                        // Reset epoch progress display
                        const epochProgressElement = document.getElementById('epoch-progress');
                        if (epochProgressElement) {
                            epochProgressElement.textContent = '0% complete';
                        }
                        
                        // Show training info
                        const trainingInfo = document.createElement('div');
                        trainingInfo.className = 'mt-4 p-4 bg-green-50 rounded-lg border border-green-200';
                        trainingInfo.innerHTML = `
                            <h4 class="text-md font-medium text-green-700 mb-2">🚀 Training Started</h4>
                            <div class="text-sm text-green-600">
                                <p><strong>Config:</strong> ${data.data.config}</p>
                                <p><strong>Data Source:</strong> ${data.data.data_source}</p>
                                <p><strong>Status:</strong> ${data.data.status}</p>
                                <p><strong>Time:</strong> ${new Date(data.timestamp).toLocaleTimeString()}</p>
                            </div>
                        `;
                        
                        // Remove existing training info if any
                        const existingInfo = document.querySelector('.bg-green-50');
                        if (existingInfo) {
                            existingInfo.remove();
                        }
                        
                        // Insert after training controls (with null check)
                        const trainingControls = document.querySelector('.p-4.bg-blue-50');
                        if (trainingControls && trainingControls.parentNode) {
                            trainingControls.parentNode.insertBefore(trainingInfo, trainingControls.nextSibling);
                        } else {
                            // Fallback: append to main container
                            const mainContainer = document.querySelector('.max-w-7xl');
                            if (mainContainer) {
                                mainContainer.appendChild(trainingInfo);
                            } else {
                                console.warn('Could not find container to insert training info');
                            }
                        }
                        
                    } else if (data.type === 'training_update') {
                        // Log training update but don't assume completion based on epoch number alone
                        console.log('📊 Training update received:', data.data);
                        
                        // Note: We should only show completion when we receive a proper 'training_complete' event
                        // Do not assume completion based on epoch numbers here
                        
                    } else if (data.type === 'training_complete') {
                        console.log('✅ Training completed:', data.data);
                        console.log('🎯 Updating status text for training_complete');
                        const statusElement = document.getElementById('status-text');
                        if (statusElement) {
                            const newStatusText = data.data.status === 'completed' ? 'Completed' : 'Failed';
                            const newStatusClass = data.data.status === 'completed' ? 
                                'text-2xl font-bold text-green-600' : 'text-2xl font-bold text-red-600';
                            
                            console.log(`🎯 Setting status from "${statusElement.textContent}" to "${newStatusText}"`);
                            statusElement.textContent = newStatusText;
                            statusElement.className = newStatusClass;
                            console.log('✅ Status text updated successfully');
                        } else {
                            console.error('❌ status-text element not found in training_complete handler');
                        }
                        
                        // Re-enable start training button
                        document.getElementById('start-training').disabled = false;
                        document.getElementById('start-training').classList.remove('opacity-50', 'cursor-not-allowed');
                        document.getElementById('stop-training').disabled = true;
                        document.getElementById('stop-training').classList.add('opacity-50', 'cursor-not-allowed');
                        
                        // Hide training status indicator
                        const trainingStatusIndicator = document.getElementById('training-status-indicator');
                        if (trainingStatusIndicator) {
                            trainingStatusIndicator.classList.add('hidden');
                        }
                        
                        // Hide overall training progress section
                        const overallProgressSection = document.getElementById('training-progress-section');
                        if (overallProgressSection) {
                            overallProgressSection.style.display = 'none';
                        }
                        
                        // Show completion info
                        const completionInfo = document.createElement('div');
                        completionInfo.className = `mt-4 p-4 ${data.data.status === 'completed' ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'} rounded-lg border`;
                        completionInfo.innerHTML = `
                            <h4 class="text-md font-medium ${data.data.status === 'completed' ? 'text-green-700' : 'text-red-700'} mb-2">
                                ${data.data.status === 'completed' ? '✅ Training Completed' : '❌ Training Failed'}
                            </h4>
                            <div class="text-sm ${data.data.status === 'completed' ? 'text-green-600' : 'text-red-600'}">
                                <p><strong>Message:</strong> ${data.data.message}</p>
                                <p><strong>Return Code:</strong> ${data.data.return_code}</p>
                                <p><strong>Time:</strong> ${new Date(data.timestamp).toLocaleTimeString()}</p>
                            </div>
                        `;
                        
                        // Remove existing completion info if any
                        const existingCompletion = document.querySelector('.bg-green-50, .bg-red-50');
                        if (existingCompletion && existingCompletion !== document.querySelector('.bg-blue-50')) {
                            existingCompletion.remove();
                        }
                        
                        // Insert after training controls (with null check)
                        const trainingControls = document.querySelector('.p-4.bg-blue-50');
                        if (trainingControls && trainingControls.parentNode) {
                            trainingControls.parentNode.insertBefore(completionInfo, trainingControls.nextSibling);
                        } else {
                            // Fallback: append to main container
                            const mainContainer = document.querySelector('.max-w-7xl');
                            if (mainContainer) {
                                mainContainer.appendChild(completionInfo);
                            } else {
                                console.warn('Could not find container to insert completion info');
                            }
                        }
                        
                    } else if (data.type === 'status_update') {
                        console.log('📊 Status update:', data.data);
                        // Update the status display
                        const statusText = document.getElementById('status-text');
                        if (statusText) {
                            statusText.textContent = data.data.status.charAt(0).toUpperCase() + data.data.status.slice(1);
                            
                            // Update status color based on status
                            if (data.data.status === 'completed') {
                                statusText.className = 'text-2xl font-bold text-green-600';
                            } else if (data.data.status === 'failed') {
                                statusText.className = 'text-2xl font-bold text-red-600';
                            } else if (data.data.status === 'running') {
                                statusText.className = 'text-2xl font-bold text-blue-600';
                            } else if (data.data.status === 'stopped') {
                                statusText.className = 'text-2xl font-bold text-gray-600';
                            } else {
                                statusText.className = 'text-2xl font-bold text-gray-600';
                            }
                        }
                        
                        // Update button states based on status
                        const startButton = document.getElementById('start-training');
                        const stopButton = document.getElementById('stop-training');
                        
                        if (startButton && stopButton) {
                            if (data.data.status === 'completed' || data.data.status === 'failed' || data.data.status === 'stopped') {
                                // Re-enable start button
                                startButton.disabled = false;
                                startButton.classList.remove('opacity-50', 'cursor-not-allowed');
                                // Disable stop button
                                stopButton.disabled = true;
                                stopButton.classList.add('opacity-50', 'cursor-not-allowed');
                            } else if (data.data.status === 'running') {
                                // Disable start button
                                startButton.disabled = true;
                                startButton.classList.add('opacity-50', 'cursor-not-allowed');
                                // Enable stop button
                                stopButton.disabled = false;
                                stopButton.classList.remove('opacity-50', 'cursor-not-allowed');
                            }
                        }
                    }
                }
                
                function updateProgress(data) {
                    // Update progress indicators
                    console.log('📊 Progress update:', data);
                    
                    if (data.type === 'progress') {
                        const progressPercent = data.progress_percent || 0;
                        
                        // Update training progress display in status cards only
                        const trainingProgressElement = document.getElementById('training-progress-display');
                        const trainingEpochElement = document.getElementById('training-epoch-display');
                        if (trainingProgressElement) {
                            trainingProgressElement.textContent = `${progressPercent.toFixed(1)}%`;
                        }
                        if (trainingEpochElement) {
                            trainingEpochElement.textContent = `Epoch ${data.epoch || 0}`;
                        }
                        
                        // Only show completion when training is actually complete (100% AND all epochs finished)
                        // We should wait for the proper 'training_complete' event rather than assuming completion from progress
                        if (progressPercent === 100) {
                            console.log('🎯 Training progress reached 100% - training should complete soon');
                            // Note: We wait for the official 'training_complete' event to update UI state
                        }
                    }
                }
                
                function updateLogs(data) {
                    // Update log display
                    console.log('📝 Log update:', data);
                    
                    if (data.type === 'log_entry') {

                        

                    }
                }
                
                // Function to show completion notifications
                function showCompletionNotification(message, type = 'info') {
                    // Remove existing notifications
                    const existingNotifications = document.querySelectorAll('.completion-notification');
                    existingNotifications.forEach(notification => notification.remove());
                    
                    // Create notification element
                    const notification = document.createElement('div');
                    notification.className = `completion-notification fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 max-w-sm ${
                        type === 'success' ? 'bg-green-500 text-white' : 
                        type === 'error' ? 'bg-red-500 text-white' : 
                        'bg-blue-500 text-white'
                    }`;
                    
                    notification.innerHTML = `
                        <div class="flex items-center">
                            <span class="mr-2">${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span>
                            <span class="font-medium">${message}</span>
                            <button onclick="this.parentElement.parentElement.remove()" class="ml-auto text-white hover:text-gray-200">×</button>
                        </div>
                    `;
                    
                    // Add to page
                    document.body.appendChild(notification);
                    
                    // Auto-remove after 5 seconds
                    setTimeout(() => {
                        if (notification.parentElement) {
                            notification.remove();
                        }
                    }, 5000);
                }
                
                // Connect to all channels on page load
                window.addEventListener('load', function() {
                    console.log('🚀 Page loaded, setting up SSE channels');
                    connectTrainingChannel();
                    connectProgressChannel();
                    // connectLogChannel(); // DISABLED: Causing SSE interference
                    console.log('🔍 Window load: Initializing data source selection...');
                    initializeDataSourceSelection();
                });
                
                // Handle page visibility changes to maintain connections
                document.addEventListener('visibilitychange', function() {
                    if (document.hidden) {
                        console.log('👁️ Page hidden');
                    } else {
                        console.log('👁️ Page visible');
                        // Reconnect if connections were lost
                        if (trainingEventSource && trainingEventSource.readyState === EventSource.CLOSED) {
                            console.log('🔄 Reconnecting training channel after page became visible');
                            connectTrainingChannel();
                        }
                        if (progressEventSource && progressEventSource.readyState === EventSource.CLOSED) {
                            console.log('🔄 Reconnecting progress channel after page became visible');
                            connectProgressChannel();
                        }
                    }
                });
                
                // Monitor connection health (reduced frequency to avoid interference)
                setInterval(() => {
                    if (trainingEventSource && trainingEventSource.readyState === EventSource.CLOSED) {
                        console.log('🔄 Training channel appears closed, reconnecting...');
                        connectTrainingChannel();
                    }
                    if (progressEventSource && progressEventSource.readyState === EventSource.CLOSED) {
                        console.log('🔄 Progress channel appears closed, reconnecting...');
                        connectProgressChannel();
                    }
                }, 30000); // Check every 30 seconds (reduced from 10s to avoid interference)
                
                // Removed auto-refresh polling - Training status is now only available via SSE
                
                // Button event listeners
                document.getElementById('start-training').addEventListener('click', async () => {
                    if (!selectedDataSource) {
                        alert('Please select a data source first (Upload CSV or Generate Sample)');
                        return;
                    }
                    
                    try {
                        // Get the epochs value from the input field
                        const epochsInput = document.getElementById('epochs-input');
                        let epochs = epochsInput ? parseInt(epochsInput.value) || 5 : 5;
                        
                        // Validate epochs value
                        if (epochs < 1) epochs = 1;
                        if (epochs > 1000) epochs = 1000;
                        
                        // Update the input field with the validated value
                        if (epochsInput) epochsInput.value = epochs;
                        
                        const response = await fetch('/api/start_training', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                config: 'config/gan_config.yaml',
                                data_source: selectedDataSource,
                                epochs: epochs
                            })
                        });
                        const result = await response.json();
                        if (result.success) {
                            document.getElementById('status-text').textContent = 'Running';
                            document.getElementById('status-text').className = 'text-2xl font-bold text-green-600';
                            
                            // Update training controls
                            document.getElementById('start-training').disabled = true;
                            document.getElementById('start-training').classList.add('opacity-50', 'cursor-not-allowed');
                            document.getElementById('stop-training').disabled = false;
                            document.getElementById('stop-training').classList.remove('opacity-50', 'cursor-not-allowed');
                            
                            // Show training status indicator
                            const trainingStatusIndicator = document.getElementById('training-status-indicator');
                            if (trainingStatusIndicator) {
                                trainingStatusIndicator.classList.remove('hidden');
                            }
                            
                            // Clear any existing completion notifications
                            const existingNotifications = document.querySelectorAll('.completion-notification');
                            existingNotifications.forEach(notification => notification.remove());
                            
                            // Clear any existing completion info
                            const existingCompletion = document.querySelector('.bg-green-50, .bg-red-50');
                            if (existingCompletion && existingCompletion !== document.querySelector('.p-4.bg-blue-50')) {
                                existingCompletion.remove();
                            }
                            
                            // Reset progress bar if it exists
                            const progressBar = document.getElementById('training-progress-bar');
                            if (progressBar) {
                                progressBar.style.width = '0%';
                                const progressText = document.getElementById('progress-text');
                                const progressEpoch = document.getElementById('progress-epoch');
                                if (progressText) progressText.textContent = '0%';
                                if (progressEpoch) progressEpoch.textContent = '0';
                            }
                            
                            // Clear any existing training info
                            const existingTrainingInfo = document.querySelector('.bg-green-50');
                            if (existingTrainingInfo && existingTrainingInfo !== document.querySelector('.p-4.bg-blue-50')) {
                                existingTrainingInfo.remove();
                            }
                            

                            
                            // Reset charts if they exist
                            if (window.lossChart) {
                                window.lossChart.data.labels = [];
                                window.lossChart.data.datasets[0].data = [];
                                window.lossChart.data.datasets[1].data = [];
                                window.lossChart.update();
                            }
                            if (window.scoresChart) {
                                window.scoresChart.data.labels = [];
                                window.scoresChart.data.datasets[0].data = [];
                                window.scoresChart.data.datasets[1].data = [];
                                window.scoresChart.update();
                            }
                        }
                    } catch (error) {
                        console.error('Error starting training:', error);
                    }
                });
                
                document.getElementById('stop-training').addEventListener('click', async () => {
                    try {
                        const response = await fetch('/api/stop_training', {method: 'POST'});
                        const result = await response.json();
                        if (result.success) {
                            document.getElementById('status-text').textContent = 'Stopped';
                            document.getElementById('status-text').className = 'text-2xl font-bold text-red-600';
                            
                            // Re-enable start training button
                            document.getElementById('start-training').disabled = false;
                            document.getElementById('start-training').classList.remove('opacity-50', 'cursor-not-allowed');
                            document.getElementById('stop-training').disabled = true;
                            document.getElementById('stop-training').classList.add('opacity-50', 'cursor-not-allowed');
                            
                            // Hide training status indicator
                            const trainingStatusIndicator = document.getElementById('training-status-indicator');
                            if (trainingStatusIndicator) {
                                trainingStatusIndicator.classList.add('hidden');
                            }
                        }
                    } catch (error) {
                        console.error('Error stopping training:', error);
                    }
                });
                
                document.getElementById('generate-sample').addEventListener('click', async () => {
                    try {
                        // Show loading state
                        const generateBtn = document.getElementById('generate-sample');
                        const originalText = generateBtn.innerHTML;
                        generateBtn.innerHTML = '🔄 Generating...';
                        generateBtn.disabled = true;
                        
                        const response = await fetch('/api/generate_sample', {method: 'POST'});
                        const result = await response.json();
                        
                        if (result.success) {
                            // Show success notification
                            showCompletionNotification(`✅ Sample generated successfully! ${result.summary.total_records} records created.`, 'success');
                            
                            // Automatically select the generated sample as data source
                            if (result.filename) {
                                window.selectDataSource(result.filename, 'treasury_orderbook');
                                
                                // Update comprehensive option to show it's available
                                const comprehensiveOption = document.getElementById('comprehensive-option');
                                if (comprehensiveOption) {
                                    comprehensiveOption.className = 'p-2 bg-gradient-to-r from-blue-50 to-purple-50 rounded border border-blue-300 hover:border-blue-400 cursor-pointer transition-all text-center';
                                    comprehensiveOption.onclick = () => window.selectDataSource('treasury_orderbook_levels.csv', 'treasury_orderbook');
                                    comprehensiveOption.innerHTML = `
                                        <div class="w-3 h-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full mx-auto mb-1"></div>
                                        <span class="text-xs font-medium text-blue-700">🎯 Treasury Orderbook Data</span>
                                        <div class="text-xs text-blue-600 mt-1">24h of 5-level orderbook data available</div>
                                    `;
                                }
                            }
                        } else {
                            showCompletionNotification(`❌ Failed to generate sample: ${result.error}`, 'error');
                        }
                    } catch (error) {
                        console.error('Error generating sample:', error);
                        showCompletionNotification(`❌ Error generating sample: ${error.message}`, 'error');
                    } finally {
                        // Restore button state
                        const generateBtn = document.getElementById('generate-sample');
                        generateBtn.innerHTML = originalText;
                        generateBtn.disabled = false;
                    }
                });
                

                
                // Training state management
                let selectedDataSource = null;
                let selectedDataType = null;
                
                // Initialize data source selection
                function initializeDataSourceSelection() {
                    console.log('🔍 initializeDataSourceSelection: Starting...');
                    const startTrainingBtn = document.getElementById('start-training');
                    const stopTrainingBtn = document.getElementById('stop-training');
                    const uploadCsvBtn = document.getElementById('upload-csv');
                    const generateSampleBtn = document.getElementById('generate-sample');
                    const csvFileInput = document.getElementById('csv-file-input');
                    const selectedFileName = document.getElementById('selected-file-name');
                    
                    // Initially disable training controls
                    startTrainingBtn.disabled = true;
                    startTrainingBtn.classList.add('opacity-50', 'cursor-not-allowed');
                    stopTrainingBtn.disabled = true;
                    stopTrainingBtn.classList.add('opacity-50', 'cursor-not-allowed');
                    
                    // Data source selection handlers
                    function selectDataSource(filename, dataType) {
                        console.log(`🔍 selectDataSource called with: ${filename}, ${dataType}`);
                        console.log(`🔍 Call stack:`, new Error().stack);
                        selectedDataSource = filename;
                        selectedDataType = dataType;
                        
                        // Update UI to show selection with enhanced styling for treasury data
                        const statusText = `Selected: ${filename} (${dataType})`;
                        document.getElementById('data-source-status').textContent = statusText;
                        
                        // Enhanced styling based on data type
                        let indicatorClass = 'mt-2 p-2 rounded border bg-green-50';
                        if (dataType === 'treasury_orderbook') {
                            indicatorClass += ' border-blue-300 bg-blue-50';
                            document.getElementById('data-source-status').className = 'text-blue-700 font-medium';
                        } else if (dataType === 'orderbook') {
                            indicatorClass += ' border-purple-300 bg-purple-50';
                            document.getElementById('data-source-status').className = 'text-purple-700 font-medium';
                        } else if (dataType === 'timeseries') {
                            indicatorClass += ' border-green-300 bg-green-50';
                            document.getElementById('data-source-status').className = 'text-green-700 font-medium';
                        } else {
                            indicatorClass += ' border-green-200 bg-green-50';
                            document.getElementById('data-source-status').className = 'text-green-700 font-medium';
                        }
                        
                        document.getElementById('data-source-indicator').className = indicatorClass;
                        
                        // Show data preview section
                        document.getElementById('data-preview-section').style.display = 'block';
                        
                        // Load and display data preview
                        loadDataPreview(filename);
                        
                        // Enable start training button
                        startTrainingBtn.disabled = false;
                        startTrainingBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                        
                        console.log(`📊 Data source selected: ${filename} (${dataType})`);
                    }
                    
                    // Make selectDataSource globally accessible
                    window.selectDataSource = selectDataSource;
                    
                    async function loadDataPreview(filename) {
                        try {
                            const response = await fetch(`/api/preview_csv?filename=${encodeURIComponent(filename)}`);
                            const result = await response.json();
                            
                            if (result.success) {
                                displayDataPreview(result.data_info, result.plot_data, filename);
                            } else {
                                console.error('Error loading data preview:', result.error);
                                // Show error message in the table preview section
                                document.getElementById('data-table-preview').innerHTML = `
                                    <div class="text-center text-red-500 py-4">
                                        <div class="text-sm">❌ Error loading data: ${result.error}</div>
                                    </div>
                                `;
                            }
                        } catch (error) {
                            console.error('Error loading data preview:', error);
                            // Show error message in the table preview section
                            document.getElementById('data-table-preview').innerHTML = `
                                <div class="text-center text-red-500 py-4">
                                    <div class="text-sm">❌ Network error: ${error.message}</div>
                                </div>
                            `;
                        }
                    }
                    
                    function displayDataPreview(dataInfo, plotData, filename) {
                        // Create chart for the single location
                        if (plotData && Object.keys(plotData).length > 1) {
                            createChartForElement('dataPreviewChart', plotData, 'Time Series Visualization');
                        } else {
                            // Show message when no plot data available
                            const chartElement = document.getElementById('dataPreviewChart');
                            
                            if (chartElement && chartElement.parentElement) {
                                chartElement.parentElement.innerHTML = `
                                    <div class="text-center text-gray-500 py-8">
                                        <div class="text-2xl mb-2">📊</div>
                                        <div class="text-sm">No numeric data available for visualization</div>
                                    </div>
                                `;
                            }
                        }
                        
                        // Update data table preview with enhanced formatting for treasury data
                        const tablePreview = document.getElementById('data-table-preview');
                        if (dataInfo.sample_data && dataInfo.sample_data.length > 0) {
                            const headers = Object.keys(dataInfo.sample_data[0]);
                            
                            // Enhanced formatting function for orderbook data
                            function formatCellValue(header, value) {
                                if (value === null || value === undefined) return '-';
                                
                                // Special formatting for orderbook-specific columns
                                if (header === 'SYM') {
                                    return `<span class="font-mono font-bold text-blue-600">${value}</span>`;
                                }
                                if (header === 'TIMESTAMP') {
                                    return `<span class="font-mono text-gray-700">${value}</span>`;
                                }
                                if (header.includes('BID_PRICE') || header.includes('ASK_PRICE')) {
                                    return `<span class="font-mono text-green-600">$${parseFloat(value).toFixed(4)}</span>`;
                                }
                                if (header.includes('BID_SIZE') || header.includes('ASK_SIZE')) {
                                    return `<span class="font-mono text-purple-600">${parseInt(value).toLocaleString()}</span>`;
                                }
                                if (header === 'TOTAL_VOLUME') {
                                    return `<span class="font-mono text-indigo-600">${parseInt(value).toLocaleString()}</span>`;
                                }
                                if (header === 'MARKET_HOUR') {
                                    return value ? 
                                        '<span class="text-green-600 font-bold">●</span>' : 
                                        '<span class="text-gray-400">○</span>';
                                }
                                
                                return value;
                            }
                            
                            const tableHTML = `
                                <div class="overflow-x-auto">
                                    <table class="w-full text-xs">
                                        <thead>
                                            <tr class="bg-gray-200">
                                                ${headers.map(h => `<th class="p-2 text-left font-medium text-gray-700">${h.replace(/_/g, ' ').toUpperCase()}</th>`).join('')}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            ${dataInfo.sample_data.map(row => 
                                                `<tr class="border-b border-gray-200 hover:bg-gray-50 transition-colors">
                                                    ${headers.map(h => `<td class="p-2">${formatCellValue(h, row[h])}</td>`).join('')}
                                                </tr>`
                                            ).join('')}
                                        </tbody>
                                    </table>
                                </div>
                                <div class="mt-2 text-xs text-gray-500 text-center">
                                    Showing ${dataInfo.sample_data.length} of ${dataInfo.row_count} total records
                                </div>
                            `;
                            tablePreview.innerHTML = tableHTML;
                        } else {
                            // Show message when no sample data available
                            tablePreview.innerHTML = `
                                <div class="text-center text-gray-500 py-4">
                                    <div class="text-sm">No sample data available</div>
                                </div>
                            `;
                        }
                    }
                    
                    // No default data source - user must select or generate data
                    
                    // Check if comprehensive sample exists and show option
                    function checkComprehensiveSample() {
                        fetch('/api/preview_csv?filename=treasury_orderbook_levels.csv')
                            .then(response => response.json())
                            .then(result => {
                                if (result.success) {
                                    // Show comprehensive option as available
                                    const comprehensiveOption = document.getElementById('comprehensive-option');
                                    if (comprehensiveOption) {
                                        comprehensiveOption.className = 'p-2 bg-gradient-to-r from-blue-50 to-purple-50 rounded border border-blue-300 hover:border-blue-400 cursor-pointer transition-all text-center';
                                        comprehensiveOption.onclick = () => window.selectDataSource('treasury_orderbook_levels.csv', 'treasury_orderbook');
                                        comprehensiveOption.innerHTML = `
                                            <div class="w-3 h-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full mx-auto mb-1"></div>
                                            <span class="text-xs font-medium text-blue-700">🎯 Treasury Orderbook Data</span>
                                            <div class="text-xs text-blue-600 mt-1">24h of 5-level orderbook data available</div>
                                        `;
                                    }
                                }
                            })
                            .catch(error => {
                                console.log('Orderbook sample not yet available');
                            });
                    }
                    
                    // Check for comprehensive sample on page load
                    checkComprehensiveSample();
                    
                    // File upload handling
                    uploadCsvBtn.addEventListener('click', () => {
                        csvFileInput.click();
                    });
                    
                    csvFileInput.addEventListener('change', (event) => {
                        const file = event.target.files[0];
                        if (file) {
                            selectedFileName.textContent = file.name;
                            
                            // Upload file
                            const formData = new FormData();
                            formData.append('file', file);
                            
                            fetch('/api/upload_csv', {
                                method: 'POST',
                                body: formData
                            })
                            .then(response => response.json())
                            .then(result => {
                                if (result.success) {
                                    selectDataSource(file.name, 'custom');
                                } else {
                                    alert('Error uploading file: ' + result.error);
                                }
                            })
                            .catch(error => {
                                console.error('Error uploading file:', error);
                                alert('Error uploading file');
                            });
                        }
                    });
                    
                    // Generate sample data
                    generateSampleBtn.addEventListener('click', async () => {
                        try {
                            const response = await fetch('/api/generate_sample', {method: 'POST'});
                            const result = await response.json();
                            if (result.success) {
                                // Use the actual generated filename from the response
                                const generatedFilename = result.filename;
                                
                                // Update the data source selection without showing popup
                                selectDataSource(generatedFilename, 'generated');
                                
                                // Show a subtle success indicator instead of alert
                                const statusElement = document.getElementById('data-source-status');
                                const originalText = statusElement.textContent;
                                statusElement.textContent = `✅ Sample generated: ${generatedFilename}`;
                                statusElement.className = 'text-green-700 font-medium';
                                
                                // Reset status after 3 seconds
                                setTimeout(() => {
                                    statusElement.textContent = originalText;
                                    statusElement.className = 'text-blue-700 font-medium';
                                }, 3000);
                                
                                // No need to call additional preview functions since selectDataSource already handles it
                            }
                        } catch (error) {
                            console.error('Error generating sample:', error);
                            // Show error in status instead of alert
                            const statusElement = document.getElementById('data-source-status');
                            statusElement.textContent = `❌ Error generating sample: ${error.message}`;
                            statusElement.className = 'text-red-700 font-medium';
                        }
                    });
                    
                    console.log('🔍 initializeDataSourceSelection: Completed - no automatic data loading');
                }
                
                // CSV upload and preview functionality
                async function uploadAndPreviewCSV(file) {
                    try {
                        const formData = new FormData();
                        formData.append('file', file);
                        
                        // Upload the file
                        const uploadResponse = await fetch('/api/upload_csv', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const uploadResult = await uploadResponse.json();
                        
                        if (uploadResult.success) {
                            // Get data preview
                            const previewResponse = await fetch(`/api/preview_csv?filename=${file.name}`);
                            const previewResult = await previewResponse.json();
                            
                            if (previewResult.success) {
                                // Show both the detailed preview and the compact preview in selector
                                showDataPreview(previewResult.data_info, previewResult.plot_data, file.name);
                                showSelectedDataPreview(previewResult.data_info, previewResult.plot_data, file.name, 'custom');
                                
                                // Update visual selection for custom upload
                                document.querySelectorAll('[onclick^="selectDataSource"]').forEach(el => {
                                    el.classList.remove('border-blue-500', 'bg-blue-50');
                                    el.classList.add('border-gray-200', 'bg-white');
                                });
                            } else {
                                console.error('Error getting preview:', previewResult.error);
                            }
                        } else {
                            console.error('Error uploading file:', uploadResult.error);
                        }
                    } catch (error) {
                        console.error('Error uploading CSV:', error);
                    }
                }
                
                // Show data preview
                function showDataPreview(dataInfo, plotData, filename) {
                    const previewSection = document.getElementById('data-preview-section');
                    const tablePreview = document.getElementById('data-table-preview');
                    
                    // Show the preview section
                    previewSection.style.display = 'block';
                    
                    // Update data table preview
                    if (dataInfo.sample_data && dataInfo.sample_data.length > 0) {
                        const columns = Object.keys(dataInfo.sample_data[0]);
                        let tableHTML = '<table class="w-full text-xs border-collapse">';
                        
                        // Header
                        tableHTML += '<thead><tr class="bg-gray-200">';
                        columns.forEach(col => {
                            tableHTML += `<th class="border border-gray-300 px-2 py-1 text-left">${col}</th>`;
                        });
                        tableHTML += '</tr></thead>';
                        
                        // Data rows
                        tableHTML += '<tbody>';
                        dataInfo.sample_data.forEach(row => {
                            tableHTML += '<tr>';
                            columns.forEach(col => {
                                const value = row[col];
                                const displayValue = typeof value === 'number' ? value.toFixed(4) : value;
                                tableHTML += `<td class="border border-gray-300 px-2 py-1">${displayValue}</td>`;
                            });
                            tableHTML += '</tr>';
                        });
                        tableHTML += '</tbody></table>';
                        
                        tablePreview.innerHTML = tableHTML;
                    }
                    
                    // Create time series chart
                    createDataPreviewChart(plotData);
                }
                
                // Hide data preview
                function hideDataPreview() {
                    const previewSection = document.getElementById('data-preview-section');
                    previewSection.style.display = 'none';
                }
                
                // Create data preview chart
                function createDataPreviewChart(plotData) {
                    // Create chart for the single location
                    createChartForElement('dataPreviewChart', plotData, 'Time Series Visualization');
                }
                
                // Helper function to create chart for a specific element
                function createChartForElement(elementId, plotData, chartTitle) {
                    const chartElement = document.getElementById(elementId);
                    if (!chartElement) {
                        console.error(`Chart element ${elementId} not found`);
                        return;
                    }
                    
                    // Validate plot data using helper function
                    const validation = validateChartData(plotData);
                    if (!validation.valid) {
                        console.error('Data validation failed:', validation.error);
                        const chartContainer = chartElement.parentElement;
                        if (chartContainer) {
                            chartContainer.innerHTML = `
                                <div class="text-center text-red-500 py-8">
                                    <div class="text-2xl mb-2">❌</div>
                                    <div class="text-sm">${validation.error}</div>
                                    <div class="text-xs text-gray-500 mt-1">Please check your data format</div>
                                </div>
                            `;
                        }
                        return;
                    }
                    
                    const ctx = chartElement.getContext('2d');
                    
                    // Destroy existing chart if it exists and has destroy method
                    const existingChart = window[elementId + 'Chart'];
                    safeDestroyChart(existingChart);
                    
                    const datasets = [];
                    const colors = ['rgb(59, 130, 246)', 'rgb(239, 68, 68)', 'rgb(34, 197, 94)', 'rgb(168, 85, 247)', 'rgb(245, 158, 11)'];
                    
                    validation.columns.forEach((column, index) => {
                        datasets.push({
                            label: column,
                            data: plotData[column],
                            borderColor: colors[index % colors.length],
                            backgroundColor: colors[index % colors.length].replace('rgb', 'rgba').replace(')', ', 0.1)'),
                            tension: 0.1,
                            pointRadius: 2
                        });
                    });
                    
                    try {
                        const newChart = new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: plotData.index || [],
                                datasets: datasets
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: {
                                    x: {
                                        title: {
                                            display: true,
                                            text: 'Time Index'
                                        }
                                    },
                                    y: {
                                        title: {
                                            display: true,
                                            text: 'Value'
                                        }
                                    }
                                },
                                plugins: {
                                    title: {
                                        display: true,
                                        text: chartTitle
                                    },
                                    legend: {
                                        position: 'top'
                                    }
                                }
                            }
                        });
                        
                        // Store chart reference
                        window[elementId + 'Chart'] = newChart;
                        console.log(`Chart ${elementId} created successfully`);
                    } catch (error) {
                        handleChartError(error, 'chart creation');
                    }
                }
                
                // Duplicate function removed - using the one defined in initializeDataSourceSelection()
                
                // Show selected data preview in the selector
                function showSelectedDataPreview(dataInfo, plotData, filename, dataType) {
                    const previewSection = document.getElementById('selected-data-preview');
                    const mainPreviewSection = document.getElementById('data-preview-section');
                    
                    // Show the main data preview section
                    if (mainPreviewSection) {
                        mainPreviewSection.style.display = 'block';
                    }
                    const previewContent = document.getElementById('data-preview-content');
                    
                    // Show the preview section
                    previewSection.classList.remove('hidden');
                    
                    // Create compact preview content
                    let previewHTML = `
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs mb-3">
                            <div class="bg-gray-50 p-2 rounded border border-gray-200">
                                <div class="font-medium text-gray-700">📁 File</div>
                                <div class="text-gray-600 truncate">${filename}</div>
                            </div>
                            <div class="bg-gray-50 p-2 rounded border border-gray-200">
                                <div class="font-medium text-gray-700">📊 Shape</div>
                                <div class="text-gray-600">${dataInfo.shape[0]} × ${dataInfo.shape[1]}</div>
                            </div>
                            <div class="bg-gray-50 p-2 rounded border border-gray-200">
                                <div class="font-medium text-gray-700">🔍 Type</div>
                                <div class="text-gray-600">${dataInfo.data_type}</div>
                            </div>
                            <div class="bg-gray-50 p-2 rounded border border-gray-200">
                                <div class="font-medium text-gray-700">📈 Columns</div>
                                <div class="text-gray-600">${dataInfo.numeric_columns.length} numeric</div>
                            </div>
                        </div>
                    `;
                    
                    // Add data statistics if available
                    if (dataInfo.summary_stats && Object.keys(dataInfo.summary_stats).length > 0) {
                        const firstNumericCol = dataInfo.numeric_columns[0];
                        if (firstNumericCol && dataInfo.summary_stats[firstNumericCol]) {
                            const stats = dataInfo.summary_stats[firstNumericCol];
                            previewHTML += `
                                <div class="mb-3 p-2 bg-blue-50 rounded border border-blue-200">
                                    <div class="font-medium text-blue-700 text-xs mb-1">📊 Sample Statistics (${firstNumericCol})</div>
                                    <div class="grid grid-cols-2 gap-2 text-xs">
                                        <div><span class="font-medium">Min:</span> ${stats.min ? stats.min.toFixed(4) : 'N/A'}</div>
                                        <div><span class="font-medium">Max:</span> ${stats.max ? stats.max.toFixed(4) : 'N/A'}</div>
                                        <div><span class="font-medium">Mean:</span> ${stats.mean ? stats.mean.toFixed(4) : 'N/A'}</div>
                                        <div><span class="font-medium">Std:</span> ${stats.std ? stats.std.toFixed(4) : 'N/A'}</div>
                                    </div>
                                </div>
                            `;
                        }
                    }
                    
                    // Add sample data preview (first 3 rows)
                    if (dataInfo.sample_data && dataInfo.sample_data.length > 0) {
                        const columns = Object.keys(dataInfo.sample_data[0]);
                        let tableHTML = '<div class="mt-3"><div class="font-medium text-gray-700 mb-2">📋 Sample Data (First 3 rows):</div><div class="overflow-x-auto"><table class="w-full text-xs border-collapse">';
                        
                        // Header
                        tableHTML += '<thead><tr class="bg-gray-200">';
                        columns.forEach(col => {
                            tableHTML += `<th class="border border-gray-300 px-1 py-1 text-left">${col}</th>`;
                        });
                        tableHTML += '</tr></thead>';
                        
                        // Data rows (first 3 only)
                        tableHTML += '<tbody>';
                        dataInfo.sample_data.slice(0, 3).forEach(row => {
                            tableHTML += '<tr>';
                            columns.forEach(col => {
                                const value = row[col];
                                const displayValue = typeof value === 'number' ? value.toFixed(3) : value;
                                tableHTML += `<td class="border border-gray-300 px-1 py-1">${displayValue}</td>`;
                            });
                            tableHTML += '</tr>';
                        });
                        tableHTML += '</tbody></table></div></div>';
                        
                        previewHTML += tableHTML;
                    }
                    
                    // Add data type specific information
                    if (dataInfo.data_type === 'multi_level_order_book') {
                        previewHTML += `
                            <div class="mt-3 p-2 bg-green-50 rounded border border-green-200">
                                <div class="font-medium text-green-700 text-xs mb-1">🏦 Order Book Structure</div>
                                <div class="grid grid-cols-2 gap-2 text-xs">
                                    <div><span class="font-medium">Bid Levels:</span> ${dataInfo.order_book_info.bid_columns.length}</div>
                                    <div><span class="font-medium">Ask Levels:</span> ${dataInfo.order_book_info.ask_columns.length}</div>
                                    <div><span class="font-medium">Price Columns:</span> ${dataInfo.order_book_info.price_columns.length}</div>
                                    <div><span class="font-medium">Size Columns:</span> ${dataInfo.order_book_info.size_columns.length}</div>
                                </div>
                            </div>
                        `;
                    }
                    
                    previewContent.innerHTML = previewHTML;
                }
                
                // Initialize Feather icons
                feather.replace();
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def training_page(self, request):
        """Training configuration and monitoring page."""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Training - Treasury GAN Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        </head>
        <body class="bg-gray-100">
            <nav class="bg-blue-600 text-white shadow-lg">
                <div class="max-w-7xl mx-auto px-4">
                    <div class="flex justify-between items-center py-4">
                        <h1 class="text-2xl font-bold">🏦 Treasury GAN Dashboard</h1>
                        <div class="flex space-x-4">
                            <a href="/" class="hover:text-blue-200">Dashboard</a>
                            <a href="/training" class="hover:text-blue-200">Training</a>
                            <a href="/evaluation" class="hover:text-blue-200">Evaluation</a>
                            <a href="/models" class="hover:text-blue-200">Models</a>
                        </div>
                    </div>
                </div>
            </nav>
            
            <div class="max-w-7xl mx-auto px-4 py-8">
                <h2 class="text-3xl font-bold text-gray-800 mb-8">Training Configuration</h2>
                
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-xl font-semibold text-gray-700 mb-4">Model Configuration</h3>
                        <form id="model-config">
                            <div class="mb-4">
                                <label class="block text-sm font-medium text-gray-700 mb-2">GAN Type</label>
                                <select class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                                    <option value="standard">Standard GAN</option>
                                    <option value="wgan">Wasserstein GAN</option>
                                </select>
                            </div>
                            
                            <div class="mb-4">
                                <label class="block text-sm font-medium text-gray-700 mb-2">Latent Dimension</label>
                                <input type="number" value="100" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            </div>
                            
                            <div class="mb-4">
                                <label class="block text-sm font-medium text-gray-700 mb-2">Sequence Length</label>
                                <input type="number" value="100" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            </div>
                            
                            <div class="mb-4">
                                <label class="block text-sm font-medium text-gray-700 mb-2">Batch Size</label>
                                <input type="number" value="32" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            </div>
                        </form>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-xl font-semibold text-gray-700 mb-4">Training Parameters</h3>
                        <form id="training-config">
                            <div class="mb-4">
                                <label class="block text-sm font-medium text-gray-700 mb-2">Epochs</label>
                                <input type="number" value="1000" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            </div>
                            
                            <div class="mb-4">
                                <label class="block text-sm font-medium text-gray-700 mb-2">Learning Rate (Generator)</label>
                                <input type="number" step="0.0001" value="0.0002" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            </div>
                            
                            <div class="mb-4">
                                <label class="block text-sm font-medium text-gray-700 mb-2">Learning Rate (Discriminator)</label>
                                <input type="number" step="0.0001" value="0.0002" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            </div>
                            
                            <div class="mb-4">
                                <label class="block text-sm font-medium text-gray-700 mb-2">Patience</label>
                                <input type="number" value="50" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            </div>
                        </form>
                    </div>
                </div>
                
                <div class="mt-8 bg-white rounded-lg shadow-md p-6">
                    <h3 class="text-xl font-semibold text-gray-700 mb-4">Training Controls</h3>
                    <div class="flex space-x-4">
                        <button id="start-training-btn" class="bg-green-600 text-white px-8 py-3 rounded-lg hover:bg-green-700 transition-colors font-semibold">
                            Start Training
                        </button>
                        <button id="stop-training-btn" class="bg-red-600 text-white px-8 py-3 rounded-lg hover:bg-red-700 transition-colors font-semibold">
                            Stop Training
                        </button>
                        <button id="save-config-btn" class="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700 transition-colors font-semibold">
                            Save Configuration
                        </button>
                    </div>
                </div>
                
                <div class="mt-8 bg-white rounded-lg shadow-md p-6">
                    <h3 class="text-xl font-semibold text-gray-700 mb-4">Training Logs</h3>
                    <div id="training-logs" class="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm h-64 overflow-y-auto">
                        <div>Ready to start training...</div>
                    </div>
                </div>
            </div>
            
            <script>
                // Training page functionality
                document.getElementById('start-training-btn').addEventListener('click', async () => {
                    const modelConfig = document.getElementById('model-config');
                    const trainingConfig = document.getElementById('training-config');
                    
                    // Collect form data and start training
                    const config = {
                        model: {
                            gan_type: modelConfig.querySelector('select').value,
                            generator: {
                                latent_dim: parseInt(modelConfig.querySelector('input[type="number"]').value),
                                sequence_length: parseInt(modelConfig.querySelectorAll('input[type="number"]')[1].value)
                            }
                        },
                        training: {
                            epochs: parseInt(trainingConfig.querySelector('input[type="number"]').value),
                            batch_size: parseInt(modelConfig.querySelectorAll('input[type="number"]')[3].value)
                        }
                    };
                    
                    try {
                        const response = await fetch('/api/start_training', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({config: config})
                        });
                        
                        if (response.ok) {
                            document.getElementById('training-logs').innerHTML += '<div>[INFO] Training started...</div>';
                        }
                    } catch (error) {
                        document.getElementById('training-logs').innerHTML += `<div>[ERROR] ${error.message}</div>`;
                    }
                });
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def evaluation_page(self, request):
        """Model evaluation page."""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Evaluation - Treasury GAN Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        </head>
        <body class="bg-gray-100">
            <nav class="bg-blue-600 text-white shadow-lg">
                <div class="max-w-7xl mx-auto px-4">
                    <div class="flex justify-between items-center py-4">
                        <h1 class="text-2xl font-bold">🏦 Treasury GAN Dashboard</h1>
                        <div class="flex space-x-4">
                            <a href="/" class="hover:text-blue-200">Dashboard</a>
                            <a href="/training" class="hover:text-blue-200">Training</a>
                            <a href="/evaluation" class="hover:text-blue-200">Evaluation</a>
                            <a href="/models" class="hover:text-blue-200">Models</a>
                        </div>
                    </div>
                </div>
            </nav>
            
            <div class="max-w-7xl mx-auto px-4 py-8">
                <h2 class="text-3xl font-bold text-gray-800 mb-8">Model Evaluation</h2>
                
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-xl font-semibold text-gray-700 mb-4">Quality Metrics</h3>
                        <div class="space-y-3">
                            <div class="flex justify-between">
                                <span class="text-gray-600">Overall Quality:</span>
                                <span id="overall-quality" class="font-semibold">-</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-600">Mean Squared Error:</span>
                                <span id="mse" class="font-semibold">-</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-600">R² Score:</span>
                                <span id="r2-score" class="font-semibold">-</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-xl font-semibold text-gray-700 mb-4">Distribution Metrics</h3>
                        <div class="space-y-3">
                            <div class="flex justify-between">
                                <span class="text-gray-600">KS Test (avg):</span>
                                <span id="ks-test" class="font-semibold">-</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-600">Wasserstein (avg):</span>
                                <span id="wasserstein" class="font-semibold">-</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-600">JS Divergence (avg):</span>
                                <span id="js-divergence" class="font-semibold">-</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-xl font-semibold text-gray-700 mb-4">Correlation Analysis</h3>
                        <div class="space-y-3">
                            <div class="flex justify-between">
                                <span class="text-gray-600">Correlation Diff:</span>
                                <span id="corr-diff" class="font-semibold">-</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-600">Autocorrelation:</span>
                                <span id="autocorr" class="font-semibold">-</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="mt-8 bg-white rounded-lg shadow-md p-6">
                    <h3 class="text-xl font-semibold text-gray-700 mb-4">Evaluation Actions</h3>
                    <div class="flex space-x-4">
                        <button id="run-evaluation" class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                            Run Evaluation
                        </button>
                        <button id="view-plots" class="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors">
                            View Plots
                        </button>
                        <button id="export-results" class="bg-purple-600 text-white px-6 py-2 rounded-lg hover:bg-purple-700 transition-colors">
                            Export Results
                        </button>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def models_page(self, request):
        """Models and checkpoints page."""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Models - Treasury GAN Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        </head>
        <body class="bg-gray-100">
            <nav class="bg-blue-600 text-white shadow-lg">
                <div class="max-w-7xl mx-auto px-4">
                    <div class="flex justify-between items-center py-4">
                        <h1 class="text-2xl font-bold">🏦 Treasury GAN Dashboard</h1>
                        <div class="flex space-x-4">
                            <a href="/" class="hover:text-blue-200">Dashboard</a>
                            <a href="/training" class="hover:text-blue-200">Training</a>
                            <a href="/evaluation" class="hover:text-blue-200">Evaluation</a>
                            <a href="/models" class="hover:text-blue-200">Models</a>
                        </div>
                    </div>
                </div>
            </nav>
            
            <div class="max-w-7xl mx-auto px-4 py-8">
                <h2 class="text-3xl font-bold text-gray-800 mb-8">Models & Checkpoints</h2>
                
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-xl font-semibold text-gray-700 mb-4">Available Checkpoints</h3>
                        <div id="checkpoints-list" class="space-y-3">
                            <div class="text-gray-500">Loading checkpoints...</div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-xl font-semibold text-gray-700 mb-4">Model Information</h3>
                        <div id="model-info" class="text-gray-500">
                            Select a checkpoint to view model information
                        </div>
                    </div>
                </div>
                
                <div class="mt-8 bg-white rounded-lg shadow-md p-6">
                    <h3 class="text-xl font-semibold text-gray-700 mb-4">Model Actions</h3>
                    <div class="flex space-x-4">
                        <button id="load-model" class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                            Load Model
                        </button>
                        <button id="delete-model" class="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700 transition-colors">
                            Delete Model
                        </button>
                        <button id="download-model" class="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors">
                            Download Model
                        </button>
                    </div>
                </div>
            </div>
            
            <script>
                // Load checkpoints on page load
                async function loadCheckpoints() {
                    try {
                        const response = await fetch('/api/models');
                        const models = await response.json();
                        
                        const checkpointsList = document.getElementById('checkpoints-list');
                        if (models.length === 0) {
                            checkpointsList.innerHTML = '<div class="text-gray-500">No checkpoints found</div>';
                            return;
                        }
                        
                        checkpointsList.innerHTML = models.map(model => `
                            <div class="border border-gray-200 rounded-lg p-3 hover:bg-gray-50 cursor-pointer" onclick="selectModel('${model.name}')">
                                <div class="font-semibold">${model.name}</div>
                                <div class="text-sm text-gray-600">Epoch: ${model.epoch} | Loss: ${model.loss?.toFixed(4) || 'N/A'}</div>
                                <div class="text-xs text-gray-500">${model.date}</div>
                            </div>
                        `).join('');
                    } catch (error) {
                        document.getElementById('checkpoints-list').innerHTML = '<div class="text-red-500">Error loading checkpoints</div>';
                    }
                }
                
                function selectModel(modelName) {
                    // Update UI to show selected model
                    document.querySelectorAll('.border-gray-200').forEach(el => el.classList.remove('border-blue-500', 'bg-blue-50'));
                    event.target.closest('.border-gray-200').classList.add('border-blue-500', 'bg-blue-50');
                    
                    // Load model info
                    loadModelInfo(modelName);
                }
                
                async function loadModelInfo(modelName) {
                    // Load and display model information
                    document.getElementById('model-info').innerHTML = `
                        <div class="space-y-2">
                            <div><strong>Name:</strong> ${modelName}</div>
                            <div><strong>Status:</strong> Loaded</div>
                        </div>
                    `;
                }
                
                // Load checkpoints when page loads
                loadCheckpoints();
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    # API endpoints
    async def start_training(self, request):
        """Start GAN training."""
        try:
            data = await request.json()
            config_path = data.get('config', 'config/gan_config.yaml')
            data_source = data.get('data_source', 'treasury_orderbook_sample.csv')
            epochs = data.get('epochs', None)
            
            if self.training_status == "running":
                return web.json_response({"success": False, "error": "Training already in progress"})
            
            logger.info(f"Starting GAN training with config: {config_path}, data: {data_source}, epochs: {epochs}")
            
            # Start real GAN training process
            cmd = [sys.executable, "train_gan_csv.py", "--config", config_path, "--data", data_source]
            if epochs is not None:
                cmd.extend(["--epochs", str(epochs)])
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered output for real-time monitoring
                universal_newlines=True,
                cwd=os.getcwd(),  # Ensure we're in the right directory
                env=dict(os.environ, PYTHONUNBUFFERED="1")  # Force Python unbuffered output
            )
            
            self.training_status = "running"
            
            # Broadcast status update
            status_msg = {
                "type": "status_update",
                "data": {
                    "status": self.training_status,
                    "message": "Training started"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                await self.broadcast_training_update(status_msg)
                logger.info("✅ Training status update broadcasted successfully")
            except Exception as e:
                logger.warning(f"Failed to broadcast training status update: {e}")
            
            # Start monitoring thread for real-time output
            def monitor_output():
                logger.info("Starting training output monitoring...")
                try:
                    while self.training_process and self.training_process.poll() is None:
                        try:
                            line = self.training_process.stdout.readline()
                            if line:
                                line = line.strip()
                                logger.info(f"Training output (stdout): {line}")
                                
                                # Parse training metrics and broadcast via SSE
                                metrics = self.extract_training_metrics(line)
                                if metrics:
                                    logger.info(f"Extracted metrics: {metrics}")
                                    # Use a safer approach for broadcasting from threads
                                    try:
                                        loop = asyncio.get_event_loop()
                                        if loop.is_running():
                                            asyncio.run_coroutine_threadsafe(
                                                self.broadcast_training_update(metrics), 
                                                loop
                                            )
                                    except RuntimeError:
                                        # Event loop not available, log instead
                                        logger.info(f"Dashboard update (training): {metrics}")
                                
                                # Extract progress info
                                progress = self.extract_progress_info(line)
                                if progress:
                                    logger.info(f"Extracted progress: {progress}")
                                    try:
                                        loop = asyncio.get_event_loop()
                                        if loop.is_running():
                                            asyncio.run_coroutine_threadsafe(
                                                self.broadcast_progress_update(progress), 
                                                loop
                                            )
                                    except RuntimeError:
                                        logger.info(f"Dashboard update (progress): {progress}")
                                
                                # Broadcast log entry
                                log_entry = {
                                    "type": "log_entry",
                                    "data": {
                                        "message": line,
                                        "source": "training",
                                        "timestamp": datetime.now().isoformat()
                                    },
                                    "timestamp": datetime.now().isoformat()
                                }
                                try:
                                    loop = asyncio.get_event_loop()
                                    if loop.is_running():
                                        asyncio.run_coroutine_threadsafe(
                                            self.broadcast_log_update(log_entry), 
                                            loop
                                        )
                                except RuntimeError:
                                    logger.info(f"Dashboard update (log): {log_entry}")
                        
                        except Exception as e:
                            logger.error(f"Error monitoring training output: {e}")
                            break
                        
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Fatal error in training output monitor: {e}")
                    
                logger.info("Training output monitoring ended.")
                
                # Check if process completed
                if self.training_process:
                    return_code = self.training_process.poll()
                    logger.info(f"Training process completed with return code: {return_code}")
                    
                    # Read any remaining stderr output to understand why training ended
                    try:
                        remaining_stderr = self.training_process.stderr.read()
                        if remaining_stderr:
                            logger.info(f"Final stderr output: {remaining_stderr}")
                    except Exception as e:
                        logger.warning(f"Could not read final stderr: {e}")
                    
                    # Read any remaining stdout output
                    try:
                        remaining_stdout = self.training_process.stdout.read()
                        if remaining_stdout:
                            logger.info(f"Final stdout output: {remaining_stdout}")
                    except Exception as e:
                        logger.warning(f"Could not read final stdout: {e}")
                    
                    # Update training status based on return code
                    self.training_status = "completed" if return_code == 0 else "failed"
                    logger.info(f"Training status updated to: {self.training_status}")
                    
                    # More detailed logging about why training ended
                    if return_code == 0:
                        logger.info("🎉 Training completed successfully!")
                    else:
                        logger.error(f"❌ Training failed with exit code {return_code}")
                    
                    # Broadcast status update first
                    status_msg = {
                        "type": "status_update",
                        "data": {
                            "status": self.training_status,
                            "message": f"Training {self.training_status}"
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Broadcast completion status
                    completion_msg = {
                        "type": "training_complete",
                        "data": {
                            "status": "completed" if return_code == 0 else "failed",
                            "return_code": return_code,
                            "message": "Training completed successfully" if return_code == 0 else f"Training failed with exit code {return_code}"
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Store completion status for immediate broadcast and pending delivery
                    self.training_status = "completed" if return_code == 0 else "failed"
                    self.pending_completion_message = completion_msg
                    self.pending_status_message = status_msg
                    
                    logger.info(f"Dashboard update (status): {status_msg}")
                    logger.info(f"Dashboard update (completion): {completion_msg}")
                    
                    # The status will be broadcast via the main event loop when SSE clients connect
                    # or when the main thread checks for pending messages

            
            # Start monitoring in a separate thread
            monitor_thread = threading.Thread(target=monitor_output, daemon=True)
            monitor_thread.start()
            
            # Also start monitoring stderr for errors and progress
            def monitor_stderr():
                logger.info("Starting training stderr monitoring...")
                while self.training_process and self.training_process.poll() is None:
                    try:
                        line = self.training_process.stderr.readline()
                        if line:
                            line = line.strip()
                            logger.debug(f"Raw stderr line: {repr(line)}")
                            
                            # Check if this is a tqdm progress bar (not a real error)
                            if '|' in line and '%' in line and 'it/s' in line:
                                # This is a progress bar, log as info and broadcast as progress
                                logger.info(f"Training progress: {line}")
                                
                                # Extract progress percentage
                                progress_match = re.search(r'(\d+)%', line)
                                if progress_match:
                                    progress_percent = int(progress_match.group(1))
                                    
                                    # Broadcast progress update
                                    progress_entry = {
                                        "type": "progress",
                                        "epoch": 0,  # Will be updated by main monitoring
                                        "progress_percent": progress_percent,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    try:
                                        loop = asyncio.get_event_loop()
                                        if loop.is_running():
                                            asyncio.run_coroutine_threadsafe(
                                                self.broadcast_progress_update(progress_entry), 
                                                loop
                                            )
                                    except RuntimeError:
                                        logger.info(f"Dashboard update (progress): {progress_entry}")
                                
                                # Also broadcast as regular log entry
                                log_entry = {
                                    "type": "log_entry",
                                    "data": {
                                        "message": line,
                                        "source": "training_progress",
                                        "timestamp": datetime.now().isoformat()
                                    },
                                    "timestamp": datetime.now().isoformat()
                                }
                            elif line.startswith('ERROR:') or line.startswith('CRITICAL:') or 'Traceback' in line or 'Exception' in line:
                                # This is a real error, log as error
                                logger.error(f"Training error: {line}")
                                
                                # Broadcast error as log entry
                                log_entry = {
                                    "type": "log_entry",
                                    "data": {
                                        "message": line,
                                        "source": "training_error",
                                        "timestamp": datetime.now().isoformat()
                                    },
                                    "timestamp": datetime.now().isoformat()
                                }
                            else:
                                # This is just regular output, log as info
                                logger.info(f"Training output: {line}")
                                
                                # Broadcast as regular log entry
                                log_entry = {
                                    "type": "log_entry",
                                    "data": {
                                        "message": line,
                                        "source": "training_output",
                                        "timestamp": datetime.now().isoformat()
                                    },
                                    "timestamp": datetime.now().isoformat()
                                }
                            
                            # Broadcast log entry
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    asyncio.run_coroutine_threadsafe(
                                        self.broadcast_log_update(log_entry), 
                                        loop
                                    )
                            except RuntimeError:
                                logger.info(f"Dashboard update (log): {log_entry}")
                    except Exception as e:
                        logger.error(f"Error monitoring stderr: {e}")
                        break
                    time.sleep(0.1)
            
            stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
            stderr_thread.start()
            
            # Broadcast training start message
            start_msg = {
                "type": "training_start",
                "data": {
                    "status": "started",
                    "config": config_path,
                    "data_source": data_source,
                    "message": "GAN training started"
                },
                "timestamp": datetime.now().isoformat()
            }
            await self.broadcast_training_update(start_msg)
            
            logger.info(f"Training started successfully with PID: {self.training_process.pid}")
            return web.json_response({
                "success": True, 
                "message": "Training started",
                "pid": self.training_process.pid,
                "config": config_path,
                "data_source": data_source
            })
            
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return web.json_response({"success": False, "error": str(e)})
    
    async def generate_synthetic_samples(self):
        """Generate synthetic samples after training completes using the trained GAN."""
        try:
            logger.info("🎨 Generating synthetic samples after training completion...")
            
            # Check if we have a trained model checkpoint
            checkpoints_dir = Path("checkpoints")
            if not checkpoints_dir.exists():
                logger.warning("No checkpoints directory found, skipping sample generation")
                return
            
            # Find the latest checkpoint
            checkpoint_files = list(checkpoints_dir.glob("*.pth"))
            if not checkpoint_files:
                logger.warning("No checkpoint files found, skipping sample generation")
                return
            
            # Get the latest checkpoint by modification time
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Using checkpoint: {latest_checkpoint}")
            
            # Load the trained model
            from models.gan_models import create_gan_models
            
            # Load config directly from YAML
            with open("config/gan_config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            generator, discriminator = create_gan_models(config, device)
            
            # Load checkpoint
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            generator.eval()
            
            # Generate synthetic samples
            num_samples = 100  # Generate 100 synthetic samples
            with torch.no_grad():
                z = torch.randn(num_samples, config['model']['generator']['latent_dim']).to(device)
                synthetic_data = generator(z)
                
                # Convert to numpy and reshape
                synthetic_data = synthetic_data.cpu().numpy()
                
                # Reshape to match treasury data format
                if len(synthetic_data.shape) == 3:
                    synthetic_data = synthetic_data.reshape(num_samples, -1)
                
                # Create realistic column names based on treasury orderbook
                columns = [
                    'timestamp', 'bid_price_1', 'bid_size_1', 'ask_price_1', 'ask_size_1',
                    'bid_price_2', 'bid_size_2', 'ask_price_2', 'ask_size_2',
                    'bid_price_3', 'bid_size_3', 'ask_price_3', 'ask_size_3',
                    'spread_1', 'spread_2', 'spread_3'
                ]
                
                # Create synthetic data with realistic values
                synthetic_df = pd.DataFrame(synthetic_data, columns=columns[:synthetic_data.shape[1]])
                
                # Add realistic timestamps
                base_time = datetime.now()
                timestamps = [base_time + timedelta(seconds=i) for i in range(num_samples)]
                synthetic_df['timestamp'] = timestamps
                
                # Ensure prices are realistic (around 100)
                for col in synthetic_df.columns:
                    if 'price' in col:
                        synthetic_df[col] = np.clip(synthetic_df[col], 95.0, 105.0)
                    elif 'size' in col:
                        synthetic_df[col] = np.clip(synthetic_df[col], 100, 10000)
                    elif 'spread' in col:
                        synthetic_df[col] = np.clip(synthetic_df[col], 0.01, 1.0)
                
                # Save synthetic samples
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/csv/generated_sample_{timestamp_str}.csv"
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                synthetic_df.to_csv(filename, index=False)
                logger.info(f"🎨 Generated {num_samples} synthetic samples: {filename}")
                
                # Broadcast sample generation completion
                await self.broadcast_training_update({
                    "type": "sample_generation_complete",
                    "data": {
                        "filename": filename,
                        "num_samples": num_samples,
                        "message": f"Generated {num_samples} synthetic samples"
                    },
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error generating synthetic samples: {e}")
            # Broadcast error
            await self.broadcast_training_update({
                "type": "sample_generation_error",
                "data": {
                    "error": str(e),
                    "message": "Failed to generate synthetic samples"
                },
                "timestamp": datetime.now().isoformat()
            })
    
    async def stop_training(self, request):
        """Stop the current training process."""
        try:
            if self.training_process and self.training_process.poll() is None:
                logger.info("Stopping training process...")
                self.training_process.terminate()
                
                # Wait a moment for graceful termination
                try:
                    self.training_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Training process didn't terminate gracefully, forcing kill...")
                    self.training_process.kill()
                
                self.training_status = "stopped"
                logger.info("Training process stopped successfully")
                
                # Broadcast status update
                status_msg = {
                    "type": "status_update",
                    "data": {
                        "status": self.training_status,
                        "message": "Training stopped by user"
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                try:
                    await self.broadcast_training_update(status_msg)
                except Exception as e:
                    logger.warning(f"Failed to broadcast training stop status update: {e}")
                
                return web.json_response({"success": True, "message": "Training stopped"})
            else:
                return web.json_response({"success": False, "message": "No training process running"})
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return web.json_response({"success": False, "error": str(e)})
    
    # Removed get_training_status method - Training status is now only available via SSE
    
    async def get_models(self, request):
        """Get available model checkpoints."""
        try:
            checkpoints_dir = Path("checkpoints")
            if not checkpoints_dir.exists():
                return web.json_response([])
            
            models = []
            for checkpoint_file in checkpoints_dir.glob("*.pth"):
                try:
                    # Load checkpoint info
                    # Assuming torch is available for this example, otherwise remove or import
                    # import torch
                    # checkpoint = torch.load(checkpoint_file, map_location='cpu')
                    # models.append({
                    #     "name": checkpoint_file.name,
                    #     "epoch": checkpoint.get('epoch', 0),
                    #     "loss": checkpoint.get('best_loss', None),
                    #     "date": datetime.fromtimestamp(checkpoint_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    # })
                    models.append({
                        "name": checkpoint_file.name,
                        "epoch": 0, # Placeholder, actual epoch would need to be parsed
                        "loss": 0.0, # Placeholder
                        "date": datetime.fromtimestamp(checkpoint_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    })
                except:
                    continue
            
            return web.json_response(models)
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return web.json_response([])
    
    async def generate_sample(self, request):
        """Generate synthetic treasury orderbook sample with multiple levels (0-4)."""
        try:
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            import random
            
            # Treasury bond specifications with realistic symbols and random base prices
            base_price_ranges = {
                'OTR2Y': (98.5, 101.0),
                'OTR5Y': (97.5, 100.5),
                'OTR10Y': (96.5, 99.5),
                'OTR30Y': (95.5, 98.5),
                'OFR2Y': (98.0, 100.5),
                'OFR5Y': (97.0, 100.0),
                'OFR10Y': (96.0, 99.0),
                'OFR30Y': (95.0, 98.0)
            }
            
            # Generate random base prices for each bond
            treasury_bonds = []
            for symbol, (min_price, max_price) in base_price_ranges.items():
                base_price = random.uniform(min_price, max_price)
                volatility = random.uniform(0.08, 0.20)  # Random volatility for each bond
                treasury_bonds.append({
                    'symbol': symbol, 
                    'base_price': base_price, 
                    'volatility': volatility,
                    'min_price': min_price,
                    'max_price': max_price
                })
            
            # Generate 24 hours of minute-by-minute data (1440 data points)
            start_time = datetime(2024, 1, 1, 9, 0, 0)  # Market open
            timestamps = [start_time + timedelta(minutes=i) for i in range(1440)]
            
            # Market hours (9:00 AM - 4:00 PM ET)
            market_hours = [(9, 16)]  # 9 AM to 4 PM
            
            data_rows = []
            
            for timestamp in timestamps:
                # Check if within market hours
                hour = timestamp.hour
                is_market_hour = any(start <= hour <= end for start, end in market_hours)
                
                # Add some weekend gaps (Saturday = 5, Sunday = 6)
                if timestamp.weekday() >= 5:  # Weekend
                    continue
                
                # Add some holiday gaps (simplified)
                if timestamp.month == 1 and timestamp.day == 1:  # New Year
                    continue
                if timestamp.month == 7 and timestamp.day == 4:  # Independence Day
                    continue
                if timestamp.month == 12 and timestamp.day == 25:  # Christmas
                    continue
                
                for bond in treasury_bonds:
                    # Random base price variation for each timestamp (more randomness)
                    base_price_variation = random.uniform(-0.5, 0.5)
                    current_base_price = bond['base_price'] + base_price_variation
                    
                    # Enhanced random market conditions
                    market_trend = random.uniform(-0.2, 0.2)  # Random trend instead of sine wave
                    intraday_volatility = random.uniform(-0.15, 0.15)  # Random volatility instead of sine wave
                    
                    # Economic news impact (random events) - increased frequency
                    news_impact = 0
                    if random.random() < 0.005:  # 0.5% chance of major news
                        news_impact = random.uniform(-0.8, 0.8)
                    elif random.random() < 0.02:  # 2% chance of minor news
                        news_impact = random.uniform(-0.3, 0.3)
                    
                    # Fed policy expectations - more random timing
                    fed_impact = 0
                    if random.random() < 0.001:  # 0.1% chance of fed impact at any time
                        fed_impact = random.uniform(-0.5, 0.5)
                    
                    # Additional random factors
                    liquidity_impact = random.uniform(-0.1, 0.1)  # Random liquidity effects
                    technical_impact = random.uniform(-0.2, 0.2)  # Random technical levels
                    
                    # Calculate price with enhanced randomness
                    price_change = (
                        market_trend + 
                        intraday_volatility + 
                        news_impact + 
                        fed_impact +
                        liquidity_impact +
                        technical_impact +
                        random.gauss(0, bond['volatility'] * 0.02)  # Increased random walk
                    )
                    
                    new_price = current_base_price + price_change
                    # Dynamic bounds based on bond's price range
                    new_price = max(bond['min_price'] - 2.0, min(bond['max_price'] + 2.0, new_price))
                    
                    # Generate orderbook levels 0-4
                    row = {
                        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        # 'SYM': bond['symbol']
                    }
                    
                    # Level 0 (tightest spread) - more random spreads
                    spread_0 = random.uniform(0.3, 2.0)  # More variable spread
                    bid_price_0 = new_price - (spread_0 / 100) / 2
                    ask_price_0 = new_price + (spread_0 / 100) / 2
                    
                    # Add random price jitter to make it less predictable
                    bid_jitter_0 = random.uniform(-0.02, 0.02)
                    ask_jitter_0 = random.uniform(-0.02, 0.02)
                    bid_price_0 += bid_jitter_0
                    ask_price_0 += ask_jitter_0
                    
                    row.update({
                        'bid_price_1': round(bid_price_0, 4),
                        'bid_size_1': int(random.uniform(1000, 5000)),
                        'ask_price_1': round(ask_price_0, 4),
                        'ask_size_1': int(random.uniform(1000, 5000))
                    })
                    
                    # Level 1 - random distance from level 0
                    level1_distance = random.uniform(0.005, 0.08)  # Random distance
                    bid_price_1 = bid_price_0 - level1_distance
                    ask_price_1 = ask_price_0 + level1_distance
                    
                    row.update({
                        'bid_price_2': round(bid_price_1, 4),
                        'bid_size_2': int(random.uniform(2000, 8000)),
                        'ask_price_2': round(ask_price_1, 4),
                        'ask_size_2': int(random.uniform(2000, 8000))
                    })
                    
                    # Level 2 - random distance from level 1
                    level2_distance = random.uniform(0.008, 0.12)
                    bid_price_2 = bid_price_1 - level2_distance
                    ask_price_2 = ask_price_1 + level2_distance
                    
                    row.update({
                        'bid_price_3': round(bid_price_2, 4),
                        'bid_size_3': int(random.uniform(3000, 12000)),
                        'ask_price_3': round(ask_price_2, 4),
                        'ask_size_3': int(random.uniform(3000, 12000))
                    })
                    
                    # Level 3 - random distance from level 2
                    level3_distance = random.uniform(0.012, 0.18)
                    bid_price_3 = bid_price_2 - level3_distance
                    ask_price_3 = ask_price_2 + level3_distance
                    
                    row.update({
                        'bid_price_4': round(bid_price_3, 4),
                        'bid_size_4': int(random.uniform(4000, 15000)),
                        'ask_price_4': round(ask_price_3, 4),
                        'ask_size_4': int(random.uniform(4000, 15000))
                    })
                    
                    # Level 4 (widest spread) - random distance from level 3
                    level4_distance = random.uniform(0.015, 0.25)
                    bid_price_4 = bid_price_3 - level4_distance
                    ask_price_4 = ask_price_3 + level4_distance
                    
                    row.update({
                        'bid_price_5': round(bid_price_4, 4),
                        'bid_size_5': int(random.uniform(5000, 20000)),
                        'ask_price_5': round(ask_price_4, 4),
                        'ask_size_5': int(random.uniform(5000, 20000))
                    })
                    
                    # Add market microstructure data with enhanced randomness
                    base_volume = random.uniform(800000, 1200000) if is_market_hour else random.uniform(50000, 200000)
                    volume_multiplier = 1.0
                    
                    # Random volume spikes and drops
                    if random.random() < 0.01:  # 1% chance of volume spike
                        volume_multiplier = random.uniform(3.0, 5.0)
                    elif random.random() < 0.02:  # 2% chance of volume drop
                        volume_multiplier = random.uniform(0.1, 0.3)
                    else:
                        # Normal volume patterns with more randomness
                        if timestamp.hour in [9, 10, 14, 15]:  # High activity periods
                            volume_multiplier = random.uniform(1.5, 3.0)
                        elif timestamp.hour in [11, 13]:  # Lunch time
                            volume_multiplier = random.uniform(0.3, 0.7)
                        else:
                            volume_multiplier = random.uniform(0.8, 1.5)
                    
                    # Additional random volume variation
                    volume_variation = random.uniform(0.6, 1.4)
                    total_volume = int(base_volume * volume_multiplier * volume_variation)
                    
                    #row.update({
                    #    'TOTAL_VOLUME': total_volume,
                    #    'MARKET_HOUR': is_market_hour
                    #})
                    
                    data_rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(data_rows)
            
            # Save to CSV
            csv_dir = Path("data/csv")
            csv_dir.mkdir(exist_ok=True)
            
            # Generate a unique filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            #filename = f"generated_sample_{timestamp_str}.csv"
            filename = "treasury_orderbook_sample.csv"

            file_path = csv_dir / filename
            
            df.to_csv(file_path, index=False)
            
            # Generate summary statistics
            summary_stats = {
                "total_records": len(df),
                #"unique_bonds": df['SYM'].nunique(),
               # "date_range": f"{df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}",
               # "price_range": f"${df['BID_PRICE_0'].min():.2f} - ${df['ASK_PRICE_0'].max():.2f}",
               # "total_volume": f"{df['TOTAL_VOLUME'].sum():,}",
               # "orderbook_levels": "5 levels (0-4)",
               # "data_structure": "TIMESTAMP, SYM, BID_PRICE_0-4, BID_SIZE_0-4, ASK_PRICE_0-4, ASK_SIZE_0-4"
            }
            
            return web.json_response({
                "success": True, 
                "message": f"Generated treasury orderbook sample with {len(df)} records and 5 levels",
                "filename": filename,
                "file_path": str(file_path),
                "summary": summary_stats,
                "preview": df.head(10).to_dict('records'),
                "generated_file": filename  # Add this for easier frontend handling
            })
            
        except Exception as e:
            logger.error(f"Error generating sample: {e}")
            return web.json_response({"success": False, "error": str(e)})
    
    async def upload_csv(self, request):
        """Handle CSV file upload."""
        try:
            data = await request.post()
            file = data['file']
            
            # Save uploaded file
            csv_dir = Path("data/csv")
            csv_dir.mkdir(exist_ok=True)
            
            file_path = csv_dir / file.filename
            with open(file_path, 'wb') as f:
                f.write(file.file.read())
            
            # Return preview information
            return web.json_response({
                "success": True,
                "message": f"File {file.filename} uploaded successfully",
                "preview": {
                    "filename": file.filename,
                    "size": file.size,
                    "type": file.content_type,
                    "path": str(file_path)
                }
            })
            
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})
    
    async def preview_csv(self, request):
        """Preview the contents of a CSV file and analyze data structure."""
        try:
            filename = request.query.get('filename')
            if not filename:
                return web.json_response({"success": False, "error": "Filename not provided"}, status=400)
            
            csv_path = Path("data/csv") / filename
            if not csv_path.exists():
                return web.json_response({"success": False, "error": f"File not found: {filename}"}, status=404)
            
            # Import pandas here to avoid startup issues
            import pandas as pd
            import numpy as np
            
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Analyze data structure
            data_info = {
                "data_type": "time_series",  # Default type
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
                "sample_data": df.head(5).to_dict('records'),  # First 5 rows for preview
                "summary_stats": {}
            }
            
            # Add summary statistics for numeric columns
            if data_info["numeric_columns"]:
                data_info["summary_stats"] = df[data_info["numeric_columns"]].describe().to_dict()
            
            # Detect if this is multi-level order book data
            order_book_indicators = ['bid', 'ask', 'price', 'size', 'level', 'depth']
            is_order_book = any(indicator in col.lower() for col in df.columns for indicator in order_book_indicators)
            
            if is_order_book:
                data_info["data_type"] = "multi_level_order_book"
                data_info["order_book_info"] = {
                    "bid_columns": [col for col in df.columns if 'bid' in col.lower()],
                    "ask_columns": [col for col in df.columns if 'ask' in col.lower()],
                    "price_columns": [col for col in df.columns if 'price' in col.lower()],
                    "size_columns": [col for col in df.columns if 'size' in col.lower()],
                    "level_columns": [col for col in df.columns if 'level' in col.lower()]
                }
            
            # Get time series data for plotting (first 50 rows to avoid overwhelming)
            plot_data = {}
            if len(df) > 0:
                sample_size = min(50, len(df))
                sample_df = df.tail(sample_size)  # Use most recent data
                
                # Limit to first 5 numeric columns for visualization
                numeric_cols = data_info["numeric_columns"][:5]
                for col in numeric_cols:
                    plot_data[col] = sample_df[col].tolist()
                
                # Add index for x-axis
                plot_data["index"] = list(range(sample_size))
            
            logger.info(f"CSV preview generated for {filename}: {data_info['row_count']} rows, {data_info['column_count']} columns")
            
            return web.json_response({
                "success": True,
                "message": f"Preview for {filename}",
                "data_info": data_info,
                "plot_data": plot_data,
                "filename": filename
            })
            
        except Exception as e:
            logger.error(f"Error previewing CSV: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)
    
    # New endpoints for separate channels like test_separate_channels.py
    async def receive_training_data(self, request):
        """Receive training data and broadcast to training clients only."""
        try:
            data = await request.json()
            logger.info(f"🎯 Received training data: {data}")
            
            # Broadcast to training clients only
            message = f"data: {json.dumps(data)}\n\n"
            disconnected_clients = set()
            
            client_count = len(self.training_clients)
            logger.info(f"📡 Broadcasting training data to {client_count} training clients")
            
            for client in self.training_clients:
                try:
                    await client.write(message.encode())
                    logger.info(f"✅ Training data broadcasted to client {id(client)}")
                except Exception as e:
                    logger.error(f"❌ Error broadcasting training data to client {id(client)}: {e}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.training_clients -= disconnected_clients
            if disconnected_clients:
                logger.info(f"🧹 Removed {len(disconnected_clients)} disconnected training clients")
            
            final_client_count = len(self.training_clients)
            logger.info(f"📊 Training broadcast completed. Active training clients: {final_client_count}")
            
            return web.json_response({
                "status": "success", 
                "clients": final_client_count,
                "channel": "training",
                "message": f"Training data broadcast to {final_client_count} training clients"
            })
            
        except Exception as e:
            logger.error(f"❌ Error processing training data: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=400)

    async def receive_progress_data(self, request):
        """Receive progress data and broadcast to progress clients only."""
        try:
            data = await request.json()
            logger.info(f"📊 Received progress data: {data}")
            
            # Broadcast to progress clients only
            message = f"data: {json.dumps(data)}\n\n"
            disconnected_clients = set()
            
            client_count = len(self.progress_clients)
            logger.info(f"📡 Broadcasting progress data to {client_count} progress clients")
            
            for client in self.progress_clients:
                try:
                    await client.write(message.encode())
                    logger.info(f"✅ Progress data broadcasted to client {id(client)}")
                except Exception as e:
                    logger.error(f"❌ Error broadcasting progress data to client {id(client)}: {e}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.progress_clients -= disconnected_clients
            if disconnected_clients:
                logger.info(f"🧹 Removed {len(disconnected_clients)} disconnected progress clients")
            
            final_client_count = len(self.progress_clients)
            logger.info(f"📊 Progress broadcast completed. Active progress clients: {final_client_count}")
            
            return web.json_response({
                "status": "success", 
                "clients": final_client_count,
                "channel": "progress",
                "message": f"Progress data broadcast to {final_client_count} progress clients"
            })
            
        except Exception as e:
            logger.error(f"❌ Error processing progress data: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=400)

    # SSE endpoints - separate channels
    async def training_events(self, request):
        """SSE endpoint for training updates only."""
        response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Cache-Control'
            }
        )
        
        await response.prepare(request)
        self.training_clients.add(response)
        
        client_count = len(self.training_clients)
        logger.info(f"🎯 New training SSE client connected. Total training clients: {client_count}")
        
        try:
            # Send connection confirmation
            connection_msg = {
                'type': 'connection', 
                'message': 'Connected to Training SSE Channel',
                'client_id': id(response),
                'total_clients': client_count,
                'channel': 'training'
            }
            await response.write(f"data: {json.dumps(connection_msg)}\n\n".encode())
            
            # Send current training status
            status_msg = {
                'type': 'status_update',
                'data': {
                    'status': self.training_status,
                    'message': f'Current training status: {self.training_status}'
                },
                'timestamp': datetime.now().isoformat()
            }
            await response.write(f"data: {json.dumps(status_msg)}\n\n".encode())
            
            # Send pending status message if available  
            if hasattr(self, 'pending_status_message') and self.pending_status_message:
                logger.info("📤 Sending pending status message to new client")
                await response.write(f"data: {json.dumps(self.pending_status_message)}\n\n".encode())
                
            # If training is completed/failed but no pending message, send current status
            elif self.training_status in ['completed', 'failed']:
                current_status_msg = {
                    'type': 'status_update',
                    'data': {
                        'status': self.training_status,
                        'message': f'Training {self.training_status}'
                    },
                    'timestamp': datetime.now().isoformat()
                }
                logger.info(f"📤 Sending current training status to new client: {self.training_status}")
                await response.write(f"data: {json.dumps(current_status_msg)}\n\n".encode())
            
            # Send pending completion message if available
            if self.pending_completion_message:
                logger.info("📤 Sending pending completion message to new client")
                await response.write(f"data: {json.dumps(self.pending_completion_message)}\n\n".encode())
                
            # Don't clear pending messages immediately - they might be needed for other clients
            
            # Send historical training data
            historical_data = self.load_historical_logs()
            for data in historical_data:
                if data.get('type') == 'training_update':
                    await response.write(f"data: {json.dumps(data)}\n\n".encode())
                    await asyncio.sleep(0.1)  # Small delay between historical data
            
            # Keep connection alive - wait for disconnection using proper SSE pattern
            try:
                # Instead of polling, create a future that completes when cancelled
                # This is the proper asyncio pattern for long-running tasks
                disconnection_event = asyncio.Event()
                
                # Create a task to wait for disconnection
                async def wait_for_disconnection():
                    try:
                        # Try to write a keep-alive message to detect disconnection
                        while True:
                            await asyncio.sleep(30)  # Check every 30 seconds
                            await response.write(b": keep-alive\n\n")
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                        disconnection_event.set()
                
                # Start the disconnection monitor task
                monitor_task = asyncio.create_task(wait_for_disconnection())
                
                # Wait for either cancellation or disconnection
                await disconnection_event.wait()
                
                # Clean up the monitor task
                if not monitor_task.done():
                    monitor_task.cancel()
                    
            except (asyncio.CancelledError, ConnectionResetError):
                logger.debug(f"Training client {id(response)} connection cancelled")
            except Exception as e:
                logger.debug(f"Training client {id(response)} disconnected: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error with training client {id(response)}: {e}")
        finally:
            self.training_clients.discard(response)
            logger.info(f"🔌 Training client {id(response)} disconnected. Total training clients: {len(self.training_clients)}")
        
        return response

    async def progress_events(self, request):
        """SSE endpoint for progress updates only."""
        response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Cache-Control'
            }
        )
        
        await response.prepare(request)
        self.progress_clients.add(response)
        
        client_count = len(self.progress_clients)
        logger.info(f"📊 New progress SSE client connected. Total progress clients: {client_count}")
        
        try:
            # Send connection confirmation
            connection_msg = {
                'type': 'connection', 
                'message': 'Connected to Progress SSE Channel',
                'client_count': client_count,
                'channel': 'progress'
            }
            await response.write(f"data: {json.dumps(connection_msg)}\n\n".encode())
            
            # Send historical progress data
            historical_data = self.load_historical_logs()
            for data in historical_data:
                if data.get('type') == 'progress':
                    await response.write(f"data: {json.dumps(data)}\n\n".encode())
                    await asyncio.sleep(0.1)  # Small delay between historical data
            
            # Keep connection alive - wait for disconnection using proper SSE pattern
            try:
                # Instead of polling, create a future that completes when cancelled
                # This is the proper asyncio pattern for long-running tasks
                disconnection_event = asyncio.Event()
                
                # Create a task to wait for disconnection
                async def wait_for_disconnection():
                    try:
                        # Try to write a keep-alive message to detect disconnection
                        while True:
                            await asyncio.sleep(30)  # Check every 30 seconds
                            await response.write(b": keep-alive\n\n")
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                        disconnection_event.set()
                
                # Start the disconnection monitor task
                monitor_task = asyncio.create_task(wait_for_disconnection())
                
                # Wait for either cancellation or disconnection
                await disconnection_event.wait()
                
                # Clean up the monitor task
                if not monitor_task.done():
                    monitor_task.cancel()
                    
            except (asyncio.CancelledError, ConnectionResetError):
                logger.debug(f"Progress client {id(response)} connection cancelled")
            except Exception as e:
                logger.debug(f"Progress client {id(response)} disconnected: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error with progress client {id(response)}: {e}")
        finally:
            self.progress_clients.discard(response)
            logger.info(f"🔌 Progress client {id(response)} disconnected. Total progress clients: {len(self.progress_clients)}")
        
        return response

    async def log_events(self, request):
        """SSE endpoint for log updates."""
        response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
            }
        )
        
        await response.prepare(request)
        
        # Add client to set
        self.log_clients.add(response)
        
        try:
            # Send connection confirmation
            connection_msg = {
                'type': 'connection', 
                'message': 'Connected to Log SSE Channel',
                'client_id': id(response),
                'total_clients': len(self.log_clients),
                'channel': 'logs'
            }
            await response.write(f"data: {json.dumps(connection_msg)}\n\n".encode())
            
            # Send historical log data
            historical_data = self.load_historical_logs()
            for data in historical_data:
                if data.get('type') == 'log_entry':
                    await response.write(f"data: {json.dumps(data)}\n\n".encode())
                    await asyncio.sleep(0.1)  # Small delay between historical data
            
            # Keep connection alive - wait for disconnection using proper SSE pattern
            try:
                # Instead of polling, create a future that completes when cancelled
                # This is the proper asyncio pattern for long-running tasks
                disconnection_event = asyncio.Event()
                
                # Create a task to wait for disconnection
                async def wait_for_disconnection():
                    try:
                        # Try to write a keep-alive message to detect disconnection
                        while True:
                            await asyncio.sleep(30)  # Check every 30 seconds
                            await response.write(b": keep-alive\n\n")
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                        disconnection_event.set()
                
                # Start the disconnection monitor task
                monitor_task = asyncio.create_task(wait_for_disconnection())
                
                # Wait for either cancellation or disconnection
                await disconnection_event.wait()
                
                # Clean up the monitor task
                if not monitor_task.done():
                    monitor_task.cancel()
                    
            except (asyncio.CancelledError, ConnectionResetError):
                logger.debug(f"Log client {id(response)} connection cancelled")
            except Exception as e:
                logger.debug(f"Log client {id(response)} disconnected: {e}")
                    
        except asyncio.CancelledError:
            pass
        finally:
            self.log_clients.discard(response)
        
        return response
    
    async def broadcast_training_update(self, data):
        """Broadcast training update to all connected training clients."""
        if not self.training_clients:
            logger.debug("No training clients connected, skipping broadcast")
            return
        
        message = f"data: {json.dumps(data)}\n\n"
        logger.info(f"📡 Broadcasting training update to {len(self.training_clients)} clients: {data.get('type', 'unknown')}")
        
        # Create a copy of the set to avoid modification during iteration
        clients_to_remove = set()
        clients_copy = self.training_clients.copy()  # Create a copy to iterate over
        
        for client in clients_copy:
            try:
                await client.write(message.encode('utf-8'))
                logger.debug(f"✅ Training update sent to client {id(client)}")
            except Exception as e:
                logger.error(f"❌ Error sending training update to client {id(client)}: {e}")
                clients_to_remove.add(client)
        
        # Remove failed clients after iteration
        for client in clients_to_remove:
            self.training_clients.discard(client)
            logger.info(f"🧹 Removed disconnected training client {id(client)}")
        
        if clients_to_remove:
            logger.info(f"📊 Training broadcast completed. Removed {len(clients_to_remove)} clients. Active: {len(self.training_clients)}")
    
    async def broadcast_progress_update(self, data):
        """Broadcast progress update to all connected progress clients."""
        if not self.progress_clients:
            logger.debug("No progress clients connected, skipping broadcast")
            return
        
        message = f"data: {json.dumps(data)}\n\n"
        logger.info(f"📡 Broadcasting progress update to {len(self.progress_clients)} clients: {data.get('type', 'unknown')}")
        
        # Create a copy of the set to avoid modification during iteration
        clients_to_remove = set()
        clients_copy = self.progress_clients.copy()  # Create a copy to iterate over
        
        for client in clients_copy:
            try:
                await client.write(message.encode('utf-8'))
                logger.debug(f"✅ Progress update sent to client {id(client)}")
            except Exception as e:
                logger.error(f"❌ Error sending progress update to client {id(client)}: {e}")
                clients_to_remove.add(client)
        
        # Remove failed clients after iteration
        for client in clients_to_remove:
            self.progress_clients.discard(client)
            logger.info(f"🧹 Removed disconnected progress client {id(client)}")
        
        if clients_to_remove:
            logger.info(f"📊 Progress broadcast completed. Removed {len(clients_to_remove)} clients. Active: {len(self.progress_clients)}")
    
    async def broadcast_log_update(self, data):
        """Broadcast log update to all connected log clients."""
        if not self.log_clients:
            logger.debug("No log clients connected, skipping broadcast")
            return
        
        message = f"data: {json.dumps(data)}\n\n"
        logger.info(f"📡 Broadcasting log update to {len(self.log_clients)} clients: {data.get('type', 'unknown')}")
        
        # Create a copy of the set to avoid modification during iteration
        clients_to_remove = set()
        clients_copy = self.log_clients.copy()  # Create a copy to iterate over
        
        for client in clients_copy:
            try:
                await client.write(message.encode('utf-8'))
                logger.debug(f"✅ Log update sent to client {id(client)}")
            except Exception as e:
                logger.error(f"❌ Error sending log update to client {id(client)}: {e}")
                clients_to_remove.add(client)
        
        # Remove failed clients after iteration
        for client in clients_to_remove:
            self.log_clients.discard(client)
            logger.info(f"🧹 Removed disconnected log client {id(client)}")
        
        if clients_to_remove:
            logger.info(f"📊 Log broadcast completed. Removed {len(clients_to_remove)} clients. Active: {len(self.log_clients)}")
    
    async def broadcast_pending_messages_periodically(self):
        """Periodically broadcast pending status and completion messages to connected clients."""
        while True:
            try:
                await asyncio.sleep(2)  # Check every 2 seconds
                
                # Broadcast pending status message
                if hasattr(self, 'pending_status_message') and self.pending_status_message and self.training_clients:
                    logger.info("📤 Broadcasting pending status message to connected clients")
                    await self.broadcast_training_update(self.pending_status_message)
                    self.pending_status_message = None  # Clear after broadcasting
                
                # Broadcast pending completion message  
                if self.pending_completion_message and self.training_clients:
                    logger.info("📤 Broadcasting pending completion message to connected clients")
                    await self.broadcast_training_update(self.pending_completion_message)
                    self.pending_completion_message = None  # Clear after broadcasting
                    
            except Exception as e:
                logger.error(f"Error in periodic broadcast: {e}")
                await asyncio.sleep(5)  # Wait longer if there's an error
    
    async def start(self):
        """Start the dashboard server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        
        # Register with port manager for automatic discovery
        dashboard_url = register_dashboard(self.port)
        
        # Start the connection monitoring task (simplified without heartbeat checking)
        asyncio.create_task(self.cleanup_stale_connections())
        
        logger.info(f"GAN Dashboard started at {dashboard_url}")
        logger.info(f"🔧 Port configuration automatically updated for all components")
    

    
    async def cleanup_stale_connections(self):
        """Simplified connection cleanup without heartbeat checking."""
        while True:
            try:
                await asyncio.sleep(600)  # Run cleanup every 10 minutes
                
                # Basic cleanup - connections will be automatically removed when they disconnect
                logger.debug(f"📊 Current connections - Training: {len(self.training_clients)}, Progress: {len(self.progress_clients)}, Logs: {len(self.log_clients)}")
                    
            except Exception as e:
                logger.error(f"Error during connection monitoring: {e}")
    
    async def stop(self):
        """Stop the dashboard server."""
        # Cleanup port manager configuration
        cleanup_dashboard()
        
        if self.runner:
            await self.runner.cleanup()
            
        logger.info("GAN Dashboard stopped and port configuration cleaned up")
    
    async def debug_sse_connection(self, request):
        """Debug page for testing SSE connections."""
        with open('debug_sse_connection.html', 'r') as f:
            html_content = f.read()
        return web.Response(text=html_content, content_type='text/html')
    
    async def minimal_sse_test(self, request):
        """Minimal SSE test page."""
        with open('minimal_sse_test.html', 'r') as f:
            html_content = f.read()
        return web.Response(text=html_content, content_type='text/html')
    
    async def test_sse_connection_debug(self, request):
        """Debug page for testing SSE connections."""
        with open('test_sse_connection_debug.html', 'r') as f:
            html_content = f.read()
        return web.Response(text=html_content, content_type='text/html')
    
    async def test_main_dashboard_simple(self, request):
        """Simple test page for main dashboard functionality."""
        with open('test_main_dashboard_simple.html', 'r') as f:
            html_content = f.read()
        return web.Response(text=html_content, content_type='text/html')
    
    async def test_sse_connection_simple(self, request):
        """Simple SSE connection test page."""
        with open('test_sse_connection_simple.html', 'r') as f:
            html_content = f.read()
        return web.Response(text=html_content, content_type='text/html')
    
    async def debug_sse_connections(self, request):
        """Debug SSE connections page."""
        with open('debug_sse_connections.html', 'r') as f:
            html_content = f.read()
        return web.Response(text=html_content, content_type='text/html')
    
    async def test_sse_simple(self, request):
        """Simple SSE test page."""
        with open('test_sse_simple.html', 'r') as f:
            html_content = f.read()
        return web.Response(text=html_content, content_type='text/html')
    
    async def test_status_fix(self, request):
        """Test page for training status updates."""
        try:
            with open('test_status_fix.html', 'r') as f:
                html_content = f.read()
            return web.Response(text=html_content, content_type='text/html')
        except FileNotFoundError:
            return web.Response(text="Test file not found", status=404)

    async def test_training_completion(self, request):
        """Test page for training completion status updates."""
        try:
            with open('test_training_completion_ui.html', 'r') as f:
                html_content = f.read()
            return web.Response(text=html_content, content_type='text/html')
        except FileNotFoundError:
            return web.Response(text="Test file not found", status=404)

    async def test_training_complete_event(self, request):
        """Test endpoint to manually trigger a training_complete event."""
        try:
            # Create a test training_complete message
            completion_msg = {
                "type": "training_complete",
                "data": {
                    "status": "completed",
                    "return_code": 0,
                    "message": "Training completed successfully (test event)"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Broadcast the test completion message
            await self.broadcast_training_update(completion_msg)
            
            logger.info("🧪 Test training_complete event sent via SSE")
            return web.json_response({
                "success": True,
                "message": "Test training_complete event sent",
                "event": completion_msg
            })
            
        except Exception as e:
            logger.error(f"❌ Error sending test training_complete event: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

async def main():
    """Main function."""
    try:
        dashboard = GANDashboard()
        logger.info(f"🚀 Starting GAN Dashboard on port {dashboard.port}")
        
        await dashboard.start()
        
        logger.info(f"✅ Dashboard is running at http://{dashboard.host}:{dashboard.port}")
        logger.info("📊 SSE channels available:")
        logger.info(f"   - Training: http://{dashboard.host}:{dashboard.port}/events/training")
        logger.info(f"   - Progress: http://{dashboard.host}:{dashboard.port}/events/progress")
        logger.info(f"   - Logs: http://{dashboard.host}:{dashboard.port}/events/logs")
        logger.info("🔌 Press Ctrl+C to stop the dashboard")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down dashboard...")
        if 'dashboard' in locals():
            await dashboard.stop()
        logger.info("✅ Dashboard stopped successfully")
    except Exception as e:
        logger.error(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1) 