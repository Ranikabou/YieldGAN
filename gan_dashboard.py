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
from datetime import datetime
import subprocess
import signal
import psutil
import re
import glob

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GANDashboard:
    def __init__(self, host='localhost', port=8081):
        self.host = host
        self.port = port
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
        
        # Set up routes
        self.setup_routes()
        
        # Start background tasks
        self.start_background_tasks()
    
    def setup_routes(self):
        """Set up web routes."""
        # Static files (commented out for now)
        # self.app.router.add_static('/static', 'static')
        
        # Main pages
        self.app.router.add_get('/', self.dashboard)
        self.app.router.add_get('/training', self.training_page)
        self.app.router.add_get('/evaluation', self.evaluation_page)
        self.app.router.add_get('/models', self.models_page)
        
        # API endpoints
        self.app.router.add_post('/api/start_training', self.start_training)
        self.app.router.add_post('/api/stop_training', self.stop_training)
        self.app.router.add_get('/api/training_status', self.get_training_status)
        self.app.router.add_get('/api/models', self.get_models)
        self.app.router.add_post('/api/generate_sample', self.generate_sample)
        self.app.router.add_post('/api/upload_csv', self.upload_csv)
        
        # New endpoints for separate channels like test_separate_channels.py
        self.app.router.add_post('/training_data', self.receive_training_data)
        self.app.router.add_post('/progress_data', self.receive_progress_data)
        
        # SSE endpoints - separate channels
        self.app.router.add_get('/events/training', self.training_events)
        self.app.router.add_get('/events/progress', self.progress_events)
        self.app.router.add_get('/events/logs', self.log_events)
    
    def start_background_tasks(self):
        """Start background monitoring tasks."""
        async def monitor_training():
            while True:
                if self.training_process and self.training_process.poll() is None:
                    # Process is still running
                    await asyncio.sleep(5)
                else:
                    if self.training_status == "running":
                        self.training_status = "completed"
                        await self.broadcast_training_update({
                            "type": "training_completed",
                            "timestamp": datetime.now().isoformat()
                        })
                await asyncio.sleep(5)
        
        # Start log file monitoring
        asyncio.create_task(self.start_log_monitoring())
        
        asyncio.create_task(monitor_training())
    
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
        """Extract training metrics from a log line."""
        try:
            # Pattern 1: "Epoch X/Y Generator Loss: X.XXXX Discriminator Loss: X.XXXX"
            epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', line)
            gen_loss_match = re.search(r'Generator Loss:\s*([\d.]+)', line)
            disc_loss_match = re.search(r'Discriminator Loss:\s*([\d.]+)', line)
            
            if epoch_match and gen_loss_match and disc_loss_match:
                epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                gen_loss = float(gen_loss_match.group(1))
                disc_loss = float(disc_loss_match.group(1))
                
                return {
                    "type": "training_update",
                    "data": {
                        "epoch": epoch,
                        "total_epochs": total_epochs,
                        "generator_loss": gen_loss,
                        "discriminator_loss": disc_loss,
                        "real_scores": 0.8,  # Placeholder - could be extracted from logs
                        "fake_scores": 0.2   # Placeholder - could be extracted from logs
                    },
                    "timestamp": datetime.now().isoformat()
                }
            
            # Pattern 2: JSON-like format
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
        """Extract progress information from a log line."""
        try:
            # Look for percentage patterns
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
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <div class="flex items-center">
                            <div class="p-3 rounded-full bg-blue-100 text-blue-600">
                                <i data-feather="activity" class="w-6 h-6"></i>
                            </div>
                            <div class="ml-4">
                                <h3 class="text-lg font-semibold text-gray-700">Training Status</h3>
                                <p id="status-text" class="text-2xl font-bold text-blue-600">Idle</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <div class="flex items-center">
                            <div class="p-3 rounded-full bg-green-100 text-green-600">
                                <i data-feather="trending-up" class="w-6 h-6"></i>
                            </div>
                            <div class="ml-4">
                                <h3 class="text-lg font-semibold text-gray-700">Generator Loss</h3>
                                <p id="gen-loss" class="text-2xl font-bold text-green-600">-</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <div class="flex items-center">
                            <div class="p-3 rounded-full bg-red-100 text-red-600">
                                <i data-feather="trending-down" class="w-6 h-6"></i>
                            </div>
                            <div class="ml-4">
                                <h3 class="text-lg font-semibold text-gray-700">Discriminator Loss</h3>
                                <p id="disc-loss" class="text-2xl font-bold text-red-600">-</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <div class="flex items-center">
                            <div class="p-3 rounded-full bg-purple-100 text-purple-600">
                                <i data-feather="clock" class="w-6 h-6"></i>
                            </div>
                            <div class="ml-4">
                                <h3 class="text-lg font-semibold text-gray-700">Epoch</h3>
                                <p id="current-epoch" class="text-2xl font-bold text-purple-600">-</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-lg font-semibold text-gray-700 mb-4">Training Progress</h3>
                        <canvas id="trainingChart" width="400" height="200"></canvas>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-lg font-semibold text-gray-700 mb-4">Real vs Synthetic Scores</h3>
                        <canvas id="scoresChart" width="400" height="200"></canvas>
                    </div>
                </div>
                
                <div class="mt-8 bg-white rounded-lg shadow-md p-6">
                    <h3 class="text-lg font-semibold text-gray-700 mb-4">Quick Actions</h3>
                    <div class="flex space-x-4">
                        <button id="start-training" class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                            Start Training
                        </button>
                        <button id="stop-training" class="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700 transition-colors">
                            Stop Training
                        </button>
                        <button id="generate-sample" class="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors">
                            Generate Sample
                        </button>
                        <button id="upload-csv" class="bg-purple-600 text-white px-6 py-2 rounded-lg hover:bg-purple-700 transition-colors">
                            Upload CSV
                        </button>
                    </div>
                </div>
                
                <!-- New section for testing separate SSE channels -->
                <div class="mt-8 bg-white rounded-lg shadow-md p-6">
                    <h3 class="text-lg font-semibold text-gray-700 mb-4">Test Separate SSE Channels</h3>
                    <p class="text-gray-600 mb-4">Test the separate channels for training and progress data, similar to test_separate_channels.py</p>
                    
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="border border-gray-200 rounded-lg p-4">
                            <h4 class="font-semibold text-gray-700 mb-3">🎯 Training Channel</h4>
                            <p class="text-sm text-gray-600 mb-3">Send training metrics to training clients only</p>
                            <button onclick="sendTrainingData()" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors">
                                Send Training Data
                            </button>
                        </div>
                        
                        <div class="border border-gray-200 rounded-lg p-4">
                            <h4 class="font-semibold text-gray-700 mb-3">📊 Progress Channel</h4>
                            <p class="text-sm text-gray-600 mb-3">Send progress updates to progress clients only</p>
                            <button onclick="sendProgressData()" class="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition-colors">
                                Send Progress Data
                            </button>
                        </div>
                    </div>
                    
                    <div class="mt-4 p-3 bg-gray-50 rounded-lg">
                        <h5 class="font-semibold text-gray-700 mb-2">SSE Channel Status</h5>
                        <div class="text-sm text-gray-600">
                            <div>🎯 Training Channel: <span id="training-status" class="font-mono">Connecting...</span></div>
                            <div>📊 Progress Channel: <span id="progress-status" class="font-mono">Connecting...</span></div>
                            <div>📝 Log Channel: <span id="log-status" class="font-mono">Connecting...</span></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Initialize charts
                const trainingCtx = document.getElementById('trainingChart').getContext('2d');
                const scoresCtx = document.getElementById('scoresChart').getContext('2d');
                
                const trainingChart = new Chart(trainingCtx, {
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
                
                const scoresChart = new Chart(scoresCtx, {
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
                
                // Connect to separate SSE channels like test_separate_channels.py
                let trainingEventSource = null;
                let progressEventSource = null;
                let logEventSource = null;
                
                function connectTrainingChannel() {
                    if (trainingEventSource) {
                        trainingEventSource.close();
                    }
                    
                    trainingEventSource = new EventSource('/events/training');
                    
                    trainingEventSource.onopen = function() {
                        console.log('🎯 Connected to Training SSE Channel');
                        document.getElementById('training-status').textContent = 'Connected';
                    };
                    
                    trainingEventSource.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        console.log('🎯 Training data received:', data);
                        
                        if (data.type === 'training_update') {
                            updateDashboard(data);
                        } else if (data.type === 'connection') {
                            console.log('🎯 Training channel connected:', data.message);
                        }
                    };
                    
                    trainingEventSource.onerror = function() {
                        console.error('🎯 Training channel connection error');
                        document.getElementById('training-status').textContent = 'Disconnected';
                    };
                }
                
                function connectProgressChannel() {
                    if (progressEventSource) {
                        progressEventSource.close();
                    }
                    
                    progressEventSource = new EventSource('/events/progress');
                    
                    progressEventSource.onopen = function() {
                        console.log('📊 Connected to Progress SSE Channel');
                        document.getElementById('progress-status').textContent = 'Connected';
                    };
                    
                    progressEventSource.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        console.log('📊 Progress data received:', data);
                        
                        if (data.type === 'progress') {
                            updateProgress(data);
                        } else if (data.type === 'connection') {
                            console.log('📊 Progress channel connected:', data.message);
                        }
                    };
                    
                    progressEventSource.onerror = function() {
                        console.error('📊 Progress channel connection error');
                        document.getElementById('progress-status').textContent = 'Disconnected';
                    };
                }
                
                function connectLogChannel() {
                    if (logEventSource) {
                        logEventSource.close();
                    }
                    
                    logEventSource = new EventSource('/events/logs');
                    
                    logEventSource.onopen = function() {
                        console.log('📝 Connected to Log SSE Channel');
                        document.getElementById('log-status').textContent = 'Connected';
                    };
                    
                    logEventSource.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        console.log('📝 Log data received:', data);
                        
                        if (data.type === 'log_entry') {
                            updateLogs(data);
                        } else if (data.type === 'connection') {
                            console.log('📝 Log channel connected:', data.message);
                        }
                    };
                    
                    logEventSource.onerror = function() {
                        console.error('📝 Log channel connection error');
                        document.getElementById('log-status').textContent = 'Disconnected';
                    };
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
                
                // Test functions for sending data to separate channels
                async function sendTrainingData() {
                    const trainingData = {
                        type: 'training_update',
                        data: {
                            epoch: Math.floor(Math.random() * 10) + 1,
                            total_epochs: 10,
                            generator_loss: (Math.random() * 0.7 + 0.5).toFixed(4),
                            discriminator_loss: (Math.random() * 0.5 + 0.6).toFixed(4),
                            real_scores: (Math.random() * 0.5 + 0.4).toFixed(4),
                            fake_scores: (Math.random() * 0.4 + 0.1).toFixed(4)
                        },
                        timestamp: new Date().toISOString()
                    };
                    
                    try {
                        const response = await fetch('/training_data', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify(trainingData)
                        });
                        
                        const result = await response.json();
                        console.log('🎯 Training data sent:', result);
                        alert(`Training data sent to training channel! Clients: ${result.clients}`);
                    } catch (error) {
                        console.error('Error sending training data:', error);
                        alert('Error sending training data');
                    }
                }
                
                async function sendProgressData() {
                    const progressData = {
                        type: 'progress',
                        epoch: Math.floor(Math.random() * 10) + 1,
                        progress_percent: Math.floor(Math.random() * 101),
                        timestamp: new Date().toISOString()
                    };
                    
                    try {
                        const response = await fetch('/progress_data', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify(progressData)
                        });
                        
                        const result = await response.json();
                        console.log('📊 Progress data sent:', result);
                        alert(`Progress data sent to progress channel! Clients: ${result.clients}`);
                    } catch (error) {
                        console.error('Error sending progress data:', error);
                        alert('Error sending progress data');
                    }
                }
                
                function updateDashboard(data) {
                    if (data.type === 'training_update') {
                        document.getElementById('gen-loss').textContent = data.data.generator_loss.toFixed(4);
                        document.getElementById('disc-loss').textContent = data.data.discriminator_loss.toFixed(4);
                        document.getElementById('current-epoch').textContent = data.data.epoch;
                        
                        // Update charts
                        trainingChart.data.labels.push(data.data.epoch);
                        trainingChart.data.datasets[0].data.push(data.data.generator_loss);
                        trainingChart.data.datasets[1].data.push(data.data.discriminator_loss);
                        trainingChart.update();
                        
                        scoresChart.data.labels.push(data.data.epoch);
                        scoresChart.data.datasets[0].data.push(data.data.real_scores);
                        scoresChart.data.datasets[1].data.push(data.data.fake_scores);
                        scoresChart.update();
                    }
                }
                
                function updateProgress(data) {
                    // Update progress indicators
                    console.log('📊 Progress update:', data);
                }
                
                function updateLogs(data) {
                    // Update log display
                    console.log('📝 Log update:', data);
                }
                
                // Connect to all channels on page load
                window.addEventListener('load', function() {
                    connectTrainingChannel();
                    connectProgressChannel();
                    connectLogChannel();
                });
                
                // Button event listeners
                document.getElementById('start-training').addEventListener('click', async () => {
                    try {
                        const response = await fetch('/api/start_training', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({config: 'config/gan_config.yaml'})
                        });
                        const result = await response.json();
                        if (result.success) {
                            document.getElementById('status-text').textContent = 'Running';
                            document.getElementById('status-text').className = 'text-2xl font-bold text-green-600';
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
                        }
                    } catch (error) {
                        console.error('Error stopping training:', error);
                    }
                });
                
                document.getElementById('generate-sample').addEventListener('click', async () => {
                    try {
                        const response = await fetch('/api/generate_sample', {method: 'POST'});
                        const result = await response.json();
                        if (result.success) {
                            alert('Sample generated successfully!');
                        }
                    } catch (error) {
                        console.error('Error generating sample:', error);
                    }
                });
                
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
            
            if self.training_status == "running":
                return web.json_response({"success": False, "error": "Training already in progress"})
            
            # Start training process
            cmd = [sys.executable, "train_gan.py", "--config", config_path]
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.training_status = "running"
            
            # Start monitoring thread
            def monitor_output():
                while self.training_process and self.training_process.poll() is None:
                    line = self.training_process.stdout.readline()
                    if line:
                        # Parse training metrics and broadcast
                        if "Epoch" in line and "Generator Loss" in line:
                            try:
                                # Extract metrics from log line
                                parts = line.split()
                                epoch = int(parts[1].split('/')[0])
                                gen_loss = float(parts[4])
                                disc_loss = float(parts[7])
                                
                                metrics = {
                                    "type": "training_update",
                                    "data": {
                                        "epoch": epoch,
                                        "generator_loss": gen_loss,
                                        "discriminator_loss": disc_loss,
                                        "real_scores": 0.8,  # Placeholder
                                        "fake_scores": 0.2   # Placeholder
                                    },
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                asyncio.create_task(self.broadcast_training_update(metrics))
                            except:
                                pass
                    
                    time.sleep(0.1)
            
            threading.Thread(target=monitor_output, daemon=True).start()
            
            return web.json_response({"success": True, "message": "Training started"})
            
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return web.json_response({"success": False, "error": str(e)})
    
    async def stop_training(self, request):
        """Stop GAN training."""
        try:
            if self.training_process:
                self.training_process.terminate()
                self.training_process.wait(timeout=5)
                self.training_process = None
            
            self.training_status = "stopped"
            return web.json_response({"success": True, "message": "Training stopped"})
            
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return web.json_response({"success": False, "error": str(e)})
    
    async def get_training_status(self, request):
        """Get current training status."""
        return web.json_response({
            "status": self.training_status,
            "metrics": self.training_metrics
        })
    
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
        """Generate synthetic sample."""
        try:
            # This would integrate with the trained model
            return web.json_response({"success": True, "message": "Sample generated"})
        except Exception as e:
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
            
            return web.json_response({"success": True, "message": f"File {file.filename} uploaded successfully"})
            
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})
    
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
            
            # Send historical training data
            historical_data = self.load_historical_logs()
            for data in historical_data:
                if data.get('type') == 'training_update':
                    await response.write(f"data: {json.dumps(data)}\n\n".encode())
                    await asyncio.sleep(0.1)  # Small delay between historical data
            
            # Keep connection alive
            while True:
                await asyncio.sleep(1)
                
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
            
            # Keep connection alive
            while True:
                await asyncio.sleep(1)
                
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
            
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self.log_clients.discard(response)
        
        return response
    
    async def broadcast_training_update(self, data):
        """Broadcast training update to all connected clients."""
        message = f"data: {json.dumps(data)}\n\n"
        
        # Create a copy of the set to avoid modification during iteration
        clients_to_remove = set()
        for client in self.training_clients:
            try:
                await client.write(message.encode('utf-8'))
            except:
                clients_to_remove.add(client)
        
        # Remove failed clients after iteration
        for client in clients_to_remove:
            self.training_clients.discard(client)
    
    async def broadcast_progress_update(self, data):
        """Broadcast progress update to all connected clients."""
        message = f"data: {json.dumps(data)}\n\n"
        
        for client in self.progress_clients:
            try:
                await client.write(message.encode('utf-8'))
            except:
                self.progress_clients.discard(client)
    
    async def broadcast_log_update(self, data):
        """Broadcast log update to all connected clients."""
        message = f"data: {json.dumps(data)}\n\n"
        
        for client in self.log_clients:
            try:
                await client.write(message.encode('utf-8'))
            except:
                self.log_clients.discard(client)
    
    async def start(self):
        """Start the dashboard server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        
        logger.info(f"GAN Dashboard started at http://{self.host}:{self.port}")
    
    async def stop(self):
        """Stop the dashboard server."""
        if self.runner:
            await self.runner.cleanup()

async def main():
    """Main function."""
    dashboard = GANDashboard()
    
    try:
        await dashboard.start()
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard...")
        await dashboard.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user") 