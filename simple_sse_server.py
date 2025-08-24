#!/usr/bin/env python3
"""
Simple HTTP server with Server-Sent Events (SSE) for real-time dashboard updates.
This is more reliable than WebSockets for one-way real-time data streaming.
"""

import asyncio
import json
import aiohttp
from aiohttp import web
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SSEServer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()  # Legacy clients
        self.training_clients = set()  # Training update clients
        self.progress_clients = set()  # Progress update clients
        self.app = web.Application()
        self.runner = None
        
        # Set up routes
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/events', self.events)
        self.app.router.add_get('/training_events', self.training_events)
        self.app.router.add_get('/progress_events', self.progress_events)
        self.app.router.add_post('/training_data', self.receive_training_data)
        self.app.router.add_post('/progress_data', self.receive_progress_data)
    
    async def index(self, request):
        """Main page with SSE test interface."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SSE Server Test</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .connected { background-color: #d4edda; color: #155724; }
                .disconnected { background-color: #f8d7da; color: #721c24; }
                button { padding: 10px 20px; margin: 5px; font-size: 16px; }
                #dataDiv { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>🚀 SSE Server Test Interface - Separate Channels</h1>
            
            <div>
                <button onclick="connectTraining()">🎯 Connect to Training Channel</button>
                <button onclick="connectProgress()">📊 Connect to Progress Channel</button>
                <button onclick="disconnectAll()">🔌 Disconnect All</button>
            </div>
            
            <div>
                <button onclick="sendTrainingData()">🎯 Send Training Data</button>
                <button onclick="sendProgressData()">📊 Send Progress Data</button>
                <button onclick="sendBothData()">🔄 Send Both Types</button>
            </div>
            
            <div id="trainingStatus" class="status disconnected">Training: Not connected</div>
            <div id="progressStatus" class="status disconnected">Progress: Not connected</div>
            <div id="trainingData">No training data received yet</div>
            <div id="progressData">No progress data received yet</div>
            
            <script>
                let trainingEventSource = null;
                let progressEventSource = null;
                
                function connectTraining() {
                    if (trainingEventSource) {
                        trainingEventSource.close();
                    }
                    
                    const statusDiv = document.getElementById('trainingStatus');
                    const dataDiv = document.getElementById('trainingData');
                    
                    statusDiv.textContent = 'Training: Connecting...';
                    statusDiv.className = 'status disconnected';
                    
                    trainingEventSource = new EventSource('/training_events');
                    
                    trainingEventSource.onopen = function() {
                        statusDiv.textContent = 'Training: Connected to Training Channel';
                        statusDiv.className = 'status connected';
                    };
                    
                    trainingEventSource.onmessage = function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            dataDiv.innerHTML = '<h3>Latest Training Data:</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                        } catch (e) {
                            dataDiv.innerHTML = '<h3>Raw Training Data:</h3><pre>' + event.data + '</pre>';
                        }
                    };
                    
                    trainingEventSource.onerror = function() {
                        statusDiv.textContent = 'Training: Connection error';
                        statusDiv.className = 'status disconnected';
                    };
                }
                
                function connectProgress() {
                    if (progressEventSource) {
                        progressEventSource.close();
                    }
                    
                    const statusDiv = document.getElementById('progressStatus');
                    const dataDiv = document.getElementById('progressData');
                    
                    statusDiv.textContent = 'Progress: Connecting...';
                    statusDiv.className = 'status disconnected';
                    
                    progressEventSource = new EventSource('/progress_events');
                    
                    progressEventSource.onopen = function() {
                        statusDiv.textContent = 'Progress: Connected to Progress Channel';
                        statusDiv.className = 'status connected';
                    };
                    
                    progressEventSource.onmessage = function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            dataDiv.innerHTML = '<h3>Latest Progress Data:</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                        } catch (e) {
                            dataDiv.innerHTML = '<h3>Raw Progress Data:</h3><pre>' + event.data + '</pre>';
                        }
                    };
                    
                    progressEventSource.onerror = function() {
                        statusDiv.textContent = 'Progress: Connection error';
                        statusDiv.className = 'status disconnected';
                    };
                }
                
                function disconnectAll() {
                    if (trainingEventSource) {
                        trainingEventSource.close();
                        trainingEventSource = null;
                        document.getElementById('trainingStatus').textContent = 'Training: Disconnected';
                        document.getElementById('trainingStatus').className = 'status disconnected';
                    }
                    
                    if (progressEventSource) {
                        progressEventSource.close();
                        progressEventSource = null;
                        document.getElementById('progressStatus').textContent = 'Progress: Disconnected';
                        document.getElementById('progressStatus').className = 'status disconnected';
                    }
                }
                
                function sendTrainingData() {
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
                    
                    fetch('/training_data', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(trainingData)
                    })
                    .then(response => response.json())
                    .then(data => {
                        console.log('Training data sent:', data);
                        alert('Training data sent to training channel!');
                    })
                    .catch(error => {
                        console.error('Error sending training data:', error);
                        alert('Error sending training data');
                    });
                }
                
                function sendProgressData() {
                    const progressData = {
                        type: 'progress',
                        epoch: Math.floor(Math.random() * 10) + 1,
                        progress_percent: Math.floor(Math.random() * 101),
                        timestamp: new Date().toISOString()
                    };
                    
                    fetch('/progress_data', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(progressData)
                    })
                    .then(response => response.json())
                    .then(data => {
                        console.log('Progress data sent:', data);
                        alert('Progress data sent to progress channel!');
                    })
                    .catch(error => {
                        console.error('Error sending progress data:', error);
                        alert('Error sending progress data');
                    });
                }
                
                function sendBothData() {
                    sendTrainingData();
                    setTimeout(sendProgressData, 500);
                }
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def events(self, request):
        """SSE endpoint for real-time updates."""
        response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
        await response.prepare(request)
        
        # Add client to set
        self.clients.add(response)
        client_count = len(self.clients)
        logger.info(f"🔌 Client connected. Total clients: {client_count}")
        
        try:
            # Send initial connection message
            connection_msg = {
                'type': 'connection', 
                'message': 'Connected to SSE Server',
                'client_id': id(response),
                'total_clients': client_count
            }
            await response.write(f"data: {json.dumps(connection_msg)}\n\n".encode())
            
            # Keep connection alive (no heartbeats)
            while True:
                await asyncio.sleep(1)
                # Connection stays alive without sending heartbeat data
                
        except Exception as e:
            logger.error(f"❌ Error with client {id(response)}: {e}")
        finally:
            self.clients.discard(response)
            logger.info(f"🔌 Client {id(response)} disconnected. Total clients: {len(self.clients)}")
        
        return response

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
            
            # Keep connection alive
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ Error with progress client {id(response)}: {e}")
        finally:
            self.progress_clients.discard(response)
            logger.info(f"🔌 Progress client {id(response)} disconnected. Total progress clients: {len(self.training_clients)}")
        
        return response

    async def receive_data(self, request):
        """Receive data and broadcast to all connected clients."""
        try:
            data = await request.json()
            logger.info(f"📥 Received data: {data}")
            
            # Broadcast to all connected clients
            message = f"data: {json.dumps(data)}\n\n"
            disconnected_clients = set()
            
            client_count = len(self.clients)
            logger.info(f"📡 Broadcasting to {client_count} connected clients")
            
            for client in self.clients:
                try:
                    await client.write(message.encode())
                    logger.info(f"✅ Broadcasted to client {id(client)}")
                except Exception as e:
                    logger.error(f"❌ Error broadcasting to client {id(client)}: {e}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.clients -= disconnected_clients
            if disconnected_clients:
                logger.info(f"🧹 Removed {len(disconnected_clients)} disconnected clients")
            
            final_client_count = len(self.clients)
            logger.info(f"📊 Broadcast completed. Active clients: {final_client_count}")
            
            return web.json_response({
                "status": "success", 
                "clients": final_client_count,
                "message": f"Data broadcast to {final_client_count} clients"
            })
            
        except Exception as e:
            logger.error(f"❌ Error processing data: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=400)

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
    
    async def start_server(self):
        """Start the HTTP server."""
        logger.info(f"🚀 Starting SSE server on http://{self.host}:{self.port}")
        
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            site = web.TCPSite(self.runner, self.host, self.port)
            await site.start()
            
            logger.info("✅ Server started successfully!")
            logger.info(f"📱 Open http://{self.host}:{self.port} in your browser")
            
            # Keep server running
            await asyncio.Future()  # run forever
            
        except Exception as e:
            logger.error(f"❌ Server error: {e}")
    
    async def stop_server(self):
        """Stop the HTTP server."""
        if self.runner:
            await self.runner.cleanup()
            logger.info("🔒 Server stopped")

async def main():
    """Main function."""
    server = SSEServer()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
        await server.stop_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n�� Server stopped") 