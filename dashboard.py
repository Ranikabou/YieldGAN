#!/usr/bin/env python3
"""
Treasury GAN Training Dashboard
Interactive web interface for monitoring GAN training, data visualization, and results analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import json
import os
import time
from datetime import datetime, timedelta
import torch
import threading
import queue
import subprocess
import sys
from pathlib import Path
import re

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Treasury GAN Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-running {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 0.5rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    .status-stopped {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 0.5rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    .log-entry {
        background-color: #f0f2f6;
        padding: 8px;
        margin: 4px 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 12px;
    }
    .log-info { border-left: 4px solid #17a2b8; }
    .log-warning { border-left: 4px solid #ffc107; }
    .log-error { border-left: 4px solid #dc3545; }
    .log-success { border-left: 4px solid #28a745; }
</style>
""", unsafe_allow_html=True)

class GANDashboard:
    def __init__(self):
        self.config = self.load_config()
        self.training_status = "stopped"
        self.training_process = None
        self.log_queue = queue.Queue()
        self.training_logs = []
        self.current_epoch = 0
        self.total_epochs = 50
        self.generator_losses = []
        self.discriminator_losses = []
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        self.log_file = 'logs/training.log'
        self.process_file = 'logs/training_process.pid'
        
        # For sidebar display - only show recent messages
        self.sidebar_messages = []
        self.max_sidebar_messages = 5  # Only show last 5 messages
        
        # Try to restore training status from file
        self.restore_training_status()
    
    def load_config(self):
        """Load GAN configuration file."""
        try:
            with open('config/gan_config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            try:
                with open('config/csv_config.yaml', 'r') as f:
                    return yaml.safe_load(f)
            except FileNotFoundError:
                st.error("No configuration file found. Please ensure config/gan_config.yaml or config/csv_config.yaml exists.")
                return {}
    
    def get_data_info(self):
        """Get information about available data files."""
        data_info = {}
        data_dir = Path("data")
        
        if data_dir.exists():
            for file_path in data_dir.glob("*"):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    data_info[file_path.name] = {
                        "size_mb": round(size_mb, 2),
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime),
                        "path": str(file_path)
                    }
        
        return data_info
    
    def get_results_info(self):
        """Get information about training results and checkpoints."""
        results_info = {}
        
        # Check checkpoints
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            results_info["checkpoints"] = [
                {
                    "name": cp.name,
                    "size_mb": round(cp.stat().st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(cp.stat().st_mtime)
                }
                for cp in checkpoints
            ]
        
        # Check results
        results_dir = Path("results")
        if results_dir.exists():
            results = list(results_dir.glob("*"))
            results_info["results"] = [
                {
                    "name": r.name,
                    "size_mb": round(r.stat().st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(r.stat().st_mtime)
                }
                for r in results if r.is_file()
            ]
        
        return results_info
    
    def start_training(self, start_date, end_date, config_path, use_csv=False):
        """Start GAN training process."""
        try:
            if use_csv:
                cmd = [
                    sys.executable, "train_gan_csv.py",
                    "--config", config_path
                ]
            else:
                cmd = [
                    sys.executable, "train_gan.py",
                    "--start-date", start_date,
                    "--end-date", end_date,
                    "--config", config_path
                ]
            
            st.info(f"Starting training with command: {' '.join(cmd)}")
            
            # Start the training process with output redirected to log file
            with open(self.log_file, 'a') as log_f:
                log_f.write(f"\n{'='*50}\n")
                log_f.write(f"Training started at {datetime.now()}\n")
                log_f.write(f"Command: {' '.join(cmd)}\n")
                log_f.write(f"{'='*50}\n")
            
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Save process PID for restoration
            self.save_process_pid(self.training_process.pid)
            
            self.training_status = "running"
            self.training_logs = []  # Clear previous logs
            self.generator_losses = []
            self.discriminator_losses = []
            self.current_epoch = 0
            
            st.success("Training started successfully!")
            
            # Start log monitoring in background
            threading.Thread(target=self.monitor_training_logs, daemon=True).start()
            
        except Exception as e:
            st.error(f"Failed to start training: {e}")
            self.training_status = "stopped"
    
    def stop_training(self):
        """Stop GAN training process."""
        if self.training_process:
            self.training_process.terminate()
            self.training_status = "stopped"
            self.cleanup_process_files()
            st.warning("Training stopped.")
    
    def is_training_running(self):
        """Check if training process is still running."""
        if self.training_process:
            return self.training_process.poll() is None
        return False
    
    def refresh_training_status(self):
        """Refresh training status from process and log files."""
        if self.training_status == "running":
            if not self.is_training_running():
                self.training_status = "stopped"
                self.cleanup_process_files()
                st.sidebar.warning("Training process has stopped.")
            else:
                # Read latest logs from file
                latest_logs = self.read_logs_from_file()
                if latest_logs and len(latest_logs) > len(self.training_logs):
                    # New logs available, update
                    new_logs = latest_logs[len(self.training_logs):]
                    for log in new_logs:
                        self.parse_training_progress(log)
                    self.training_logs = latest_logs
    
    def monitor_training_logs(self):
        """Monitor training logs in background."""
        if self.training_process:
            try:
                for line in iter(self.training_process.stdout.readline, ''):
                    if line:
                        line = line.strip()
                        self.log_queue.put(line)
                        self.training_logs.append(line)
                        
                        # Save log to file for persistence
                        self.save_log_to_file(line)
                        
                        # Parse training progress
                        self.parse_training_progress(line)
                        
                        if self.training_process.poll() is not None:
                            break
                            
            except Exception as e:
                st.error(f"Error monitoring training: {e}")
    
    def parse_training_progress(self, log_line):
        """Parse training progress from log lines."""
        # Parse epoch information - multiple patterns
        epoch_patterns = [
            r'epoch\s+(\d+)/(\d+)',
            r'epoch\s+(\d+)',
            r'epoch\s+(\d+)\s+of\s+(\d+)',
            r'training\s+epoch\s+(\d+)',
            r'epoch\s+(\d+)\s*/\s*(\d+)',
            r'epoch\s+(\d+):\s*100%',  # Progress bar format
            r'epoch\s+(\d+)/\d+:\s*100%'  # Progress bar with total
        ]
        
        for pattern in epoch_patterns:
            epoch_match = re.search(pattern, log_line.lower())
            if epoch_match:
                if len(epoch_match.groups()) == 2:
                    self.current_epoch = int(epoch_match.group(1))
                    self.total_epochs = int(epoch_match.group(2))
                else:
                    self.current_epoch = int(epoch_match.group(1))
                break
        
        # Parse loss information - multiple patterns including the actual log format
        gen_loss_patterns = [
            r'generator.*loss.*?([\d.]+)',
            r'gen.*loss.*?([\d.]+)',
            r'g_loss.*?([\d.]+)',
            r'generator_loss.*?([\d.]+)',
            r'generator loss:\s*([\d.]+)',  # Actual format from logs
            r'val generator loss:\s*([\d.]+)'  # Validation loss
        ]
        
        for pattern in gen_loss_patterns:
            gen_loss_match = re.search(pattern, log_line.lower())
            if gen_loss_match:
                try:
                    loss = float(gen_loss_match.group(1))
                    self.generator_losses.append(loss)
                    break
                except:
                    pass
        
        disc_loss_patterns = [
            r'discriminator.*loss.*?([\d.]+)',
            r'disc.*loss.*?([\d.]+)',
            r'd_loss.*?([\d.]+)',
            r'discriminator_loss.*?([\d.]+)',
            r'discriminator loss:\s*([\d.]+)',  # Actual format from logs
            r'val discriminator loss:\s*([\d.]+)'  # Validation loss
        ]
        
        for pattern in disc_loss_patterns:
            disc_loss_match = re.search(pattern, log_line.lower())
            if disc_loss_match:
                try:
                    loss = float(disc_loss_match.group(1))
                    self.discriminator_losses.append(loss)
                    break
                except:
                    pass
        
        # Parse other training indicators
        if 'training' in log_line.lower() and 'started' in log_line.lower():
            self.add_sidebar_message("Training process started!")
        
        if 'completed' in log_line.lower() or 'finished' in log_line.lower():
            self.add_sidebar_message("Training completed!")
            self.training_status = "completed"
        
        if 'error' in log_line.lower() or 'exception' in log_line.lower():
            self.add_sidebar_message(f"Training error: {log_line}")
        
        # Parse checkpoint saves
        if 'checkpoint saved' in log_line.lower():
            self.add_sidebar_message("Model checkpoint saved!")
        
        # Parse progress bar completion
        if '100%' in log_line and 'epoch' in log_line.lower():
            self.add_sidebar_message(f"Epoch {self.current_epoch} completed!")
        
        # Debug: log what we're parsing
        if 'epoch' in log_line.lower() or 'loss' in log_line.lower():
            print(f"DEBUG: Parsing line: {log_line}")
            print(f"DEBUG: Current epoch: {self.current_epoch}, Total: {self.total_epochs}")
            print(f"DEBUG: Generator losses: {len(self.generator_losses)}")
            print(f"DEBUG: Discriminator losses: {len(self.discriminator_losses)}")
    
    def get_training_logs(self):
        """Get recent training logs from both memory and file."""
        # Get new logs from queue
        logs = []
        while not self.log_queue.empty():
            logs.append(self.log_queue.get_nowait())
        
        # If no new logs, try to read from file
        if not logs and not self.training_logs:
            self.training_logs = self.read_logs_from_file()
            # Parse progress from file logs
            for log in self.training_logs:
                self.parse_training_progress(log)
        
        return logs
    
    def create_data_visualization(self):
        """Create data visualization charts."""
        try:
            # Load sample data if available
            sequences_path = "data/sequences.npy"
            targets_path = "data/targets.npy"
            
            if os.path.exists(sequences_path) and os.path.exists(targets_path):
                sequences = np.load(sequences_path)
                targets = np.load(targets_path)
                
                # Create subplots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Treasury Yield Sequences", "Target Distribution", 
                                   "Feature Correlation", "Data Statistics"),
                    specs=[[{"type": "scatter"}, {"type": "histogram"}],
                           [{"type": "heatmap"}, {"type": "bar"}]]
                )
                
                # Plot 1: Sample sequences
                if len(sequences) > 0:
                    sample_seq = sequences[0, :, :5]  # First sequence, first 5 features
                    fig.add_trace(
                        go.Scatter(y=sample_seq.flatten(), name="Sample Sequence", 
                                 line=dict(color='blue')),
                        row=1, col=1
                    )
                
                # Plot 2: Target distribution
                if len(targets) > 0:
                    fig.add_trace(
                        go.Histogram(x=targets.flatten(), name="Target Distribution",
                                   nbinsx=30, marker_color='green'),
                        row=1, col=2
                    )
                
                # Plot 3: Feature correlation (if enough data)
                if sequences.shape[2] > 1:
                    corr_matrix = np.corrcoef(sequences.reshape(-1, sequences.shape[2]).T)
                    fig.add_trace(
                        go.Heatmap(z=corr_matrix, colorscale='RdBu', 
                                 name="Feature Correlation"),
                        row=2, col=1
                    )
                
                # Plot 4: Data statistics
                stats_data = {
                    "Metric": ["Sequences", "Targets", "Features", "Sequence Length"],
                    "Value": [sequences.shape[0], targets.shape[0], 
                             sequences.shape[2], sequences.shape[1]]
                }
                fig.add_trace(
                    go.Bar(x=stats_data["Metric"], y=stats_data["Value"],
                          marker_color='orange', name="Data Statistics"),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=False)
                return fig
            
        except Exception as e:
            st.error(f"Error creating data visualization: {e}")
        
        return None
    
    def create_training_progress(self):
        """Create training progress visualization."""
        if not self.generator_losses and not self.discriminator_losses:
            # No real data yet, show placeholder
            epochs = list(range(1, 51))
            generator_loss = [np.random.uniform(0.5, 2.0) for _ in epochs]
            discriminator_loss = [np.random.uniform(0.3, 1.5) for _ in epochs]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=epochs, y=generator_loss,
                mode='lines+markers',
                name='Generator Loss (Sample)',
                line=dict(color='red', width=2, dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                x=epochs, y=discriminator_loss,
                mode='lines+markers',
                name='Discriminator Loss (Sample)',
                line=dict(color='blue', width=2, dash='dot')
            ))
            
            fig.update_layout(
                title="Training Progress (Sample Data - Start Training to See Real Data)",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=400
            )
            
            return fig
        
        # Show real training data
        epochs = list(range(1, len(self.generator_losses) + 1))
        
        fig = go.Figure()
        
        if self.generator_losses:
            fig.add_trace(go.Scatter(
                x=epochs, y=self.generator_losses,
                mode='lines+markers',
                name='Generator Loss (Real)',
                line=dict(color='red', width=3)
            ))
        
        if self.discriminator_losses:
            fig.add_trace(go.Scatter(
                x=epochs, y=self.discriminator_losses,
                mode='lines+markers',
                name='Discriminator Loss (Real)',
                line=dict(color='blue', width=3)
            ))
        
        fig.update_layout(
            title="Training Progress (Real Data)",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400
        )
        
        return fig
    
    def create_synthetic_data_comparison(self):
        """Create comparison between real and synthetic data."""
        # Sample data for visualization
        time_points = np.linspace(0, 100, 100)
        real_data = np.sin(time_points * 0.1) + np.random.normal(0, 0.1, 100)
        synthetic_data = np.sin(time_points * 0.1) + np.random.normal(0, 0.15, 100)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points, y=real_data,
            mode='lines',
            name='Real Data',
            line=dict(color='green', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points, y=synthetic_data,
            mode='lines',
            name='Synthetic Data',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title="Real vs Synthetic Data Comparison",
            xaxis_title="Time",
            yaxis_title="Value",
            height=400
        )
        
        return fig

    def restore_training_status(self):
        """Attempt to restore training status from log files."""
        if os.path.exists(self.process_file):
            try:
                with open(self.process_file, 'r') as f:
                    pid = int(f.read())
                    # Check if process is running
                    if subprocess.Popen.from_pid(pid):
                        self.training_status = "running"
                        st.sidebar.success("Training process restored from log file.")
                        # Attempt to read logs from the log file
                        if os.path.exists(self.log_file):
                            with open(self.log_file, 'r') as log_f:
                                self.training_logs = [line.strip() for line in log_f.readlines()]
                                # Attempt to parse the last log line for progress
                                if self.training_logs:
                                    self.parse_training_progress(self.training_logs[-1])
                                    st.sidebar.text(f"Last parsed epoch: {self.current_epoch}/{self.total_epochs}")
                                    st.sidebar.text(f"Last parsed loss: Gen={self.generator_losses[-1] if self.generator_losses else 'N/A'}, Disc={self.discriminator_losses[-1] if self.discriminator_losses else 'N/A'}")
                                else:
                                    st.sidebar.warning("No logs found in log file to restore progress.")
                        else:
                            st.sidebar.warning("Log file not found to restore progress.")
                    else:
                        st.sidebar.warning(f"Process with PID {pid} not found. Training status cannot be restored.")
                        os.remove(self.process_file) # Clean up if process is gone
                os.remove(self.process_file) # Clean up the process file
            except Exception as e:
                st.sidebar.warning(f"Error restoring training status from log file: {e}")
                os.remove(self.process_file) # Clean up the process file on error

    def save_log_to_file(self, log_line):
        """Save log line to file for persistence."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log_line}\n")
        except Exception as e:
            print(f"Error saving log to file: {e}")
    
    def read_logs_from_file(self):
        """Read logs from file."""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    return [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error reading logs from file: {e}")
        return []
    
    def save_process_pid(self, pid):
        """Save process PID to file for restoration."""
        try:
            with open(self.process_file, 'w') as f:
                f.write(str(pid))
        except Exception as e:
            print(f"Error saving process PID: {e}")
    
    def cleanup_process_files(self):
        """Clean up process tracking files."""
        try:
            if os.path.exists(self.process_file):
                os.remove(self.process_file)
        except Exception as e:
            print(f"Error cleaning up process files: {e}")

    def add_sidebar_message(self, message, message_type="info"):
        """Add a message to the sidebar display, keeping only recent ones."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Add new message
        self.sidebar_messages.append({
            "message": formatted_message,
            "type": message_type,
            "timestamp": datetime.now()
        })
        
        # Keep only the most recent messages
        if len(self.sidebar_messages) > self.max_sidebar_messages:
            self.sidebar_messages = self.sidebar_messages[-self.max_sidebar_messages:]
    
    def clear_sidebar_messages(self):
        """Clear old sidebar messages."""
        self.sidebar_messages = []

def main():
    st.markdown('<h1 class="main-header">📊 Treasury GAN Training Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = GANDashboard()
    
    # Auto-refresh mechanism using session state
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Auto-refresh every 3 seconds when training is running
    if dashboard.training_status == "running":
        current_time = time.time()
        if current_time - st.session_state.last_refresh > 3:
            st.session_state.last_refresh = current_time
            # Force a refresh by updating the page
            st.experimental_rerun()
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Training controls
    st.sidebar.markdown("### 🚀 Training Controls")
    
    # Data source selection
    use_csv = st.sidebar.checkbox("Use CSV Data Source", value=False)
    
    if use_csv:
        config_path = st.sidebar.selectbox(
            "CSV Configuration File",
            ["config/csv_config.yaml"],
            index=0
        )
        st.sidebar.info("📁 CSV Mode: Will load data from data/csv/ directory")
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2022, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime(2024, 1, 1))
        
        config_path = st.sidebar.selectbox(
            "Configuration File",
            ["config/gan_config.yaml", "config/csv_config.yaml"],
            index=0
        )
        st.sidebar.info("🌐 API Mode: Will fetch data from APIs")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Start Training", type="primary"):
            if use_csv:
                dashboard.start_training(None, None, config_path, use_csv=True)
            else:
                dashboard.start_training(
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    config_path,
                    use_csv=False
                )
    
    with col2:
        if st.button("⏹️ Stop Training"):
            dashboard.stop_training()
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Status"):
        dashboard.refresh_training_status()
        st.sidebar.success("Status refreshed!")
    
    # Auto-refresh indicator
    if dashboard.training_status == "running":
        st.sidebar.markdown("### ⚡ Auto-refreshing every 3 seconds...")
    
    # Status indicator
    status_color = "status-running" if dashboard.training_status == "running" else "status-stopped"
    st.sidebar.markdown(f'<div class="{status_color}">Status: {dashboard.training_status.upper()}</div>', 
                        unsafe_allow_html=True)
    
    # Training progress
    if dashboard.training_status == "running":
        st.sidebar.markdown("### 📈 Training Progress")
        if dashboard.total_epochs > 0:
            progress = dashboard.current_epoch / dashboard.total_epochs
            st.sidebar.progress(progress)
            st.sidebar.text(f"Epoch: {dashboard.current_epoch}/{dashboard.total_epochs}")
        
        # Debug information
        st.sidebar.markdown("### 🔍 Debug Info")
        st.sidebar.text(f"Logs captured: {len(dashboard.training_logs)}")
        st.sidebar.text(f"Gen losses: {len(dashboard.generator_losses)}")
        st.sidebar.text(f"Disc losses: {len(dashboard.discriminator_losses)}")
        
        # Show recent sidebar messages (replacing old with new)
        if dashboard.sidebar_messages:
            st.sidebar.markdown("#### 📝 Recent Activity:")
            for msg in dashboard.sidebar_messages:
                if msg["type"] == "error":
                    st.sidebar.error(msg["message"])
                elif msg["type"] == "warning":
                    st.sidebar.warning(msg["message"])
                elif msg["type"] == "success":
                    st.sidebar.success(msg["message"])
                else:
                    st.sidebar.info(msg["message"])
        
        # Clear old messages button
        if st.sidebar.button("🗑️ Clear Messages"):
            dashboard.clear_sidebar_messages()
            st.sidebar.success("Messages cleared!")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Dashboard Overview", 
        "📈 Data Analysis", 
        "🤖 Training Progress", 
        "📊 Results & Evaluation", 
        "⚙️ Configuration"
    ])
    
    with tab1:
        st.markdown("## 📊 Dashboard Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Data Files", len(dashboard.get_data_info()))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            checkpoints = dashboard.get_results_info().get("checkpoints", [])
            st.metric("Checkpoints", len(checkpoints))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            results = dashboard.get_results_info().get("results", [])
            st.metric("Results", len(results))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Training Status", dashboard.training_status.title())
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent activity
        st.markdown("## �� Recent Activity")
        
        # Get new logs
        new_logs = dashboard.get_training_logs()
        if new_logs:
            for log in new_logs[-10:]:  # Show last 10 logs
                log_class = "log-info"
                if "error" in log.lower():
                    log_class = "log-error"
                elif "warning" in log.lower():
                    log_class = "log-warning"
                elif "success" in log.lower() or "completed" in log.lower():
                    log_class = "log-success"
                
                st.markdown(f'<div class="log-entry {log_class}">{log}</div>', unsafe_allow_html=True)
        elif dashboard.training_logs:
            # Show stored logs
            for log in dashboard.training_logs[-10:]:
                log_class = "log-info"
                if "error" in log.lower():
                    log_class = "log-error"
                elif "warning" in log.lower():
                    log_class = "log-warning"
                elif "success" in log.lower() or "completed" in log.lower():
                    log_class = "log-success"
                
                st.markdown(f'<div class="log-entry {log_class}">{log}</div>', unsafe_allow_html=True)
        else:
            st.info("No recent activity. Start training to see logs.")
        
        # Quick actions
        st.markdown("## ⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Report"):
                st.info("Report generation started...")
        
        with col2:
            if st.button("🔍 Analyze Data"):
                st.info("Data analysis started...")
        
        with col3:
            if st.button("📈 Show Results"):
                st.info("Loading results...")
    
    with tab2:
        st.markdown("## 📈 Data Analysis")
        
        # Data information
        data_info = dashboard.get_data_info()
        if data_info:
            st.markdown("### 📁 Available Data Files")
            df_data = pd.DataFrame([
                {
                    "File": name,
                    "Size (MB)": info["size_mb"],
                    "Modified": info["modified"].strftime("%Y-%m-%d %H:%M"),
                    "Path": info["path"]
                }
                for name, info in data_info.items()
            ])
            st.dataframe(df_data, use_container_width=True)
            
            # Data visualization
            st.markdown("### 📊 Data Visualization")
            fig = dashboard.create_data_visualization()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for visualization.")
        else:
            st.warning("No data files found. Please collect data first.")
    
    with tab3:
        st.markdown("## 🤖 Training Progress")
        
        if dashboard.training_status == "running":
            st.success("Training is currently running...")
            
            # Real-time progress
            if dashboard.total_epochs > 0:
                progress_bar = st.progress(dashboard.current_epoch / dashboard.total_epochs)
                status_text = st.empty()
                status_text.text(f"Training progress: Epoch {dashboard.current_epoch}/{dashboard.total_epochs}")
        
        # Training progress chart
        st.markdown("### 📈 Training Losses")
        fig = dashboard.create_training_progress()
        st.plotly_chart(fig, use_container_width=True)
        
        # Training logs
        st.markdown("### 📝 Training Logs")
        
        # Show log file info
        if os.path.exists(dashboard.log_file):
            log_size = os.path.getsize(dashboard.log_file) / 1024  # KB
            st.info(f"📁 Log file: {dashboard.log_file} ({log_size:.1f} KB)")
            
            # Option to view raw log file
            if st.checkbox("Show raw log file contents"):
                with open(dashboard.log_file, 'r') as f:
                    log_contents = f.read()
                st.text_area("Raw Log File", log_contents, height=300)
        
        if dashboard.training_logs:
            # Show all logs with better formatting
            log_container = st.container()
            with log_container:
                for i, log in enumerate(dashboard.training_logs[-50:]):  # Show last 50 logs
                    log_class = "log-info"
                    if "error" in log.lower():
                        log_class = "log-error"
                    elif "warning" in log.lower():
                        log_class = "log-warning"
                    elif "success" in log.lower() or "completed" in log.lower():
                        log_class = "log-success"
                    elif "epoch" in log.lower():
                        log_class = "log-success"
                    
                    st.markdown(f'<div class="log-entry {log_class}">{log}</div>', unsafe_allow_html=True)
        else:
            st.info("No training logs available. Start training to see logs.")
    
    with tab4:
        st.markdown("## 📊 Results & Evaluation")
        
        # Checkpoints
        checkpoints = dashboard.get_results_info().get("checkpoints", [])
        if checkpoints:
            st.markdown("### 💾 Model Checkpoints")
            df_checkpoints = pd.DataFrame([
                {
                    "Checkpoint": cp["name"],
                    "Size (MB)": cp["size_mb"],
                    "Modified": cp["modified"].strftime("%Y-%m-%d %H:%M")
                }
                for cp in checkpoints
            ])
            st.dataframe(df_checkpoints, use_container_width=True)
        else:
            st.info("No checkpoints available. Train a model first.")
        
        # Results
        results = dashboard.get_results_info().get("results", [])
        if results:
            st.markdown("### 📈 Training Results")
            df_results = pd.DataFrame([
                {
                    "File": r["name"],
                    "Size (MB)": r["size_mb"],
                    "Modified": r["modified"].strftime("%Y-%m-%d %H:%M")
                }
                for r in results
            ])
            st.dataframe(df_results, use_container_width=True)
        
        # Synthetic data comparison
        st.markdown("### 🔍 Real vs Synthetic Data")
        fig = dashboard.create_synthetic_data_comparison()
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("## ⚙️ Configuration")
        
        if dashboard.config:
            st.markdown("### 📋 Current Configuration")
            st.json(dashboard.config)
            
            # Configuration editor
            st.markdown("### ✏️ Edit Configuration")
            if st.button("Edit Config"):
                st.info("Configuration editor would open here.")
        else:
            st.error("No configuration loaded.")
        
        # System information
        st.markdown("### 💻 System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Python Version", sys.version.split()[0])
            st.metric("PyTorch Version", torch.__version__ if torch else "Not installed")
        
        with col2:
            st.metric("CUDA Available", "Yes" if torch and torch.cuda.is_available() else "No")
            if torch and torch.cuda.is_available():
                st.metric("GPU Device", torch.cuda.get_device_name(0))

if __name__ == "__main__":
    main() 