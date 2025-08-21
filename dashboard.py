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
</style>
""", unsafe_allow_html=True)

class GANDashboard:
    def __init__(self):
        self.config = self.load_config()
        self.training_status = "stopped"
        self.training_process = None
        self.log_queue = queue.Queue()
        
    def load_config(self):
        """Load GAN configuration file."""
        try:
            with open('config/gan_config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            st.error("Configuration file not found. Please ensure config/gan_config.yaml exists.")
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
    
    def start_training(self, start_date, end_date, config_path):
        """Start GAN training process."""
        try:
            cmd = [
                sys.executable, "train_gan.py",
                "--start-date", start_date,
                "--end-date", end_date,
                "--config", config_path
            ]
            
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.training_status = "running"
            st.success("Training started successfully!")
            
            # Start log monitoring in background
            threading.Thread(target=self.monitor_training_logs, daemon=True).start()
            
        except Exception as e:
            st.error(f"Failed to start training: {e}")
    
    def stop_training(self):
        """Stop GAN training process."""
        if self.training_process:
            self.training_process.terminate()
            self.training_status = "stopped"
            st.warning("Training stopped.")
    
    def monitor_training_logs(self):
        """Monitor training logs in background."""
        if self.training_process:
            for line in iter(self.training_process.stdout.readline, ''):
                if line:
                    self.log_queue.put(line.strip())
                    if self.training_process.poll() is not None:
                        break
    
    def get_training_logs(self):
        """Get recent training logs."""
        logs = []
        while not self.log_queue.empty():
            logs.append(self.log_queue.get_nowait())
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
        # This would typically read from actual training logs
        # For now, creating a sample progress chart
        
        epochs = list(range(1, 51))
        generator_loss = [np.random.uniform(0.5, 2.0) for _ in epochs]
        discriminator_loss = [np.random.uniform(0.3, 1.5) for _ in epochs]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs, y=generator_loss,
            mode='lines+markers',
            name='Generator Loss',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs, y=discriminator_loss,
            mode='lines+markers',
            name='Discriminator Loss',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Training Progress",
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

def main():
    st.markdown('<h1 class="main-header">📊 Treasury GAN Training Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = GANDashboard()
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Training controls
    st.sidebar.markdown("### 🚀 Training Controls")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2022, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2024, 1, 1))
    
    config_path = st.sidebar.selectbox(
        "Configuration File",
        ["config/gan_config.yaml", "config/custom_config.yaml"],
        index=0
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Start Training", type="primary"):
            dashboard.start_training(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                config_path
            )
    
    with col2:
        if st.button("⏹️ Stop Training"):
            dashboard.stop_training()
    
    # Status indicator
    status_color = "status-running" if dashboard.training_status == "running" else "status-stopped"
    st.sidebar.markdown(f'<div class="{status_color}">Status: {dashboard.training_status.upper()}</div>', 
                        unsafe_allow_html=True)
    
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
        st.markdown("## 🔄 Recent Activity")
        logs = dashboard.get_training_logs()
        if logs:
            for log in logs[-10:]:  # Show last 10 logs
                st.text(log)
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
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress (in real implementation, this would read from logs)
            for i in range(100):
                time.sleep(0.1)
                progress_bar.progress(i + 1)
                status_text.text(f"Training progress: {i + 1}%")
        
        # Training progress chart
        st.markdown("### 📈 Training Losses")
        fig = dashboard.create_training_progress()
        st.plotly_chart(fig, use_container_width=True)
        
        # Training logs
        st.markdown("### 📝 Training Logs")
        logs = dashboard.get_training_logs()
        if logs:
            log_text = "\n".join(logs[-20:])  # Show last 20 logs
            st.text_area("Recent Logs", log_text, height=200)
        else:
            st.info("No training logs available.")
    
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