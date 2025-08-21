#!/bin/bash

echo "🚀 Launching Treasury GAN Dashboard..."
echo "📊 Installing dashboard dependencies..."

# Install dashboard requirements
pip install -r requirements_dashboard.txt

echo "🎯 Starting Streamlit dashboard..."
echo "🌐 Dashboard will open in your browser at: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the dashboard"

# Launch the dashboard
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 