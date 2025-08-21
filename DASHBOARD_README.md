# 🎯 Treasury GAN Training Dashboard

A beautiful, interactive web dashboard for monitoring and controlling your Treasury GAN training pipeline in real-time.

## ✨ Features

### 🏠 **Dashboard Overview**
- **Real-time metrics**: Data files, checkpoints, results, and training status
- **Quick actions**: Generate reports, analyze data, show results
- **Recent activity**: Live training logs and updates

### 📈 **Data Analysis**
- **Data file explorer**: View all available data files with sizes and timestamps
- **Interactive visualizations**: Treasury yield sequences, target distributions, feature correlations
- **Data statistics**: Comprehensive overview of your dataset

### 🤖 **Training Progress**
- **Real-time monitoring**: Live training progress with progress bars
- **Loss visualization**: Interactive charts showing generator vs discriminator losses
- **Training logs**: Real-time log streaming from the training process

### 📊 **Results & Evaluation**
- **Model checkpoints**: View and manage saved model states
- **Training results**: Access all generated outputs and metrics
- **Data comparison**: Side-by-side comparison of real vs synthetic data

### ⚙️ **Configuration**
- **Current settings**: View and edit GAN configuration parameters
- **System info**: Python, PyTorch, and CUDA availability
- **Parameter tuning**: Modify training parameters on the fly

## 🚀 Quick Start

### **Option 1: Automatic Launch (Recommended)**

#### **On macOS/Linux:**
```bash
./run_dashboard.sh
```

#### **On Windows:**
```cmd
run_dashboard.bat
```

### **Option 2: Manual Launch**

1. **Install dependencies:**
   ```bash
   pip install -r requirements_dashboard.txt
   ```

2. **Launch dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

3. **Open browser:**
   Navigate to `http://localhost:8501`

## 📋 Requirements

The dashboard requires these packages:
- `streamlit` - Web framework
- `plotly` - Interactive charts
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `pyyaml` - Configuration parsing
- `torch` - PyTorch for model operations
- `scikit-learn` - Data preprocessing

## 🎮 How to Use

### **1. Start Training**
- Set your desired date range in the sidebar
- Choose configuration file
- Click "▶️ Start Training"
- Monitor progress in real-time

### **2. Monitor Progress**
- Watch training losses in the "Training Progress" tab
- View real-time logs as they happen
- Check GPU/CPU utilization

### **3. Analyze Results**
- Compare real vs synthetic data
- View model checkpoints
- Access evaluation metrics

### **4. Control Training**
- Start/stop training at any time
- Modify configuration parameters
- Generate reports on demand

## 🔧 Configuration

The dashboard automatically loads your GAN configuration from `config/gan_config.yaml`. You can:

- **Modify parameters** through the web interface
- **Save changes** to update your configuration
- **Switch between** different config files

## 📊 Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    📊 Treasury GAN Dashboard                │
├─────────────────────────────────────────────────────────────┤
│ 🏠 Overview │ 📈 Data │ 🤖 Training │ 📊 Results │ ⚙️ Config │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    Main Content Area                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ 🎛️ Control Panel                                          │
│ ├─ 🚀 Training Controls                                    │
│ ├─ 📅 Date Selection                                       │
│ ├─ ⚙️ Configuration                                        │
│ └─ 📊 Status Indicator                                     │
└─────────────────────────────────────────────────────────────┘
```

## 🎨 Customization

### **Adding New Visualizations**
The dashboard is built with Plotly, making it easy to add custom charts:

```python
def create_custom_chart():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1,2,3], y=[1,2,3]))
    return fig

# Use in dashboard
st.plotly_chart(create_custom_chart())
```

### **Modifying Layout**
Adjust the dashboard layout by modifying the tab structure in `dashboard.py`.

## 🐛 Troubleshooting

### **Dashboard won't start:**
1. Check if all dependencies are installed: `pip install -r requirements_dashboard.txt`
2. Ensure you're in the project directory
3. Check if port 8501 is available

### **No data showing:**
1. Ensure you have data files in the `data/` directory
2. Check if the GAN has been trained at least once
3. Verify configuration file exists

### **Training won't start:**
1. Check if `train_gan.py` exists and is executable
2. Verify configuration file is valid
3. Ensure all required Python modules are available

## 🔄 Integration with Training Pipeline

The dashboard seamlessly integrates with your existing GAN training pipeline:

- **Real-time monitoring** of `train_gan.py` execution
- **Automatic data loading** from your data directories
- **Direct access** to model checkpoints and results
- **Configuration management** for your GAN parameters

## 📱 Mobile Friendly

The dashboard is fully responsive and works on:
- 🖥️ Desktop computers
- 📱 Mobile phones
- 💻 Tablets
- 🖥️ Different screen sizes

## 🚀 Performance Tips

- **Close unused tabs** to improve performance
- **Limit log history** for very long training sessions
- **Use GPU acceleration** when available
- **Monitor memory usage** during long training runs

## 🤝 Contributing

To add new features to the dashboard:

1. **Fork the repository**
2. **Create a feature branch**
3. **Add your improvements**
4. **Submit a pull request**

## 📄 License

This dashboard is part of the Treasury GAN project and follows the same MIT license.

---

**🎯 Ready to visualize your GAN training? Launch the dashboard now!**

```bash
./run_dashboard.sh  # macOS/Linux
# or
run_dashboard.bat   # Windows
``` 