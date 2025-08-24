# Treasury GAN Training System

A comprehensive system for training Generative Adversarial Networks (GANs) on Treasury curve data, featuring a modern web dashboard for real-time monitoring and control.

## 🚀 Features

- **Multiple GAN Architectures**: Standard GAN, Wasserstein GAN (WGAN), and Conditional GAN
- **Real-time Training Dashboard**: Web-based UI with live metrics and controls
- **Flexible Data Sources**: Support for CSV files and synthetic data generation
- **Comprehensive Evaluation**: Multiple metrics for assessing synthetic data quality
- **Model Management**: Checkpoint saving/loading and model versioning
- **Real-time Monitoring**: Server-Sent Events (SSE) for live updates

## 📁 Project Structure

```
treasury-gan-training/
├── config/                 # Configuration files
│   ├── gan_config.yaml    # Main GAN configuration
│   └── csv_config.yaml    # CSV-specific configuration
├── data/                  # Data handling modules
│   ├── csv_collector.py   # CSV data processing
│   └── __init__.py
├── models/                # GAN model definitions
│   ├── gan_models.py      # Generator, Discriminator, WGAN models
│   └── __init__.py
├── training/              # Training logic
│   ├── trainer.py         # Main training class
│   └── __init__.py
├── utils/                 # Utility functions
│   ├── data_utils.py      # Data processing utilities
│   └── __init__.py
├── evaluation/            # Model evaluation
│   ├── metrics.py         # Evaluation metrics and plots
│   └── __init__.py
├── gan_dashboard.py       # Modern web dashboard
├── train_gan.py           # Main training script
├── train_gan_csv.py       # CSV-based training script
├── simple_sse_server.py   # Legacy SSE server
├── test_separate_channels.py # Channel testing
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd treasury-gan-training
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create necessary directories**:
   ```bash
   mkdir -p data/csv data/processed checkpoints results logs
   ```

## 🚀 Quick Start

### 1. Start the Dashboard

Launch the modern web dashboard:

```bash
python gan_dashboard.py
```

Open your browser and navigate to `http://localhost:8080`

### 2. Start Training

#### Option A: Using the Dashboard
1. Open the Training page in the dashboard
2. Configure your model parameters
3. Click "Start Training"

#### Option B: Command Line
```bash
# Standard training
python train_gan.py --config config/gan_config.yaml

# CSV-based training
python train_gan_csv.py --config config/csv_config.yaml

# Training with custom parameters
python train_gan.py --start-date 2022-01-01 --end-date 2024-01-01
```

### 3. Monitor Progress

The dashboard provides real-time updates on:
- Training metrics (Generator/Discriminator loss)
- Progress indicators
- Model performance
- Generated samples

## 📊 Dashboard Features

### Main Dashboard
- **Real-time Metrics**: Live updates of training progress
- **Interactive Charts**: Training curves and score comparisons
- **Quick Actions**: Start/stop training, generate samples

### Training Page
- **Configuration Editor**: Modify model and training parameters
- **Training Controls**: Start, stop, and monitor training
- **Live Logs**: Real-time training output

### Evaluation Page
- **Quality Metrics**: Overall quality score, MSE, R²
- **Distribution Analysis**: KS test, Wasserstein distance, JS divergence
- **Correlation Analysis**: Feature correlations and autocorrelations

### Models Page
- **Checkpoint Management**: View, load, and manage saved models
- **Model Information**: Detailed model metadata
- **Export Options**: Download trained models

## ⚙️ Configuration

### GAN Configuration (`config/gan_config.yaml`)

```yaml
model:
  gan_type: "standard"  # "standard" or "wgan"
  generator:
    latent_dim: 100
    hidden_dims: [256, 512, 256, 128]
    dropout: 0.3
  discriminator:
    hidden_dims: [128, 256, 128, 64]
    dropout: 0.3

training:
  epochs: 1000
  learning_rate_generator: 0.0002
  learning_rate_discriminator: 0.0002
  batch_size: 32
  patience: 50
```

### CSV Configuration (`config/csv_config.yaml`)

```yaml
data_source:
  csv_directory: "data/csv"
  file_patterns: ["*.csv"]
  encoding: "utf-8"
  delimiter: ","

features:
  include_spreads: true
  include_volatility: true
  volatility_window: 20
```

## 📈 Data Format

### CSV Data Structure
Your CSV files should contain:
- **Date column**: Time series index
- **Numeric columns**: Treasury yields, spreads, or other features
- **Clean data**: No missing values or non-numeric entries

Example CSV structure:
```csv
date,2Y_yield,5Y_yield,10Y_yield,30Y_yield
2022-01-01,2.45,3.12,3.67,4.23
2022-01-02,2.47,3.15,3.69,4.25
...
```

### Synthetic Data
If no CSV data is provided, the system generates realistic synthetic treasury data with:
- 4 yield curves (2Y, 5Y, 10Y, 30Y)
- Derived spreads
- Volatility features
- Seasonal and trend components

## 🔧 Advanced Usage

### Custom GAN Architectures

Extend the system by adding new models in `models/gan_models.py`:

```python
class CustomGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, sequence_length):
        super().__init__()
        # Your custom architecture here
        pass
    
    def forward(self, z):
        # Your forward pass here
        pass
```

### Custom Evaluation Metrics

Add new evaluation metrics in `evaluation/metrics.py`:

```python
def custom_metric(real_data, synthetic_data):
    """Your custom evaluation metric."""
    # Implementation here
    return metric_value
```

### Integration with External Data Sources

Modify `data/csv_collector.py` to connect to:
- Database systems
- API endpoints
- Real-time data feeds

## 📊 Evaluation Metrics

The system provides comprehensive evaluation:

### Statistical Metrics
- **Basic Statistics**: Mean, standard deviation, min/max
- **Distribution Similarity**: KS test, Wasserstein distance, JS divergence
- **Correlation Analysis**: Feature correlations, autocorrelations

### Quality Metrics
- **Overall Quality Score**: Combined metric for model performance
- **Feature-wise Analysis**: Individual feature performance
- **Time Series Metrics**: MSE, MAE, RMSE, R²

### Visualization
- **Distribution Comparison**: Histograms of real vs synthetic data
- **Correlation Heatmaps**: Feature correlation matrices
- **Time Series Plots**: Sequence comparisons

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Use CPU training: `device = torch.device('cpu')`

2. **Training Instability**
   - Adjust learning rates
   - Use WGAN instead of standard GAN
   - Increase gradient penalty weight

3. **Poor Data Quality**
   - Check data preprocessing
   - Verify feature scaling
   - Ensure sufficient training data

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Financial data community for insights
- Open source contributors

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the configuration examples

---

**Happy Training! 🎯📈** 