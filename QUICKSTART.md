# Treasury Curve GAN - Quick Start Guide

This guide will get you up and running with the Treasury Curve GAN project in minutes.

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies
```bash
make install
```

### 2. Test the Setup
```bash
make test
```

### 3. Run the Full Pipeline
```bash
make quickstart
```

That's it! The project will automatically:
- Collect treasury data (2022-2024)
- Train a GAN model
- Evaluate the model
- Generate synthetic data for nowcasting and hedging

## 📁 Project Structure

```
treasury-gan/
├── config/                 # Configuration files
│   └── gan_config.yaml    # Main GAN configuration
├── data/                   # Data storage and processing
│   └── collector.py       # Data collection script
├── models/                 # GAN model implementations
│   ├── gan_models.py      # Generator, Discriminator, WGAN
│   └── generate.py        # Synthetic data generation
├── training/               # Training pipeline
│   └── trainer.py         # GAN training class
├── evaluation/             # Evaluation metrics
│   └── metrics.py         # Data quality assessment
├── notebooks/              # Jupyter notebooks
│   └── treasury_gan_exploration.ipynb
├── utils/                  # Utility functions
│   └── data_utils.py      # Data processing utilities
├── checkpoints/            # Model checkpoints (created during training)
├── results/                # Evaluation results (created during evaluation)
├── synthetic_data/         # Generated synthetic data
├── train_gan.py            # Main training script
├── test_project.py         # Project testing script
├── requirements.txt        # Python dependencies
├── Makefile               # Common commands
└── README.md              # Project documentation
```

## 🎯 What This Project Does

The Treasury Curve GAN generates synthetic treasury data that can be used for:

### Nowcasting
- **Economic Indicators**: Generate synthetic scenarios for current economic conditions
- **Market Stress**: Create realistic stress scenarios for risk assessment
- **Policy Impact**: Simulate the effects of monetary policy changes

### Hedging
- **Risk Management**: Generate synthetic scenarios for portfolio stress testing
- **Capital Planning**: Create extreme market scenarios for capital adequacy
- **Regulatory Compliance**: Generate scenarios for regulatory stress tests

## 🔧 Key Features

- **5-Level Order Book**: Simulates realistic market microstructure
- **Multiple Instruments**: 2Y, 5Y, 10Y, 30Y yields + SOFR
- **GAN Variants**: Standard GAN, Wasserstein GAN (WGAN)
- **Comprehensive Evaluation**: Statistical tests, financial metrics, distribution analysis
- **Production Ready**: Includes data collection, training, evaluation, and generation

## 📊 Data Sources

- **Treasury Yields**: Yahoo Finance API (^UST2YR, ^UST5YR, etc.)
- **SOFR**: Secured Overnight Financing Rate
- **Order Book**: Simulated 5-level order book with realistic spreads and volumes

## 🚀 Common Commands

### Setup
```bash
make install          # Install dependencies
make test            # Test the setup
```

### Data Collection
```bash
make data            # Collect treasury data
python data/collector.py --help  # See all options
```

### Training
```bash
make train           # Train the model
make train-custom    # Train with custom dates
make evaluate        # Evaluate trained model
```

### Generation
```bash
make generate        # Generate all synthetic data
make generate-normal # Normal scenarios only
make generate-stress # Stress scenarios only
```

### Utilities
```bash
make notebook        # Launch Jupyter notebook
make clean           # Clean generated files
make help            # Show all available commands
```

## 🔍 Exploring the Project

### Jupyter Notebook
```bash
make notebook
```
Open `treasury_gan_exploration.ipynb` for interactive exploration.

### Custom Configuration
Edit `config/gan_config.yaml` to modify:
- Model architecture
- Training parameters
- Data settings

### Advanced Usage
```bash
# Custom data collection
python data/collector.py --start-date 2020-01-01 --end-date 2024-01-01

# Custom training
python train_gan.py --config config/gan_config.yaml --start-date 2020-01-01

# Generate specific scenarios
python models/generate.py --model-path checkpoints/best_model.pth --scenario-type stress
```

## 📈 Expected Results

After running `make quickstart`, you'll have:

1. **Trained Model**: `checkpoints/best_model.pth`
2. **Evaluation Results**: `results/evaluation_results.json`
3. **Synthetic Data**: 
   - `synthetic_data/nowcasting/` - Nowcasting scenarios
   - `synthetic_data/hedging/` - Hedging scenarios
4. **Training Curves**: `training_curves.png`
5. **Evaluation Plots**: `evaluation_results.png`

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Run `make install` to install dependencies
2. **CUDA Issues**: The project automatically falls back to CPU if CUDA is unavailable
3. **Data Collection Failures**: Check internet connection and Yahoo Finance API availability
4. **Memory Issues**: Reduce batch size in `config/gan_config.yaml`

### Getting Help

1. Run `make test` to verify setup
2. Check the logs for detailed error messages
3. Review the Jupyter notebook for examples
4. Check the configuration file for parameter settings

## 🔬 Advanced Features

### Conditional Generation
Generate scenarios based on economic conditions:
```python
from models.generate import SyntheticDataGenerator

generator = SyntheticDataGenerator('checkpoints/best_model.pth', config, device)
scenarios = generator.generate_conditional_scenarios(
    num_scenarios=100,
    economic_conditions={'volatility': 0.5, 'yield_level': 2.0}
)
```

### Custom Evaluation
```python
from evaluation.metrics import TreasuryDataEvaluator

evaluator = TreasuryDataEvaluator(real_data, synthetic_data)
results = evaluator.comprehensive_evaluation()
evaluator.plot_evaluation_results(results)
```

### Model Architecture
The project supports multiple GAN variants:
- **Standard GAN**: Good for general use
- **WGAN**: More stable training, better convergence
- **Conditional GAN**: Generate scenarios based on conditions

## 📚 Next Steps

1. **Explore the Code**: Review the implementation in each module
2. **Customize Models**: Modify the GAN architecture in `models/gan_models.py`
3. **Add Real Data**: Connect to live market data feeds
4. **Deploy**: Integrate with risk management systems
5. **Research**: Extend with advanced GAN architectures

## 🤝 Contributing

This project is designed to be extensible:
- Add new GAN architectures in `models/gan_models.py`
- Implement new evaluation metrics in `evaluation/metrics.py`
- Create new data sources in `data/collector.py`
- Add new training strategies in `training/trainer.py`

---

**Happy Generating! 🎉**

For questions or issues, check the logs and run `make test` to verify your setup. 