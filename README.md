# Treasury Curve GAN Project

This project implements Generative Adversarial Networks (GANs) to generate synthetic treasury curve data from 5-level order book data, including SOFR (Secured Overnight Financing Rate) and the broader treasury curve.

## Project Overview

The goal is to train GANs that can generate realistic synthetic treasury data for:
- **Nowcasting**: Predicting current economic conditions using high-frequency market data
- **Hedging**: Creating synthetic scenarios for risk management and portfolio optimization

## Features

- **Data Collection**: Fetches 5-level order book data for treasury instruments
- **Data Preprocessing**: Handles missing data, normalization, and feature engineering
- **GAN Architecture**: Implements multiple GAN variants (DCGAN, WGAN, etc.)
- **Training Pipeline**: Complete training loop with monitoring and checkpointing
- **Evaluation**: Metrics for assessing synthetic data quality
- **Synthetic Data Generation**: Production-ready data generation pipeline

## Project Structure

```
treasury-gan/
├── data/                   # Data storage and processing
├── models/                 # GAN model implementations
├── training/               # Training scripts and utilities
├── evaluation/             # Evaluation metrics and visualization
├── notebooks/              # Jupyter notebooks for exploration
├── config/                 # Configuration files
└── utils/                  # Utility functions
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Collection**:
   ```bash
   python data/collector.py
   ```

2. **Training**:
   ```bash
   python training/train_gan.py --config config/gan_config.yaml
   ```

3. **Generate Synthetic Data**:
   ```bash
   python models/generate.py --model_path checkpoints/best_model.pth
   ```

## Data Sources

- **Treasury Curve**: 2Y, 5Y, 10Y, 30Y yields
- **SOFR**: Secured Overnight Financing Rate
- **Order Book Data**: Bid/ask prices and volumes at 5 levels

## Model Architecture

The GAN consists of:
- **Generator**: Transforms random noise into synthetic treasury data
- **Discriminator**: Distinguishes real from synthetic data
- **Conditional Inputs**: Economic indicators and market conditions

## License

MIT License
