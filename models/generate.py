#!/usr/bin/env python3
"""
Synthetic data generation script for Treasury Curve GAN.
Generates synthetic treasury data for nowcasting and hedging applications.
"""

import argparse
import logging
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.gan_models import create_gan_models, create_wgan_models
from utils.data_utils import TreasuryDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Generates synthetic treasury data using trained GAN models.
    """
    
    def __init__(self, model_path: str, config: dict, device: torch.device):
        """
        Initialize generator with trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration dictionary
            device: Device to run generation on
        """
        self.config = config
        self.device = device
        
        # Load model
        self._load_model(model_path)
        
        # Initialize data processor for scaling
        self.processor = TreasuryDataProcessor(
            instruments=config['instruments'],
            sequence_length=config['data']['sequence_length']
        )
    
    def _load_model(self, model_path: str):
        """Load trained GAN model from checkpoint."""
        logger.info(f"Loading model from {model_path}")
        
        # Create models
        if self.config.get('gan_type') == 'wgan':
            self.generator, self.critic = create_wgan_models(self.config, self.device)
        else:
            self.generator, self.discriminator = create_gan_models(self.config, self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        
        # Set to evaluation mode
        self.generator.eval()
        
        logger.info("Model loaded successfully")
    
    def generate_scenarios(self, num_scenarios: int, scenario_type: str = 'normal') -> np.ndarray:
        """
        Generate synthetic treasury scenarios.
        
        Args:
            num_scenarios: Number of scenarios to generate
            scenario_type: Type of scenario ('normal', 'stress', 'extreme')
            
        Returns:
            Generated scenarios array
        """
        logger.info(f"Generating {num_scenarios} {scenario_type} scenarios")
        
        with torch.no_grad():
            # Generate noise
            latent_dim = self.config['model']['generator']['latent_dim']
            
            if scenario_type == 'normal':
                noise = torch.randn(num_scenarios, latent_dim).to(self.device)
            elif scenario_type == 'stress':
                # Increase variance for stress scenarios
                noise = torch.randn(num_scenarios, latent_dim).to(self.device) * 1.5
            elif scenario_type == 'extreme':
                # Much higher variance for extreme scenarios
                noise = torch.randn(num_scenarios, latent_dim).to(self.device) * 3.0
            else:
                raise ValueError(f"Unknown scenario type: {scenario_type}")
            
            # Generate synthetic data
            synthetic_data = self.generator(noise)
            
        return synthetic_data.cpu().numpy()
    
    def generate_conditional_scenarios(self, num_scenarios: int, 
                                    economic_conditions: dict) -> np.ndarray:
        """
        Generate conditional scenarios based on economic conditions.
        
        Args:
            num_scenarios: Number of scenarios to generate
            economic_conditions: Dictionary with economic indicators
            
        Returns:
            Generated conditional scenarios
        """
        logger.info(f"Generating {num_scenarios} conditional scenarios")
        
        # This would require a conditional GAN model
        # For now, we'll use the standard generator with modified noise
        
        with torch.no_grad():
            latent_dim = self.config['model']['generator']['latent_dim']
            
            # Create noise based on economic conditions
            base_noise = torch.randn(num_scenarios, latent_dim).to(self.device)
            
            # Modify noise based on conditions
            if 'volatility' in economic_conditions:
                vol_factor = economic_conditions['volatility']
                base_noise *= (1 + vol_factor)
            
            if 'yield_level' in economic_conditions:
                # Shift noise based on yield level
                yield_shift = economic_conditions['yield_level']
                base_noise += yield_shift * 0.1
            
            # Generate synthetic data
            synthetic_data = self.generator(base_noise)
            
        return synthetic_data.cpu().numpy()
    
    def generate_nowcasting_data(self, num_scenarios: int = 100) -> dict:
        """
        Generate synthetic data for nowcasting applications.
        
        Args:
            num_scenarios: Number of scenarios to generate
            
        Returns:
            Dictionary with nowcasting data
        """
        logger.info("Generating nowcasting data")
        
        # Generate normal scenarios
        normal_scenarios = self.generate_scenarios(num_scenarios, 'normal')
        
        # Generate stress scenarios
        stress_scenarios = self.generate_scenarios(num_scenarios // 2, 'stress')
        
        # Calculate scenario statistics
        nowcasting_data = {
            'normal_scenarios': normal_scenarios,
            'stress_scenarios': stress_scenarios,
            'scenario_metadata': {
                'num_normal': num_scenarios,
                'num_stress': num_scenarios // 2,
                'generation_timestamp': datetime.now().isoformat(),
                'model_config': self.config
            }
        }
        
        return nowcasting_data
    
    def generate_hedging_data(self, num_scenarios: int = 200) -> dict:
        """
        Generate synthetic data for hedging applications.
        
        Args:
            num_scenarios: Number of scenarios to generate
            
        Returns:
            Dictionary with hedging data
        """
        logger.info("Generating hedging data")
        
        # Generate different types of scenarios
        normal_scenarios = self.generate_scenarios(num_scenarios // 2, 'normal')
        stress_scenarios = self.generate_scenarios(num_scenarios // 4, 'stress')
        extreme_scenarios = self.generate_scenarios(num_scenarios // 4, 'extreme')
        
        # Calculate risk metrics
        all_scenarios = np.concatenate([normal_scenarios, stress_scenarios, extreme_scenarios], axis=0)
        
        # Calculate Value at Risk (VaR) at different confidence levels
        scenario_returns = np.diff(all_scenarios, axis=1)  # Calculate returns
        var_95 = np.percentile(scenario_returns, 5, axis=0)
        var_99 = np.percentile(scenario_returns, 1, axis=0)
        
        # Calculate Expected Shortfall (ES)
        es_95 = np.mean(scenario_returns[scenario_returns <= var_95], axis=0)
        es_99 = np.mean(scenario_returns[scenario_returns <= var_99], axis=0)
        
        hedging_data = {
            'normal_scenarios': normal_scenarios,
            'stress_scenarios': stress_scenarios,
            'extreme_scenarios': extreme_scenarios,
            'risk_metrics': {
                'var_95': var_95,
                'var_99': var_99,
                'es_95': es_95,
                'es_99': es_99,
                'max_loss': np.min(scenario_returns, axis=0),
                'volatility': np.std(scenario_returns, axis=0)
            },
            'metadata': {
                'num_scenarios': num_scenarios,
                'generation_timestamp': datetime.now().isoformat(),
                'model_config': self.config
            }
        }
        
        return hedging_data
    
    def save_synthetic_data(self, data: dict, output_dir: str = 'synthetic_data'):
        """
        Save generated synthetic data to files.
        
        Args:
            data: Dictionary with synthetic data
            output_dir: Directory to save data
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save different types of data
        if 'normal_scenarios' in data:
            np.save(f'{output_dir}/normal_scenarios_{timestamp}.npy', data['normal_scenarios'])
        
        if 'stress_scenarios' in data:
            np.save(f'{output_dir}/stress_scenarios_{timestamp}.npy', data['stress_scenarios'])
        
        if 'extreme_scenarios' in data:
            np.save(f'{output_dir}/extreme_scenarios_{timestamp}.npy', data['extreme_scenarios'])
        
        # Save metadata
        metadata_file = f'{output_dir}/metadata_{timestamp}.json'
        with open(metadata_file, 'w') as f:
            json.dump(data.get('metadata', {}), f, indent=2, default=str)
        
        # Save risk metrics if available
        if 'risk_metrics' in data:
            risk_metrics_file = f'{output_dir}/risk_metrics_{timestamp}.json'
            with open(risk_metrics_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                risk_data = {}
                for key, value in data['risk_metrics'].items():
                    if isinstance(value, np.ndarray):
                        risk_data[key] = value.tolist()
                    else:
                        risk_data[key] = value
                json.dump(risk_data, f, indent=2)
        
        logger.info(f"Synthetic data saved to {output_dir}")
    
    def generate_and_save_all(self, output_dir: str = 'synthetic_data'):
        """
        Generate and save all types of synthetic data.
        
        Args:
            output_dir: Directory to save data
        """
        logger.info("Generating comprehensive synthetic data")
        
        # Generate nowcasting data
        nowcasting_data = self.generate_nowcasting_data()
        self.save_synthetic_data(nowcasting_data, f'{output_dir}/nowcasting')
        
        # Generate hedging data
        hedging_data = self.generate_hedging_data()
        self.save_synthetic_data(hedging_data, f'{output_dir}/hedging')
        
        logger.info("All synthetic data generated and saved")

def load_config(config_path: str) -> dict:
    """Load configuration from file."""
    # Handle relative paths
    if not os.path.isabs(config_path):
        # If it's a relative path, try to resolve it
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if config_path.startswith('../'):
            config_path = os.path.join(script_dir, config_path)
        else:
            config_path = os.path.join(script_dir, '..', config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(description='Generate synthetic treasury data')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='../config/gan_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num-scenarios', type=int, default=100,
                       help='Number of scenarios to generate')
    parser.add_argument('--scenario-type', type=str, default='all',
                       choices=['normal', 'stress', 'extreme', 'all'],
                       help='Type of scenarios to generate')
    parser.add_argument('--output-dir', type=str, default='synthetic_data',
                       help='Output directory for generated data')
    parser.add_argument('--nowcasting', action='store_true',
                       help='Generate nowcasting data')
    parser.add_argument('--hedging', action='store_true',
                       help='Generate hedging data')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize generator
    try:
        generator = SyntheticDataGenerator(args.model_path, config, device)
        logger.info("Generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        return
    
    # Generate data based on arguments
    if args.nowcasting or args.hedging:
        if args.nowcasting:
            logger.info("Generating nowcasting data...")
            nowcasting_data = generator.generate_nowcasting_data(args.num_scenarios)
            generator.save_synthetic_data(nowcasting_data, f'{args.output_dir}/nowcasting')
        
        if args.hedging:
            logger.info("Generating hedging data...")
            hedging_data = generator.generate_hedging_data(args.num_scenarios)
            generator.save_synthetic_data(hedging_data, f'{args.output_dir}/hedging')
    else:
        # Generate specific scenario type or all
        if args.scenario_type == 'all':
            generator.generate_and_save_all(args.output_dir)
        else:
            scenarios = generator.generate_scenarios(args.num_scenarios, args.scenario_type)
            data = {
                f'{args.scenario_type}_scenarios': scenarios,
                'metadata': {
                    'num_scenarios': args.num_scenarios,
                    'scenario_type': args.scenario_type,
                    'generation_timestamp': datetime.now().isoformat()
                }
            }
            generator.save_synthetic_data(data, args.output_dir)
    
    logger.info("Synthetic data generation completed!")

if __name__ == "__main__":
    main() 