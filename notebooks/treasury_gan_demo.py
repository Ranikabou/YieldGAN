#!/usr/bin/env python3
"""
Treasury GAN Demo - Show Realistic Time Series Predictions
This script demonstrates what the GAN actually generates in understandable terms.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.generate import SyntheticDataGenerator
import yaml
import torch

# Set plotting style for better readability
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_config():
    """Load GAN configuration."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'gan_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def interpret_treasury_features(feature_idx):
    """
    Convert abstract feature indices to understandable treasury concepts.
    
    Args:
        feature_idx: Feature index (0-124)
        
    Returns:
        Human-readable description of the feature
    """
    # 125 features = 5 instruments × 5 levels × 5 features
    instruments = ['2Y', '5Y', '10Y', '30Y', 'SOFR']
    levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
    features = ['Bid Price', 'Bid Volume', 'Ask Price', 'Ask Volume', 'Spread']
    
    if feature_idx >= 125:
        return "Unknown Feature"
    
    instrument_idx = feature_idx // 25
    level_idx = (feature_idx % 25) // 5
    feature_idx_in_group = feature_idx % 5
    
    instrument = instruments[instrument_idx]
    level = levels[level_idx]
    feature = features[feature_idx_in_group]
    
    return f"{instrument} - {level} - {feature}"

def generate_realistic_scenarios():
    """Generate and display realistic treasury scenarios."""
    print("🎯 Generating Realistic Treasury Scenarios...")
    
    # Load configuration and model
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fix model path
    model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_model.pth')
    generator = SyntheticDataGenerator(model_path, config, device)
    
    # Generate different types of scenarios
    print("\n📊 Generating 5 different market scenarios...")
    
    scenarios = {}
    global scenario_names
    scenario_names = {
        'normal': 'Normal Market Conditions',
        'stress': 'Market Stress Scenario',
        'extreme': 'Extreme Market Conditions'
    }
    
    for scenario_type in scenario_names.keys():
        print(f"   • {scenario_names[scenario_type]}")
        scenarios[scenario_type] = generator.generate_scenarios(1, scenario_type)
    
    return scenarios, generator

def plot_treasury_yield_curves(scenarios, generator):
    """Plot realistic treasury yield curves from different scenarios."""
    print("\n📈 Plotting Treasury Yield Curves...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Treasury GAN: Realistic Yield Curve Scenarios', fontsize=16, fontweight='bold')
    
    # Define realistic yield curve points
    tenors = ['2Y', '5Y', '10Y', '30Y', 'SOFR']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    scenario_types = list(scenarios.keys())
    
    for idx, (scenario_type, data) in enumerate(scenarios.items()):
        row = idx // 3
        col = idx % 3
        
        # Extract yield curve data (simplified - using first few features)
        # In reality, you'd map specific features to yields
        synthetic_data = data[0]  # First scenario
        
        # Create realistic yield curve
        base_yields = {
            '2Y': 4.5, '5Y': 4.2, '10Y': 4.0, '30Y': 3.8, 'SOFR': 5.3
        }
        
        # Add GAN-generated variation
        yields = []
        for i, tenor in enumerate(tenors):
            # Use GAN output to modify base yields
            variation = synthetic_data[0, i*25] * 0.5  # Scale GAN output
            yield_value = base_yields[tenor] + variation
            yields.append(max(0, yield_value))  # Ensure positive yields
        
        # Plot yield curve
        axes[row, col].plot(tenors, yields, 'o-', linewidth=2, markersize=8, 
                           color='#2E86AB', markerfacecolor='#A23B72')
        axes[row, col].set_title(f'{scenario_names[scenario_type]}', fontsize=14, fontweight='bold')
        axes[row, col].set_ylabel('Yield (%)', fontsize=12)
        axes[row, col].set_xlabel('Tenor', fontsize=12)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_ylim(0, 8)
        
        # Add yield values as text
        for i, (tenor, yield_val) in enumerate(zip(tenors, yields)):
            axes[row, col].text(i, yield_val + 0.1, f'{yield_val:.2f}%', 
                               ha='center', va='bottom', fontweight='bold')
    
    # Remove empty subplot
    if len(scenarios) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig('treasury_yield_scenarios.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return scenarios

def plot_order_book_dynamics(scenarios):
    """Plot realistic order book dynamics."""
    print("\n📊 Plotting Order Book Dynamics...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Treasury GAN: Order Book Microstructure', fontsize=16, fontweight='bold')
    
    scenario_types = list(scenarios.keys())
    
    for idx, (scenario_type, data) in enumerate(scenarios.items()):
        row = idx // 3
        col = idx % 3
        
        synthetic_data = data[0]  # First scenario
        
        # Extract order book data for 10Y Treasury (middle of curve)
        # Features 50-74 represent 10Y Treasury (5 levels × 5 features)
        start_idx = 50
        levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
        
        bid_prices = []
        ask_prices = []
        spreads = []
        
        for level in range(5):
            # Extract bid price, ask price, and spread for each level
            bid_idx = start_idx + level * 5 + 0  # Bid price
            ask_idx = start_idx + level * 5 + 2  # Ask price
            spread_idx = start_idx + level * 5 + 4  # Spread
            
            # Convert GAN output to realistic prices
            base_price = 100.0  # Par value
            bid_price = base_price + synthetic_data[0, bid_idx] * 2
            ask_price = base_price + synthetic_data[0, ask_idx] * 2
            spread = synthetic_data[0, spread_idx] * 0.5
            
            bid_prices.append(bid_price)
            ask_prices.append(ask_price)
            spreads.append(spread)
        
        # Plot order book
        x_pos = np.arange(len(levels))
        width = 0.35
        
        axes[row, col].bar(x_pos - width/2, bid_prices, width, label='Bid Prices', 
                           color='#2E86AB', alpha=0.8)
        axes[row, col].bar(x_pos + width/2, ask_prices, width, label='Ask Prices', 
                           color='#A23B72', alpha=0.8)
        
        axes[row, col].set_title(f'{scenario_names[scenario_type]}', fontsize=14, fontweight='bold')
        axes[row, col].set_ylabel('Price ($)', fontsize=12)
        axes[row, col].set_xlabel('Order Book Level', fontsize=12)
        axes[row, col].set_xticks(x_pos)
        axes[row, col].set_xticklabels(levels)
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
        
        # Add spread annotations
        for i, spread in enumerate(spreads):
            axes[row, col].text(i, (bid_prices[i] + ask_prices[i])/2, 
                               f'Spread: {spread:.2f}', ha='center', va='center',
                               fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor="yellow", alpha=0.7))
    
    # Remove empty subplot
    if len(scenarios) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig('order_book_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_time_series_evolution(scenarios):
    """Plot how treasury metrics evolve over time."""
    print("\n⏰ Plotting Time Series Evolution...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Treasury GAN: Time Series Evolution (100 Timesteps)', fontsize=16, fontweight='bold')
    
    scenario_types = list(scenarios.keys())
    
    for idx, (scenario_type, data) in enumerate(scenarios.items()):
        row = idx // 3
        col = idx % 3
        
        synthetic_data = data[0]  # First scenario
        
        # Extract time series for key metrics
        time_steps = np.arange(100)
        
        # 10Y Treasury yield evolution (using feature 50 as proxy)
        yield_evolution = 4.0 + synthetic_data[:, 50] * 0.5
        
        # SOFR evolution (using feature 100 as proxy)
        sofr_evolution = 5.3 + synthetic_data[:, 100] * 0.3
        
        # Volatility evolution (using feature 75 as proxy)
        vol_evolution = 0.1 + np.abs(synthetic_data[:, 75]) * 0.2
        
        # Plot multiple metrics
        axes[row, col].plot(time_steps, yield_evolution, 'b-', linewidth=2, 
                           label='10Y Treasury Yield', alpha=0.8)
        axes[row, col].plot(time_steps, sofr_evolution, 'r-', linewidth=2, 
                           label='SOFR Rate', alpha=0.8)
        axes[row, col].plot(time_steps, vol_evolution, 'g-', linewidth=2, 
                           label='Volatility', alpha=0.8)
        
        axes[row, col].set_title(f'{scenario_names[scenario_type]}', fontsize=14, fontweight='bold')
        axes[row, col].set_ylabel('Rate/Volatility (%)', fontsize=12)
        axes[row, col].set_xlabel('Time Steps', fontsize=12)
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
        
        # Add scenario-specific annotations
        if scenario_type == 'stress':
            axes[row, col].axvspan(40, 60, alpha=0.2, color='red', label='Stress Period')
        elif scenario_type == 'recovery':
            axes[row, col].axvspan(30, 70, alpha=0.2, color='green', label='Recovery Period')
    
    # Remove empty subplot
    if len(scenarios) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig('time_series_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_practical_applications():
    """Show practical applications of the generated scenarios."""
    print("\n🎯 Practical Applications of Treasury GAN Scenarios:")
    print("=" * 60)
    
    applications = [
        {
            "Use Case": "Portfolio Stress Testing",
            "Description": "Generate extreme market scenarios to test portfolio resilience",
            "Example": "What happens to your bond portfolio if yields spike 200bps?",
            "GAN Value": "Creates realistic stress scenarios based on historical patterns"
        },
        {
            "Use Case": "Risk Management",
            "Description": "Calculate Value at Risk (VaR) under different market conditions",
            "Example": "What's the 95% VaR for your treasury position?",
            "GAN Value": "Generates thousands of realistic market scenarios for VaR calculation"
        },
        {
            "Use Case": "Capital Planning",
            "Description": "Determine capital adequacy under various market stresses",
            "Example": "How much capital do you need for regulatory compliance?",
            "GAN Value": "Creates extreme but realistic scenarios for capital adequacy testing"
        },
        {
            "Use Case": "Trading Strategy Backtesting",
            "Description": "Test trading strategies against synthetic market conditions",
            "Example": "How does your yield curve strategy perform in volatile markets?",
            "GAN Value": "Provides diverse market scenarios beyond historical data"
        },
        {
            "Use Case": "Economic Nowcasting",
            "Description": "Generate scenarios for current economic conditions",
            "Example": "What are the possible paths for Fed policy impact?",
            "GAN Value": "Creates realistic policy impact scenarios for economic forecasting"
        }
    ]
    
    for i, app in enumerate(applications, 1):
        print(f"\n{i}. {app['Use Case']}")
        print(f"   📝 {app['Description']}")
        print(f"   💡 {app['Example']}")
        print(f"   🎯 {app['GAN Value']}")
    
    print("\n" + "=" * 60)

def main():
    """Main demonstration function."""
    print("🚀 Treasury GAN: Realistic Time Series Demonstration")
    print("=" * 60)
    print("This demo shows what the GAN actually generates in terms you can understand!")
    print("=" * 60)
    
    try:
        # Generate scenarios
        scenarios, generator = generate_realistic_scenarios()
        
        # Show what each feature actually represents
        print(f"\n🔍 Understanding the 125 Features:")
        print("   The GAN generates 125 features representing:")
        print("   • 5 Treasury instruments (2Y, 5Y, 10Y, 30Y, SOFR)")
        print("   • 5 Order book levels (Level 1 = tightest, Level 5 = widest)")
        print("   • 5 Features per level (Bid Price, Bid Volume, Ask Price, Ask Volume, Spread)")
        print("   • 100 time steps (representing market evolution)")
        
        # Show feature mapping examples
        print(f"\n📋 Feature Mapping Examples:")
        for i in [0, 25, 50, 75, 100]:
            print(f"   Feature {i}: {interpret_treasury_features(i)}")
        
        # Plot realistic scenarios
        scenarios = plot_treasury_yield_curves(scenarios, generator)
        plot_order_book_dynamics(scenarios)
        plot_time_series_evolution(scenarios)
        
        # Show practical applications
        show_practical_applications()
        
        print(f"\n✅ Demonstration Complete!")
        print(f"   Generated plots saved as:")
        print(f"   • treasury_yield_scenarios.png")
        print(f"   • order_book_dynamics.png") 
        print(f"   • time_series_evolution.png")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 