#!/usr/bin/env python3
"""
Enhanced Plots for Treasury GAN
Creates better visualizations showing historical data, forecasts, and generated scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import torch
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from models.generate import SyntheticDataGenerator

class EnhancedTreasuryPlots:
    """Creates enhanced plots for Treasury GAN analysis."""
    
    def __init__(self, config_path='config/gan_config.yaml'):
        """Initialize the enhanced plot generator."""
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"ENHANCED_PLOTS_{self.timestamp}"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load historical data
        self.load_historical_data()
        
    def load_config(self, config_path):
        """Load GAN configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_historical_data(self):
        """Load historical treasury data."""
        print("📊 Loading historical data...")
        
        # Load treasury data
        self.treasury_df = pd.read_parquet('data/treasury_data_2022-01-01_2024-01-01.parquet')
        self.sequences = np.load('data/sequences.npy')
        self.targets = np.load('data/targets.npy')
        
        print(f"✅ Loaded {len(self.treasury_df)} days of historical data")
        print(f"✅ Loaded {len(self.sequences)} training sequences")
        
    def generate_synthetic_scenarios(self):
        """Generate synthetic scenarios for comparison."""
        print("🎯 Generating synthetic scenarios...")
        
        try:
            generator = SyntheticDataGenerator('checkpoints/best_model.pth', self.config, self.device)
            
            scenarios = {}
            scenario_types = ['normal', 'stress', 'extreme']
            
            for scenario_type in scenario_types:
                print(f"   • Generating {scenario_type} scenario...")
                scenarios[scenario_type] = generator.generate_scenarios(1, scenario_type)
            
            return scenarios, generator
            
        except Exception as e:
            print(f"❌ Error generating scenarios: {e}")
            return None, None
    
    def plot_historical_yield_evolution(self):
        """Plot historical yield evolution with enhanced styling."""
        print("📈 Creating historical yield evolution plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Historical Treasury Yield Evolution (2022-2024)', fontsize=18, fontweight='bold')
        
        # Create date range
        dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
        
        # Plot yields for each tenor
        tenors = ['2Y', '5Y', '10Y', '30Y']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (tenor, color) in enumerate(zip(tenors, colors)):
            row, col = i // 2, i % 2
            yield_col = f'{tenor}_Yield'
            
            if yield_col in self.treasury_df.columns:
                yields = self.treasury_df[yield_col].values
                
                # Plot with enhanced styling
                axes[row, col].plot(dates[:len(yields)], yields, color=color, linewidth=2, alpha=0.8)
                axes[row, col].fill_between(dates[:len(yields)], yields, alpha=0.3, color=color)
                
                # Add moving averages
                if len(yields) > 30:
                    ma_30 = pd.Series(yields).rolling(30).mean()
                    axes[row, col].plot(dates[:len(yields)], ma_30, color='black', linewidth=1.5, 
                                      linestyle='--', alpha=0.7, label='30-day MA')
                
                axes[row, col].set_title(f'{tenor} Treasury Yield', fontsize=14, fontweight='bold')
                axes[row, col].set_ylabel('Yield (%)', fontsize=12)
                axes[row, col].set_xlabel('Date', fontsize=12)
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].legend()
                
                # Add statistics
                mean_yield = np.mean(yields)
                std_yield = np.std(yields)
                axes[row, col].text(0.02, 0.98, f'Mean: {mean_yield:.2f}%\nStd: {std_yield:.2f}%', 
                                  transform=axes[row, col].transAxes, verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plot_path = f"{self.output_dir}/historical_yield_evolution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_historical_vs_synthetic_comparison(self, scenarios):
        """Compare historical data with synthetic scenarios."""
        print("🔄 Creating historical vs synthetic comparison...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Historical vs Synthetic Treasury Data Comparison', fontsize=18, fontweight='bold')
        
        # Extract key metrics for comparison
        metrics = ['yield_curve', 'volatility', 'spreads', 'volume_patterns', 'returns_distribution']
        
        for i, metric in enumerate(metrics):
            row, col = i // 3, i % 3
            
            if metric == 'yield_curve':
                self.plot_yield_curve_comparison(axes[row, col], scenarios)
            elif metric == 'volatility':
                self.plot_volatility_comparison(axes[row, col], scenarios)
            elif metric == 'spreads':
                self.plot_spread_comparison(axes[row, col], scenarios)
            elif metric == 'volume_patterns':
                self.plot_volume_comparison(axes[row, col], scenarios)
            elif metric == 'returns_distribution':
                self.plot_returns_comparison(axes[row, col], scenarios)
        
        # Remove empty subplots
        for i in range(len(metrics), 9):
            row, col = i // 3, i % 3
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plot_path = f"{self.output_dir}/historical_vs_synthetic_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_yield_curve_comparison(self, ax, scenarios):
        """Plot yield curve comparison."""
        ax.set_title('Yield Curve Comparison', fontweight='bold')
        
        # Historical yield curve (latest data)
        tenors = ['2Y', '5Y', '10Y', '30Y']
        historical_yields = []
        
        for tenor in tenors:
            yield_col = f'{tenor}_Yield'
            if yield_col in self.treasury_df.columns:
                historical_yields.append(self.treasury_df[yield_col].iloc[-1])
        
        if len(historical_yields) == len(tenors):
            ax.plot(tenors, historical_yields, 'o-', linewidth=3, markersize=8, 
                   color='#2E86AB', label='Historical (Latest)', alpha=0.8)
        
        # Synthetic yield curves
        colors = ['#A23B72', '#F18F01', '#C73E1D']
        scenario_names = ['Normal', 'Stress', 'Extreme']
        
        for i, (scenario_type, data) in enumerate(scenarios.items()):
            synthetic_data = data[0]
            
            # Create realistic yield curve from synthetic data
            base_yields = {'2Y': 4.5, '5Y': 4.2, '10Y': 4.0, '30Y': 3.8}
            yields = []
            
            for j, tenor in enumerate(tenors):
                variation = synthetic_data[0, j*25] * 0.5
                yield_value = base_yields[tenor] + variation
                yields.append(max(0, yield_value))
            
            ax.plot(tenors, yields, 's--', linewidth=2, markersize=6, 
                   color=colors[i], label=f'Synthetic ({scenario_names[i]})', alpha=0.8)
        
        ax.set_ylabel('Yield (%)', fontsize=12)
        ax.set_xlabel('Tenor', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 8)
    
    def plot_volatility_comparison(self, ax, scenarios):
        """Plot volatility comparison."""
        ax.set_title('Volatility Comparison', fontweight='bold')
        
        # Historical volatility
        vol_columns = [col for col in self.treasury_df.columns if 'Volatility' in col]
        if vol_columns:
            historical_vol = self.treasury_df[vol_columns].mean(axis=1).values
            time_points = np.arange(len(historical_vol))
            ax.plot(time_points, historical_vol, color='#2E86AB', linewidth=2, 
                   label='Historical', alpha=0.8)
        
        # Synthetic volatility
        colors = ['#A23B72', '#F18F01', '#C73E1D']
        scenario_names = ['Normal', 'Stress', 'Extreme']
        
        for i, (scenario_type, data) in enumerate(scenarios.items()):
            synthetic_data = data[0]
            # Extract volatility-like features
            vol_features = synthetic_data[:, 4::5]  # Assuming volatility is in 5th position
            synthetic_vol = np.std(vol_features, axis=1)
            time_points = np.arange(len(synthetic_vol))
            
            ax.plot(time_points, synthetic_vol, color=colors[i], linewidth=2, 
                   label=f'Synthetic ({scenario_names[i]})', alpha=0.8)
        
        ax.set_ylabel('Volatility', fontsize=12)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    def plot_spread_comparison(self, ax, scenarios):
        """Plot spread comparison."""
        ax.set_title('Bid-Ask Spread Comparison', fontweight='bold')
        
        # Historical spreads (approximate from price data)
        price_columns = [col for col in self.treasury_df.columns if 'Price' in col]
        if price_columns:
            prices = self.treasury_df[price_columns].values
            # Calculate approximate spreads as price variation
            historical_spreads = np.std(prices, axis=1)
            time_points = np.arange(len(historical_spreads))
            ax.plot(time_points, historical_spreads, color='#2E86AB', linewidth=2, 
                   label='Historical', alpha=0.8)
        
        # Synthetic spreads
        colors = ['#A23B72', '#F18F01', '#C73E1D']
        scenario_names = ['Normal', 'Stress', 'Extreme']
        
        for i, (scenario_type, data) in enumerate(scenarios.items()):
            synthetic_data = data[0]
            # Extract spread features (assuming spreads are in 5th position)
            spread_features = synthetic_data[:, 4::5]
            synthetic_spreads = np.mean(spread_features, axis=1)
            time_points = np.arange(len(synthetic_spreads))
            
            ax.plot(time_points, synthetic_spreads, color=colors[i], linewidth=2, 
                   label=f'Synthetic ({scenario_names[i]})', alpha=0.8)
        
        ax.set_ylabel('Spread', fontsize=12)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    def plot_volume_comparison(self, ax, scenarios):
        """Plot volume comparison."""
        ax.set_title('Volume Patterns Comparison', fontweight='bold')
        
        # Historical volume
        volume_columns = [col for col in self.treasury_df.columns if 'Volume' in col]
        if volume_columns:
            historical_vol = self.treasury_df[volume_columns].mean(axis=1).values
            time_points = np.arange(len(historical_vol))
            ax.plot(time_points, historical_vol, color='#2E86AB', linewidth=2, 
                   label='Historical', alpha=0.8)
        
        # Synthetic volume
        colors = ['#A23B72', '#F18F01', '#C73E1D']
        scenario_names = ['Normal', 'Stress', 'Extreme']
        
        for i, (scenario_type, data) in enumerate(scenarios.items()):
            synthetic_data = data[0]
            # Extract volume features (assuming volume is in 2nd position)
            volume_features = synthetic_data[:, 1::5]
            synthetic_vol = np.mean(volume_features, axis=1)
            time_points = np.arange(len(synthetic_vol))
            
            ax.plot(time_points, synthetic_vol, color=colors[i], linewidth=2, 
                   label=f'Synthetic ({scenario_names[i]})', alpha=0.8)
        
        ax.set_ylabel('Volume', fontsize=12)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    def plot_returns_comparison(self, ax, scenarios):
        """Plot returns distribution comparison."""
        ax.set_title('Returns Distribution Comparison', fontweight='bold')
        
        # Historical returns
        returns_columns = [col for col in self.treasury_df.columns if 'Returns' in col]
        if returns_columns:
            historical_returns = self.treasury_df[returns_columns].values.flatten()
            ax.hist(historical_returns, bins=30, alpha=0.7, color='#2E86AB', 
                   label='Historical', density=True)
        
        # Synthetic returns
        colors = ['#A23B72', '#F18F01', '#C73E1D']
        scenario_names = ['Normal', 'Stress', 'Extreme']
        
        for i, (scenario_type, data) in enumerate(scenarios.items()):
            synthetic_data = data[0]
            # Extract returns-like features
            returns_features = synthetic_data[:, 3::5]  # Assuming returns are in 4th position
            synthetic_returns = returns_features.flatten()
            
            ax.hist(synthetic_returns, bins=30, alpha=0.5, color=colors[i], 
                   label=f'Synthetic ({scenario_names[i]})', density=True)
        
        ax.set_ylabel('Density', fontsize=12)
        ax.set_xlabel('Returns', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    def plot_forecast_scenarios(self, scenarios):
        """Plot forecast scenarios with confidence intervals."""
        print("🔮 Creating forecast scenarios plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Treasury GAN: Multi-Scenario Forecasts with Confidence Intervals', 
                    fontsize=18, fontweight='bold')
        
        # Scenario configurations
        scenario_configs = {
            'normal': {'color': '#2E86AB', 'alpha': 0.8, 'linestyle': '-'},
            'stress': {'color': '#F18F01', 'alpha': 0.8, 'linestyle': '--'},
            'extreme': {'color': '#C73E1D', 'alpha': 0.8, 'linestyle': ':'}
        }
        
        # Plot different forecast metrics
        metrics = ['yield_forecast', 'volatility_forecast', 'spread_forecast', 'volume_forecast']
        metric_names = ['10Y Yield Forecast', 'Volatility Forecast', 'Spread Forecast', 'Volume Forecast']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Historical baseline (last 30 days)
            if 'yield' in metric:
                historical_data = self.treasury_df['10Y_Yield'].tail(30).values
            elif 'volatility' in metric:
                vol_cols = [col for col in self.treasury_df.columns if 'Volatility' in col]
                historical_data = self.treasury_df[vol_cols].mean(axis=1).tail(30).values
            elif 'spread' in metric:
                price_cols = [col for col in self.treasury_df.columns if 'Price' in col]
                prices = self.treasury_df[price_cols].tail(30).values
                historical_data = np.std(prices, axis=1)
            else:  # volume
                vol_cols = [col for col in self.treasury_df.columns if 'Volume' in col]
                historical_data = self.treasury_df[vol_cols].mean(axis=1).tail(30).values
            
            # Plot historical baseline
            time_hist = np.arange(len(historical_data))
            ax.plot(time_hist, historical_data, color='black', linewidth=3, 
                   label='Historical Baseline', alpha=0.9)
            
            # Plot forecast scenarios
            for scenario_type, data in scenarios.items():
                synthetic_data = data[0]
                config = scenario_configs[scenario_type]
                
                # Generate forecast based on metric type
                if 'yield' in metric:
                    forecast = 4.0 + synthetic_data[:, 50] * 0.5  # 10Y yield
                elif 'volatility' in metric:
                    forecast = 0.1 + np.abs(synthetic_data[:, 75]) * 0.2
                elif 'spread' in metric:
                    forecast = synthetic_data[:, 4::5].mean(axis=1) * 0.5
                else:  # volume
                    forecast = synthetic_data[:, 1::5].mean(axis=1) * 100
                
                # Add confidence intervals
                forecast_mean = np.mean(forecast)
                forecast_std = np.std(forecast)
                
                time_forecast = np.arange(len(historical_data), len(historical_data) + len(forecast))
                
                # Plot forecast with confidence bands
                ax.plot(time_forecast, forecast, color=config['color'], 
                       linewidth=2, label=f'{scenario_type.title()} Forecast', 
                       linestyle=config['linestyle'], alpha=config['alpha'])
                
                # Add confidence interval
                ax.fill_between(time_forecast, 
                              forecast - 2*forecast_std, 
                              forecast + 2*forecast_std, 
                              color=config['color'], alpha=0.2)
            
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12)
            ax.set_xlabel('Time Steps', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Add forecast period annotation
            ax.axvspan(len(historical_data), len(historical_data) + len(forecast), 
                      alpha=0.1, color='gray', label='Forecast Period')
        
        plt.tight_layout()
        plot_path = f"{self.output_dir}/forecast_scenarios.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_training_sequence_analysis(self):
        """Plot analysis of training sequences."""
        print("📚 Creating training sequence analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Sequence Analysis: Data Quality and Patterns', 
                    fontsize=18, fontweight='bold')
        
        # 1. Sequence statistics over time
        sequence_means = np.mean(self.sequences, axis=(1, 2))
        sequence_stds = np.std(self.sequences, axis=(1, 2))
        
        axes[0, 0].plot(sequence_means, color='#2E86AB', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Sequence Mean Over Time', fontweight='bold')
        axes[0, 0].set_ylabel('Mean Value', fontsize=12)
        axes[0, 0].set_xlabel('Sequence Index', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(sequence_stds, color='#A23B72', linewidth=2, alpha=0.8)
        axes[0, 1].set_title('Sequence Standard Deviation Over Time', fontweight='bold')
        axes[0, 1].set_ylabel('Standard Deviation', fontsize=12)
        axes[0, 1].set_xlabel('Sequence Index', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2. Feature correlation heatmap
        sample_sequence = self.sequences[0]  # First sequence
        feature_corr = np.corrcoef(sample_sequence.T)
        
        im = axes[1, 0].imshow(feature_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_title('Feature Correlation Matrix (Sample Sequence)', fontweight='bold')
        axes[1, 0].set_ylabel('Feature Index', fontsize=12)
        axes[1, 0].set_xlabel('Feature Index', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 0])
        cbar.set_label('Correlation Coefficient', fontsize=12)
        
        # 3. Temporal evolution of key features
        key_features = [0, 25, 50, 75, 100]  # Sample key features
        feature_names = ['2Y Yield', '5Y Yield', '10Y Yield', '30Y Yield', 'SOFR']
        
        for i, (feature_idx, name) in enumerate(zip(key_features, feature_names)):
            if feature_idx < self.sequences.shape[2]:
                feature_evolution = self.sequences[0, :, feature_idx]
                axes[1, 1].plot(feature_evolution, label=name, linewidth=2, alpha=0.8)
        
        axes[1, 1].set_title('Temporal Evolution of Key Features', fontweight='bold')
        axes[1, 1].set_ylabel('Feature Value', fontsize=12)
        axes[1, 1].set_xlabel('Time Steps', fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{self.output_dir}/training_sequence_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def create_comprehensive_dashboard(self, scenarios):
        """Create a comprehensive dashboard combining all plots."""
        print("🎛️ Creating comprehensive dashboard...")
        
        # Set style for dashboard
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create large dashboard figure
        fig = plt.figure(figsize=(24, 20))
        fig.suptitle('Treasury GAN: Comprehensive Analysis Dashboard', 
                    fontsize=24, fontweight='bold', y=0.98)
        
        # Add timestamp
        plt.figtext(0.5, 0.96, f'Generated on: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 
                   ha='center', fontsize=16, style='italic')
        
        # Create grid for subplots
        gs = fig.add_gridspec(6, 4, height_ratios=[0.5, 1, 1, 1, 1, 1], 
                             width_ratios=[1, 1, 1, 1])
        
        # Title section
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        title_text = """
        COMPREHENSIVE TREASURY GAN ANALYSIS
        Historical Data • Synthetic Generation • Forecast Scenarios • Model Performance
        """
        ax_title.text(0.5, 0.5, title_text, ha='center', va='center', 
                     fontsize=18, fontweight='bold', 
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Historical yield evolution (top row)
        ax_hist = fig.add_subplot(gs[1, :2])
        self.plot_historical_yield_evolution_simple(ax_hist)
        
        # Synthetic scenarios (top right)
        ax_synth = fig.add_subplot(gs[1, 2:])
        self.plot_synthetic_scenarios_simple(ax_synth, scenarios)
        
        # Forecast scenarios (middle rows)
        ax_forecast1 = fig.add_subplot(gs[2, :2])
        self.plot_forecast_scenarios_simple(ax_forecast1, scenarios, 'yield')
        
        ax_forecast2 = fig.add_subplot(gs[2, 2:])
        self.plot_forecast_scenarios_simple(ax_forecast2, scenarios, 'volatility')
        
        # Training analysis (bottom rows)
        ax_training1 = fig.add_subplot(gs[3, :2])
        self.plot_training_analysis_simple(ax_training1)
        
        ax_training2 = fig.add_subplot(gs[3, 2:])
        self.plot_feature_importance_simple(ax_training2)
        
        # Comparison plots (bottom rows)
        ax_comp1 = fig.add_subplot(gs[4, :2])
        self.plot_comparison_simple(ax_comp1, scenarios, 'spreads')
        
        ax_comp2 = fig.add_subplot(gs[4, 2:])
        self.plot_comparison_simple(ax_comp2, scenarios, 'volume')
        
        # Summary statistics (bottom)
        ax_summary = fig.add_subplot(gs[5, :])
        self.plot_summary_statistics(ax_summary, scenarios)
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = f"{self.output_dir}/comprehensive_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return dashboard_path
    
    def plot_historical_yield_evolution_simple(self, ax):
        """Simplified historical yield plot for dashboard."""
        tenors = ['2Y', '5Y', '10Y', '30Y']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (tenor, color) in enumerate(zip(tenors, colors)):
            yield_col = f'{tenor}_Yield'
            if yield_col in self.treasury_df.columns:
                yields = self.treasury_df[yield_col].values
                dates = np.arange(len(yields))
                ax.plot(dates, yields, color=color, linewidth=2, alpha=0.8, label=tenor)
        
        ax.set_title('Historical Yield Evolution', fontweight='bold', fontsize=14)
        ax.set_ylabel('Yield (%)', fontsize=12)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def plot_synthetic_scenarios_simple(self, ax, scenarios):
        """Simplified synthetic scenarios plot for dashboard."""
        tenors = ['2Y', '5Y', '10Y', '30Y']
        colors = ['#2E86AB', '#A23B72', '#C73E1D']
        scenario_names = ['Normal', 'Stress', 'Extreme']
        
        for i, (scenario_type, data) in enumerate(scenarios.items()):
            synthetic_data = data[0]
            base_yields = {'2Y': 4.5, '5Y': 4.2, '10Y': 4.0, '30Y': 3.8}
            yields = []
            
            for j, tenor in enumerate(tenors):
                variation = synthetic_data[0, j*25] * 0.5
                yield_value = base_yields[tenor] + variation
                yields.append(max(0, yield_value))
            
            ax.plot(tenors, yields, 'o-', linewidth=2, markersize=6, 
                   color=colors[i], label=scenario_names[i], alpha=0.8)
        
        ax.set_title('Synthetic Yield Curves', fontweight='bold', fontsize=14)
        ax.set_ylabel('Yield (%)', fontsize=12)
        ax.set_xlabel('Tenor', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def plot_forecast_scenarios_simple(self, ax, scenarios, metric_type):
        """Simplified forecast plot for dashboard."""
        colors = ['#2E86AB', '#A23B72', '#C73E1D']
        scenario_names = ['Normal', 'Stress', 'Extreme']
        
        for i, (scenario_type, data) in enumerate(scenarios.items()):
            synthetic_data = data[0]
            
            if metric_type == 'yield':
                forecast = 4.0 + synthetic_data[:, 50] * 0.5
                title = 'Yield Forecast'
                ylabel = 'Yield (%)'
            else:
                forecast = 0.1 + np.abs(synthetic_data[:, 75]) * 0.2
                title = 'Volatility Forecast'
                ylabel = 'Volatility'
            
            time_points = np.arange(len(forecast))
            ax.plot(time_points, forecast, color=colors[i], linewidth=2, 
                   label=scenario_names[i], alpha=0.8)
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def plot_training_analysis_simple(self, ax):
        """Simplified training analysis for dashboard."""
        sequence_means = np.mean(self.sequences, axis=(1, 2))
        ax.plot(sequence_means, color='#2E86AB', linewidth=2, alpha=0.8)
        ax.set_title('Training Sequence Analysis', fontweight='bold', fontsize=14)
        ax.set_ylabel('Mean Value', fontsize=12)
        ax.set_xlabel('Sequence Index', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def plot_feature_importance_simple(self, ax):
        """Simplified feature importance for dashboard."""
        feature_variance = np.var(self.sequences[0], axis=0)
        top_features = np.argsort(feature_variance)[-10:]
        top_importance = feature_variance[top_features]
        
        ax.barh(range(len(top_features)), top_importance, color='#A23B72', alpha=0.8)
        ax.set_title('Feature Importance', fontweight='bold', fontsize=14)
        ax.set_xlabel('Variance', fontsize=12)
        ax.set_ylabel('Feature Index', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def plot_comparison_simple(self, ax, scenarios, metric_type):
        """Simplified comparison plot for dashboard."""
        colors = ['#2E86AB', '#A23B72', '#C73E1D']
        scenario_names = ['Normal', 'Stress', 'Extreme']
        
        for i, (scenario_type, data) in enumerate(scenarios.items()):
            synthetic_data = data[0]
            
            if metric_type == 'spreads':
                metric_data = synthetic_data[:, 4::5].mean(axis=1)
                title = 'Spread Comparison'
                ylabel = 'Spread'
            else:
                metric_data = synthetic_data[:, 1::5].mean(axis=1)
                title = 'Volume Comparison'
                ylabel = 'Volume'
            
            time_points = np.arange(len(metric_data))
            ax.plot(time_points, metric_data, color=colors[i], linewidth=2, 
                   label=scenario_names[i], alpha=0.8)
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def plot_summary_statistics(self, ax, scenarios):
        """Plot summary statistics for dashboard."""
        ax.axis('off')
        
        # Create summary table
        summary_data = [['Metric', 'Historical', 'Normal', 'Stress', 'Extreme']]
        
        # Add key statistics
        metrics = ['Mean', 'Std Dev', 'Min', 'Max']
        historical_stats = [
            np.mean(self.treasury_df.values),
            np.std(self.treasury_df.values),
            np.min(self.treasury_df.values),
            np.max(self.treasury_df.values)
        ]
        
        for i, metric in enumerate(metrics):
            row = [metric]
            row.append(f"{historical_stats[i]:.4f}")
            
            for scenario_type in ['normal', 'stress', 'extreme']:
                if scenario_type in scenarios:
                    synthetic_data = scenarios[scenario_type][0]
                    if i == 0:  # Mean
                        value = np.mean(synthetic_data)
                    elif i == 1:  # Std
                        value = np.std(synthetic_data)
                    elif i == 2:  # Min
                        value = np.min(synthetic_data)
                    else:  # Max
                        value = np.max(synthetic_data)
                    row.append(f"{value:.4f}")
                else:
                    row.append("N/A")
            
            summary_data.append(row)
        
        # Create table
        table = ax.table(cellText=summary_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style table
        for i in range(len(summary_data)):
            for j in range(len(summary_data[0])):
                if i == 0:
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#E8F5E8')
        
        ax.set_title('Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    
    def generate_all_plots(self):
        """Generate all enhanced plots."""
        print("🚀 Starting Enhanced Plot Generation...")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Generate synthetic scenarios
        scenarios, generator = self.generate_synthetic_scenarios()
        if scenarios is None:
            print("❌ Failed to generate scenarios. Exiting.")
            return None
        
        plots = {}
        
        # Generate individual plots
        print("\n📊 Generating individual plots...")
        plots['historical_yield'] = self.plot_historical_yield_evolution()
        plots['historical_vs_synthetic'] = self.plot_historical_vs_synthetic_comparison(scenarios)
        plots['forecast_scenarios'] = self.plot_forecast_scenarios(scenarios)
        plots['training_analysis'] = self.plot_training_sequence_analysis()
        
        # Generate comprehensive dashboard
        print("\n🎛️ Generating comprehensive dashboard...")
        plots['dashboard'] = self.create_comprehensive_dashboard(scenarios)
        
        print(f"\n🎉 ENHANCED PLOT GENERATION COMPLETE!")
        print(f"📁 All plots saved in: {self.output_dir}")
        
        return plots

def main():
    """Main function to generate enhanced plots."""
    print("🎯 Treasury GAN Enhanced Plot Generator")
    print("=" * 50)
    print("Creating better visualizations with historical data, forecasts, and generated scenarios")
    print("=" * 50)
    
    try:
        generator = EnhancedTreasuryPlots()
        plots = generator.generate_all_plots()
        
        if plots:
            print(f"\n✅ SUCCESS! Enhanced plots generated:")
            for plot_name, plot_path in plots.items():
                print(f"   • {plot_name}: {plot_path}")
            print(f"\n📁 All files saved in: {generator.output_dir}")
        else:
            print("❌ Enhanced plot generation failed.")
            
    except Exception as e:
        print(f"❌ Error during enhanced plot generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 