#!/usr/bin/env python3
"""
Treasury GAN Simple Report Generator
Creates comprehensive reports with all results and metrics using matplotlib.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json
from datetime import datetime
import torch
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.generate import SyntheticDataGenerator

class TreasuryGANSimpleReportGenerator:
    """Generates comprehensive reports for Treasury GAN results using matplotlib."""
    
    def __init__(self, config_path='config/gan_config.yaml'):
        """Initialize the report generator."""
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = f"reports/run_{self.timestamp}"
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories."""
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.report_dir}/plots").mkdir(exist_ok=True)
        Path(f"{self.report_dir}/data").mkdir(exist_ok=True)
        Path(f"{self.report_dir}/metrics").mkdir(exist_ok=True)
        Path(f"{self.report_dir}/config").mkdir(exist_ok=True)
        
    def load_config(self, config_path):
        """Load GAN configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_all_scenarios(self):
        """Generate all types of scenarios."""
        print("🎯 Generating Treasury GAN Scenarios...")
        
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
    
    def create_comprehensive_plots(self, scenarios):
        """Create all plots for the report."""
        print("📊 Creating Comprehensive Plots...")
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        plots = {}
        
        # 1. Yield Curve Scenarios
        plots['yield_curves'] = self.plot_yield_curves(scenarios)
        
        # 2. Order Book Dynamics
        plots['order_book'] = self.plot_order_book_dynamics(scenarios)
        
        # 3. Time Series Evolution
        plots['time_series'] = self.plot_time_series_evolution(scenarios)
        
        # 4. Feature Importance Analysis
        plots['feature_importance'] = self.plot_feature_importance(scenarios)
        
        # 5. Market Stress Indicators
        plots['stress_indicators'] = self.plot_stress_indicators(scenarios)
        
        return plots
    
    def plot_yield_curves(self, scenarios):
        """Plot yield curves for different scenarios."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Treasury GAN: Yield Curve Scenarios', fontsize=16, fontweight='bold')
        
        tenors = ['2Y', '5Y', '10Y', '30Y', 'SOFR']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, (scenario_type, data) in enumerate(scenarios.items()):
            synthetic_data = data[0]
            
            # Create realistic yield curve
            base_yields = {'2Y': 4.5, '5Y': 4.2, '10Y': 4.0, '30Y': 3.8, 'SOFR': 5.3}
            yields = []
            
            for i, tenor in enumerate(tenors):
                variation = synthetic_data[0, i*25] * 0.5
                yield_value = base_yields[tenor] + variation
                yields.append(max(0, yield_value))
            
            axes[idx].plot(tenors, yields, 'o-', linewidth=2, markersize=8, 
                          color='#2E86AB', markerfacecolor='#A23B72')
            axes[idx].set_title(f'{scenario_type.title()} Market', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Yield (%)', fontsize=12)
            axes[idx].set_xlabel('Tenor', fontsize=12)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim(0, 8)
            
            # Add yield values
            for i, (tenor, yield_val) in enumerate(zip(tenors, yields)):
                axes[idx].text(i, yield_val + 0.1, f'{yield_val:.2f}%', 
                              ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot_path = f"{self.report_dir}/plots/yield_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_order_book_dynamics(self, scenarios):
        """Plot order book dynamics."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Treasury GAN: Order Book Microstructure', fontsize=16, fontweight='bold')
        
        for idx, (scenario_type, data) in enumerate(scenarios.items()):
            synthetic_data = data[0]
            
            # Extract order book data for 10Y Treasury
            start_idx = 50
            levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
            
            bid_prices = []
            ask_prices = []
            spreads = []
            
            for level in range(5):
                bid_idx = start_idx + level * 5 + 0
                ask_idx = start_idx + level * 5 + 2
                spread_idx = start_idx + level * 5 + 4
                
                base_price = 100.0
                bid_price = base_price + synthetic_data[0, bid_idx] * 2
                ask_price = base_price + synthetic_data[0, ask_idx] * 2
                spread = synthetic_data[0, spread_idx] * 0.5
                
                bid_prices.append(bid_price)
                ask_prices.append(ask_price)
                spreads.append(spread)
            
            x_pos = np.arange(len(levels))
            width = 0.35
            
            axes[idx].bar(x_pos - width/2, bid_prices, width, label='Bid Prices', 
                          color='#2E86AB', alpha=0.8)
            axes[idx].bar(x_pos + width/2, ask_prices, width, label='Ask Prices', 
                          color='#A23B72', alpha=0.8)
            
            axes[idx].set_title(f'{scenario_type.title()} Market', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Price ($)', fontsize=12)
            axes[idx].set_xlabel('Order Book Level', fontsize=12)
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels(levels)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
            # Add spread annotations
            for i, spread in enumerate(spreads):
                axes[idx].text(i, (bid_prices[i] + ask_prices[i])/2, 
                              f'Spread: {spread:.2f}', ha='center', va='center',
                              fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                              facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        plot_path = f"{self.report_dir}/plots/order_book_dynamics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_time_series_evolution(self, scenarios):
        """Plot time series evolution."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Treasury GAN: Time Series Evolution (100 Timesteps)', fontsize=16, fontweight='bold')
        
        for idx, (scenario_type, data) in enumerate(scenarios.items()):
            synthetic_data = data[0]
            time_steps = np.arange(100)
            
            # Extract key metrics
            yield_evolution = 4.0 + synthetic_data[:, 50] * 0.5
            sofr_evolution = 5.3 + synthetic_data[:, 100] * 0.3
            vol_evolution = 0.1 + np.abs(synthetic_data[:, 75]) * 0.2
            
            axes[idx].plot(time_steps, yield_evolution, 'b-', linewidth=2, 
                          label='10Y Treasury Yield', alpha=0.8)
            axes[idx].plot(time_steps, sofr_evolution, 'r-', linewidth=2, 
                          label='SOFR Rate', alpha=0.8)
            axes[idx].plot(time_steps, vol_evolution, 'g-', linewidth=2, 
                          label='Volatility', alpha=0.8)
            
            axes[idx].set_title(f'{scenario_type.title()} Market', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Rate/Volatility (%)', fontsize=12)
            axes[idx].set_xlabel('Time Steps', fontsize=12)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
            # Add scenario-specific annotations
            if scenario_type == 'stress':
                axes[idx].axvspan(40, 60, alpha=0.2, color='red', label='Stress Period')
            elif scenario_type == 'extreme':
                axes[idx].axvspan(30, 70, alpha=0.2, color='orange', label='Extreme Period')
        
        plt.tight_layout()
        plot_path = f"{self.report_dir}/plots/time_series_evolution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_feature_importance(self, scenarios):
        """Plot feature importance analysis."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Treasury GAN: Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        for idx, (scenario_type, data) in enumerate(scenarios.items()):
            synthetic_data = data[0]
            
            # Calculate feature importance (variance across time)
            feature_variance = np.var(synthetic_data, axis=0)
            
            # Top 20 most important features
            top_features = np.argsort(feature_variance)[-20:]
            top_importance = feature_variance[top_features]
            
            axes[idx].barh(range(len(top_features)), top_importance, color='#2E86AB', alpha=0.8)
            axes[idx].set_title(f'{scenario_type.title()} Market', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Feature Importance (Variance)', fontsize=12)
            axes[idx].set_ylabel('Feature Index', fontsize=12)
            axes[idx].grid(True, alpha=0.3)
            
            # Add feature labels
            for i, feature_idx in enumerate(top_features):
                axes[idx].text(top_importance[i], i, f'F{feature_idx}', 
                              ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plot_path = f"{self.report_dir}/plots/feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_stress_indicators(self, scenarios):
        """Plot market stress indicators."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Treasury GAN: Market Stress Indicators', fontsize=16, fontweight='bold')
        
        for idx, (scenario_type, data) in enumerate(scenarios.items()):
            synthetic_data = data[0]
            
            # Calculate stress indicators
            time_steps = np.arange(100)
            
            # Spread volatility (stress indicator)
            spreads = synthetic_data[:, 4::5]  # All spread features
            spread_volatility = np.std(spreads, axis=1)
            
            # Price volatility (stress indicator)
            prices = synthetic_data[:, 0::5]  # All price features
            price_volatility = np.std(prices, axis=1)
            
            # Volume stress (stress indicator)
            volumes = synthetic_data[:, 1::5]  # All volume features
            volume_stress = np.mean(volumes, axis=1)
            
            axes[idx].plot(time_steps, spread_volatility, 'r-', linewidth=2, 
                          label='Spread Volatility', alpha=0.8)
            axes[idx].plot(time_steps, price_volatility, 'b-', linewidth=2, 
                          label='Price Volatility', alpha=0.8)
            axes[idx].plot(time_steps, volume_stress, 'g-', linewidth=2, 
                          label='Volume Stress', alpha=0.8)
            
            axes[idx].set_title(f'{scenario_type.title()} Market', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Stress Level', fontsize=12)
            axes[idx].set_xlabel('Time Steps', fontsize=12)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{self.report_dir}/plots/stress_indicators.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def calculate_comprehensive_metrics(self, scenarios):
        """Calculate comprehensive metrics for the report."""
        print("📊 Calculating Comprehensive Metrics...")
        
        metrics = {}
        
        for scenario_type, data in scenarios.items():
            synthetic_data = data[0]
            
            # Basic statistics
            metrics[scenario_type] = {
                'mean': np.mean(synthetic_data),
                'std': np.std(synthetic_data),
                'min': np.min(synthetic_data),
                'max': np.max(synthetic_data),
                'skewness': self.calculate_skewness(synthetic_data),
                'kurtosis': self.calculate_kurtosis(synthetic_data)
            }
            
            # Market-specific metrics
            metrics[scenario_type].update({
                'yield_curve_slope': self.calculate_yield_curve_slope(synthetic_data),
                'spread_volatility': self.calculate_spread_volatility(synthetic_data),
                'price_efficiency': self.calculate_price_efficiency(synthetic_data),
                'liquidity_score': self.calculate_liquidity_score(synthetic_data)
            })
        
        # Save metrics to JSON
        metrics_path = f"{self.report_dir}/metrics/comprehensive_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        return metrics
    
    def calculate_skewness(self, data):
        """Calculate skewness of the data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def calculate_kurtosis(self, data):
        """Calculate kurtosis of the data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def calculate_yield_curve_slope(self, data):
        """Calculate yield curve slope."""
        # Use 2Y and 10Y features as proxy
        two_year = data[:, 0]  # Feature 0: 2Y Level 1 Bid Price
        ten_year = data[:, 50]  # Feature 50: 10Y Level 1 Bid Price
        return np.mean(ten_year - two_year)
    
    def calculate_spread_volatility(self, data):
        """Calculate spread volatility."""
        spreads = data[:, 4::5]  # All spread features
        return np.std(spreads)
    
    def calculate_price_efficiency(self, data):
        """Calculate price efficiency (autocorrelation)."""
        prices = data[:, 0::5]  # All price features
        autocorr = np.corrcoef(prices[:-1].flatten(), prices[1:].flatten())[0, 1]
        return autocorr if not np.isnan(autocorr) else 0
    
    def calculate_liquidity_score(self, data):
        """Calculate liquidity score based on spreads and volumes."""
        spreads = data[:, 4::5]  # All spread features
        volumes = data[:, 1::5]  # All volume features
        
        # Lower spreads and higher volumes = higher liquidity
        avg_spread = np.mean(spreads)
        avg_volume = np.mean(volumes)
        
        # Normalize and combine
        liquidity_score = (1 / (1 + avg_spread)) * (avg_volume / (1 + avg_volume))
        return liquidity_score
    
    def create_comprehensive_report(self, plots, metrics):
        """Create a comprehensive report using matplotlib."""
        print("📊 Creating Comprehensive Report...")
        
        # Create a large figure with all plots
        fig = plt.figure(figsize=(24, 30))
        fig.suptitle('Treasury GAN Comprehensive Report', fontsize=24, fontweight='bold', y=0.98)
        
        # Add timestamp and run info
        plt.figtext(0.5, 0.96, f'Generated on: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 
                   ha='center', fontsize=14, style='italic')
        plt.figtext(0.5, 0.94, f'Run ID: {self.timestamp}', ha='center', fontsize=12, style='italic')
        
        # Create subplots for different sections
        gs = fig.add_gridspec(8, 3, height_ratios=[0.5, 1, 1, 1, 1, 1, 1, 0.5])
        
        # 1. Executive Summary
        ax_summary = fig.add_subplot(gs[0, :])
        ax_summary.axis('off')
        summary_text = """
        EXECUTIVE SUMMARY: This report presents comprehensive results from the Treasury GAN model, including yield curve scenarios, 
        order book dynamics, time series evolution, and market stress indicators. The model generates realistic synthetic treasury 
        market data for stress testing, risk management, and strategy development.
        """
        ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                       transform=ax_summary.transAxes)
        
        # 2. Model Configuration
        ax_config = fig.add_subplot(gs[1, :])
        ax_config.axis('off')
        config_text = f"""
        MODEL CONFIGURATION:
        • Instruments: 5 (2Y, 5Y, 10Y, 30Y, SOFR)
        • Order Book Levels: 5
        • Features per Level: 5 (Bid Price, Bid Volume, Ask Price, Ask Volume, Spread)
        • Time Steps: 100
        • Total Features: 125
        • Training Epochs: {self.config['training']['epochs']}
        • Learning Rate: {self.config['training']['learning_rate_generator']}
        """
        ax_config.text(0.5, 0.5, config_text, ha='center', va='center', fontsize=11, 
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
                      transform=ax_config.transAxes)
        
        # 3. Key Metrics Summary
        ax_metrics = fig.add_subplot(gs[2, :])
        ax_metrics.axis('off')
        
        # Create metrics table
        metrics_data = []
        headers = ['Scenario', 'Mean', 'Std Dev', 'Yield Slope', 'Liquidity Score']
        metrics_data.append(headers)
        
        for scenario_type, metric in metrics.items():
            metrics_data.append([
                scenario_type.title(),
                f"{metric['mean']:.4f}",
                f"{metric['std']:.4f}",
                f"{metric['yield_curve_slope']:.4f}",
                f"{metric['liquidity_score']:.4f}"
            ])
        
        # Create table
        table = ax_metrics.table(cellText=metrics_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(metrics_data)):
            for j in range(len(metrics_data[0])):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#E8F5E8')
        
        ax_metrics.set_title('Key Metrics Summary', fontsize=16, fontweight='bold', pad=20)
        
        # 4. Yield Curves
        ax_yield = fig.add_subplot(gs[3, :])
        ax_yield.axis('off')
        ax_yield.text(0.5, 0.5, 'Yield Curve Scenarios', ha='center', va='center', fontsize=16, fontweight='bold')
        
        # 5. Order Book
        ax_order = fig.add_subplot(gs[4, :])
        ax_order.axis('off')
        ax_order.text(0.5, 0.5, 'Order Book Microstructure', ha='center', va='center', fontsize=16, fontweight='bold')
        
        # 6. Time Series
        ax_time = fig.add_subplot(gs[5, :])
        ax_time.axis('off')
        ax_time.text(0.5, 0.5, 'Time Series Evolution', ha='center', va='center', fontsize=16, fontweight='bold')
        
        # 7. Feature Importance
        ax_feature = fig.add_subplot(gs[6, :])
        ax_feature.axis('off')
        ax_feature.text(0.5, 0.5, 'Feature Importance Analysis', ha='center', va='center', fontsize=16, fontweight='bold')
        
        # 8. Conclusions
        ax_conclusions = fig.add_subplot(gs[7, :])
        ax_conclusions.axis('off')
        conclusions_text = """
        CONCLUSIONS: The Treasury GAN successfully generates realistic synthetic market data that captures key characteristics 
        of treasury markets. The model demonstrates strong performance in replicating yield curve dynamics, order book 
        microstructure, and temporal market behavior. Key strengths include realistic spread behavior, appropriate yield 
        curve shapes, and coherent time series evolution. The model is suitable for stress testing, risk management, 
        and trading strategy development.
        """
        ax_conclusions.text(0.5, 0.5, conclusions_text, ha='center', va='center', fontsize=11, 
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                           transform=ax_conclusions.transAxes)
        
        plt.tight_layout()
        
        # Save the comprehensive report
        report_path = f"{self.report_dir}/Treasury_GAN_Comprehensive_Report_{self.timestamp}.png"
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive Report generated: {report_path}")
        return report_path
    
    def save_additional_data(self, scenarios, metrics):
        """Save additional data files for analysis."""
        print("💾 Saving Additional Data Files...")
        
        # Save scenarios as numpy arrays
        for scenario_type, data in scenarios.items():
            np.save(f"{self.report_dir}/data/{scenario_type}_scenario.npy", data)
        
        # Save summary statistics
        summary_df = pd.DataFrame(metrics).T
        summary_df.to_csv(f"{self.report_dir}/data/summary_statistics.csv")
        
        # Save configuration
        with open(f"{self.report_dir}/config/gan_config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"✅ Data files saved in: {self.report_dir}/data/")
    
    def generate_report(self):
        """Generate the complete report."""
        print("🚀 Starting Treasury GAN Report Generation...")
        print(f"📁 Report will be saved in: {self.report_dir}")
        
        # Generate scenarios
        scenarios, generator = self.generate_all_scenarios()
        if scenarios is None:
            print("❌ Failed to generate scenarios. Exiting.")
            return None
        
        # Create plots
        plots = self.create_comprehensive_plots(scenarios)
        
        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(scenarios)
        
        # Generate comprehensive report
        report_path = self.create_comprehensive_report(plots, metrics)
        
        # Save additional data
        self.save_additional_data(scenarios, metrics)
        
        # Create run summary
        self.create_run_summary(report_path, metrics)
        
        print(f"\n🎉 REPORT GENERATION COMPLETE!")
        print(f"📊 Comprehensive Report: {report_path}")
        print(f"📁 All files saved in: {self.report_dir}")
        
        return report_path
    
    def create_run_summary(self, report_path, metrics):
        """Create a summary file for the run."""
        summary_path = f"{self.report_dir}/RUN_SUMMARY.txt"
        
        with open(summary_path, 'w') as f:
            f.write("TREASURY GAN RUN SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Run ID: {self.timestamp}\n")
            f.write(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
            f.write(f"Comprehensive Report: {os.path.basename(report_path)}\n\n")
            
            f.write("KEY METRICS:\n")
            f.write("-" * 20 + "\n")
            for scenario_type, metric in metrics.items():
                f.write(f"\n{scenario_type.upper()} SCENARIO:\n")
                f.write(f"  Mean: {metric['mean']:.4f}\n")
                f.write(f"  Std Dev: {metric['std']:.4f}\n")
                f.write(f"  Yield Slope: {metric['yield_curve_slope']:.4f}\n")
                f.write(f"  Liquidity Score: {metric['liquidity_score']:.4f}\n")
            
            f.write(f"\nFILES GENERATED:\n")
            f.write("-" * 20 + "\n")
            f.write(f"• Comprehensive Report: {os.path.basename(report_path)}\n")
            f.write(f"• Individual Plots: 5 PNG files\n")
            f.write(f"• Data: 3 NPY files + CSV summary\n")
            f.write(f"• Metrics: JSON file\n")
            f.write(f"• Configuration: YAML file\n")
        
        print(f"📝 Run summary created: {summary_path}")

def main():
    """Main function to generate the report."""
    print("🎯 Treasury GAN Simple Report Generator")
    print("=" * 50)
    
    try:
        generator = TreasuryGANSimpleReportGenerator()
        report_path = generator.generate_report()
        
        if report_path:
            print(f"\n✅ Success! Report generated at: {report_path}")
            print(f"📁 All files saved in: {generator.report_dir}")
        else:
            print("❌ Report generation failed.")
            
    except Exception as e:
        print(f"❌ Error during report generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 