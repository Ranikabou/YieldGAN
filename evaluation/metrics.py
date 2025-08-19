"""
Evaluation metrics for treasury GAN synthetic data.
Implements statistical tests, distribution comparisons, and financial metrics.
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TreasuryDataEvaluator:
    """
    Evaluates the quality of synthetic treasury data against real data.
    """
    
    def __init__(self, real_data: np.ndarray, synthetic_data: np.ndarray):
        """
        Initialize evaluator with real and synthetic data.
        
        Args:
            real_data: Real treasury data array
            synthetic_data: Synthetic treasury data array
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        
        # Ensure data has same shape
        if real_data.shape != synthetic_data.shape:
            raise ValueError("Real and synthetic data must have the same shape")
        
        self.n_samples, self.sequence_length, self.n_features = real_data.shape
        
    def basic_statistics(self) -> Dict[str, np.ndarray]:
        """
        Calculate basic statistics for both datasets.
        
        Returns:
            Dictionary with mean, std, min, max for each dataset
        """
        logger.info("Calculating basic statistics")
        
        # Flatten data for statistics
        real_flat = self.real_data.reshape(-1, self.n_features)
        synthetic_flat = self.synthetic_data.reshape(-1, self.n_features)
        
        stats_dict = {
            'real': {
                'mean': np.mean(real_flat, axis=0),
                'std': np.std(real_flat, axis=0),
                'min': np.min(real_flat, axis=0),
                'max': np.max(real_flat, axis=0),
                'median': np.median(real_flat, axis=0)
            },
            'synthetic': {
                'mean': np.mean(synthetic_flat, axis=0),
                'std': np.std(synthetic_flat, axis=0),
                'min': np.min(synthetic_flat, axis=0),
                'max': np.max(synthetic_flat, axis=0),
                'median': np.median(synthetic_flat, axis=0)
            }
        }
        
        return stats_dict
    
    def distribution_similarity(self) -> Dict[str, np.ndarray]:
        """
        Calculate distribution similarity metrics.
        
        Returns:
            Dictionary with KS test statistics and p-values
        """
        logger.info("Calculating distribution similarity")
        
        real_flat = self.real_data.reshape(-1, self.n_features)
        synthetic_flat = self.synthetic_data.reshape(-1, self.n_features)
        
        ks_stats = []
        ks_pvalues = []
        wasserstein_distances = []
        
        for feature_idx in range(self.n_features):
            real_feature = real_flat[:, feature_idx]
            synthetic_feature = synthetic_flat[:, feature_idx]
            
            # KS test
            ks_stat, ks_pvalue = ks_2samp(real_feature, synthetic_feature)
            ks_stats.append(ks_stat)
            ks_pvalues.append(ks_pvalue)
            
            # Wasserstein distance
            w_dist = wasserstein_distance(real_feature, synthetic_feature)
            wasserstein_distances.append(w_dist)
        
        return {
            'ks_statistics': np.array(ks_stats),
            'ks_pvalues': np.array(ks_pvalues),
            'wasserstein_distances': np.array(wasserstein_distances)
        }
    
    def correlation_analysis(self) -> Dict[str, np.ndarray]:
        """
        Analyze correlations between features.
        
        Returns:
            Dictionary with correlation matrices
        """
        logger.info("Analyzing feature correlations")
        
        real_flat = self.real_data.reshape(-1, self.n_features)
        synthetic_flat = self.synthetic_data.reshape(-1, self.n_features)
        
        # Handle NaN values
        real_flat = np.nan_to_num(real_flat, nan=0.0)
        synthetic_flat = np.nan_to_num(synthetic_flat, nan=0.0)
        
        real_corr = np.corrcoef(real_flat.T)
        synthetic_corr = np.corrcoef(synthetic_flat.T)
        
        # Handle NaN values in correlation matrices
        real_corr = np.nan_to_num(real_corr, nan=0.0)
        synthetic_corr = np.nan_to_num(synthetic_corr, nan=0.0)
        
        # Correlation difference
        corr_diff = np.abs(real_corr - synthetic_corr)
        
        # Calculate MSE only if both matrices are finite
        if np.all(np.isfinite(real_corr)) and np.all(np.isfinite(synthetic_corr)):
            correlation_mse = mean_squared_error(real_corr.flatten(), synthetic_corr.flatten())
        else:
            correlation_mse = float('inf')
        
        return {
            'real_correlation': real_corr,
            'synthetic_correlation': synthetic_corr,
            'correlation_difference': corr_diff,
            'correlation_mse': correlation_mse
        }
    
    def temporal_dynamics(self) -> Dict[str, np.ndarray]:
        """
        Analyze temporal dynamics and autocorrelations.
        
        Returns:
            Dictionary with autocorrelation metrics
        """
        logger.info("Analyzing temporal dynamics")
        
        # Calculate autocorrelations for each feature
        real_autocorr = []
        synthetic_autocorr = []
        
        for feature_idx in range(self.n_features):
            real_feature_seq = self.real_data[:, :, feature_idx]  # (n_samples, sequence_length)
            synthetic_feature_seq = self.synthetic_data[:, :, feature_idx]
            
            # Calculate autocorrelation for each sample
            real_ac = []
            synthetic_ac = []
            
            for sample_idx in range(min(10, self.n_samples)):  # Use first 10 samples
                real_ac.append(np.corrcoef(real_feature_seq[sample_idx, :-1], 
                                         real_feature_seq[sample_idx, 1:])[0, 1])
                synthetic_ac.append(np.corrcoef(synthetic_feature_seq[sample_idx, :-1], 
                                              synthetic_feature_seq[sample_idx, 1:])[0, 1])
            
            real_autocorr.append(np.mean(real_ac))
            synthetic_autocorr.append(np.mean(synthetic_ac))
        
        return {
            'real_autocorrelation': np.array(real_autocorr),
            'synthetic_autocorrelation': np.array(synthetic_autocorr),
            'autocorrelation_difference': np.abs(np.array(real_autocorr) - np.array(synthetic_autocorr))
        }
    
    def financial_metrics(self) -> Dict[str, np.ndarray]:
        """
        Calculate financial-specific metrics.
        
        Returns:
            Dictionary with financial metrics
        """
        logger.info("Calculating financial metrics")
        
        real_flat = self.real_data.reshape(-1, self.n_features)
        synthetic_flat = self.synthetic_data.reshape(-1, self.n_features)
        
        # Volatility (standard deviation)
        real_vol = np.std(real_flat, axis=0)
        synthetic_vol = np.std(synthetic_flat, axis=0)
        
        # Skewness
        real_skew = stats.skew(real_flat, axis=0)
        synthetic_skew = stats.skew(synthetic_flat, axis=0)
        
        # Kurtosis
        real_kurt = stats.kurtosis(real_flat, axis=0)
        synthetic_kurt = stats.kurtosis(synthetic_flat, axis=0)
        
        # Value at Risk (95%)
        real_var = np.percentile(real_flat, 5, axis=0)
        synthetic_var = np.percentile(synthetic_flat, 5, axis=0)
        
        return {
            'real_volatility': real_vol,
            'synthetic_volatility': synthetic_vol,
            'volatility_difference': np.abs(real_vol - synthetic_vol),
            'real_skewness': real_skew,
            'synthetic_skewness': synthetic_skew,
            'skewness_difference': np.abs(real_skew - synthetic_skew),
            'real_kurtosis': real_kurt,
            'synthetic_kurtosis': synthetic_kurt,
            'kurtosis_difference': np.abs(real_kurt - synthetic_kurt),
            'real_var_95': real_var,
            'synthetic_var_95': synthetic_var,
            'var_difference': np.abs(real_var - synthetic_var)
        }
    
    def feature_importance_analysis(self) -> Dict[str, np.ndarray]:
        """
        Analyze which features are most important for discrimination.
        
        Returns:
            Dictionary with feature importance metrics
        """
        logger.info("Analyzing feature importance")
        
        # Calculate feature-wise statistics
        real_flat = self.real_data.reshape(-1, self.n_features)
        synthetic_flat = self.synthetic_data.reshape(-1, self.n_features)
        
        feature_importance = []
        
        for feature_idx in range(self.n_features):
            real_feature = real_flat[:, feature_idx]
            synthetic_feature = synthetic_flat[:, feature_idx]
            
            # Calculate separation metric (higher = better separation)
            combined = np.concatenate([real_feature, synthetic_feature])
            labels = np.concatenate([np.ones(len(real_feature)), np.zeros(len(synthetic_feature))])
            
            # Calculate F-statistic (ANOVA)
            f_stat, _ = stats.f_oneway(real_feature, synthetic_feature)
            feature_importance.append(f_stat)
        
        return {
            'feature_importance': np.array(feature_importance),
            'feature_ranking': np.argsort(feature_importance)[::-1]
        }
    
    def comprehensive_evaluation(self) -> Dict[str, Dict]:
        """
        Run comprehensive evaluation of all metrics.
        
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Running comprehensive evaluation")
        
        results = {}
        
        # Basic statistics
        results['basic_statistics'] = self.basic_statistics()
        
        # Distribution similarity
        results['distribution_similarity'] = self.distribution_similarity()
        
        # Correlation analysis
        results['correlation_analysis'] = self.correlation_analysis()
        
        # Temporal dynamics
        results['temporal_dynamics'] = self.temporal_dynamics()
        
        # Financial metrics
        results['financial_metrics'] = self.financial_metrics()
        
        # Feature importance
        results['feature_importance'] = self.feature_importance_analysis()
        
        # Overall quality score
        results['overall_quality'] = self._calculate_overall_quality(results)
        
        return results
    
    def _calculate_overall_quality(self, results: Dict) -> float:
        """
        Calculate overall quality score.
        
        Args:
            results: Dictionary with all evaluation results
            
        Returns:
            Overall quality score (0-1, higher is better)
        """
        # Normalize different metrics to 0-1 scale
        scores = []
        
        # Distribution similarity (lower KS stat is better)
        ks_scores = 1 - results['distribution_similarity']['ks_statistics']
        scores.append(np.mean(ks_scores))
        
        # Correlation similarity (lower difference is better)
        corr_scores = 1 - np.tanh(results['correlation_analysis']['correlation_difference'].mean())
        scores.append(corr_scores)
        
        # Temporal dynamics (lower difference is better)
        temp_scores = 1 - np.tanh(results['temporal_dynamics']['autocorrelation_difference'].mean())
        scores.append(temp_scores)
        
        # Financial metrics (lower difference is better)
        fin_scores = 1 - np.tanh(results['financial_metrics']['volatility_difference'].mean())
        scores.append(fin_scores)
        
        # Overall score is average of all normalized scores
        overall_score = np.mean(scores)
        
        return overall_score
    
    def plot_evaluation_results(self, results: Dict, save_path: str = None) -> None:
        """
        Plot comprehensive evaluation results optimized for tighter screens.
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save plots
        """
        logger.info("Creating evaluation plots")
        
        # Set figure size optimized for tighter screens
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Treasury GAN Evaluation Results', fontsize=16)
        
        # Get number of features and create optimized x-axis labels
        n_features = self.n_features
        feature_indices = np.arange(n_features)
        
        # Reduce x-axis labels for better readability on tight screens
        if n_features > 50:
            # Show every 10th label for large feature sets
            label_step = max(1, n_features // 20)
            x_labels = [f'F{i}' if i % label_step == 0 else '' for i in range(n_features)]
        else:
            # Show every 5th label for smaller feature sets
            label_step = max(1, n_features // 10)
            x_labels = [f'F{i}' if i % label_step == 0 else '' for i in range(n_features)]
        
        # Basic statistics comparison
        basic_stats = results['basic_statistics']
        
        axes[0, 0].bar(feature_indices, basic_stats['real']['mean'], width=0.8, 
                       label='Real', alpha=0.7, color='#1f77b4')
        axes[0, 0].bar(feature_indices, basic_stats['synthetic']['mean'], width=0.8, 
                       label='Synthetic', alpha=0.7, color='#ff7f0e')
        axes[0, 0].set_title('Feature Means Comparison', fontsize=12)
        axes[0, 0].set_xlabel('Features')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].set_xticks(feature_indices[::label_step])
        axes[0, 0].set_xticklabels([f'F{i}' for i in feature_indices[::label_step]], rotation=45, fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution similarity
        dist_sim = results['distribution_similarity']
        axes[0, 1].bar(feature_indices, dist_sim['ks_statistics'], 
                       color='#2ca02c', alpha=0.7)
        axes[0, 1].set_title('KS Test Statistics\n(Lower is Better)', fontsize=12)
        axes[0, 1].set_xlabel('Features')
        axes[0, 1].set_ylabel('KS Statistic')
        axes[0, 1].set_xticks(feature_indices[::label_step])
        axes[0, 1].set_xticklabels([f'F{i}' for i in feature_indices[::label_step]], rotation=45, fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Correlation difference
        corr_diff = results['correlation_analysis']['correlation_difference']
        im = axes[0, 2].imshow(corr_diff, cmap='viridis', aspect='auto', 
                               vmin=0, vmax=np.percentile(corr_diff, 95))
        axes[0, 2].set_title('Correlation Matrix\nDifference', fontsize=12)
        axes[0, 2].set_xlabel('Features')
        axes[0, 2].set_ylabel('Features')
        # Reduce tick labels for correlation matrix
        tick_step = max(1, n_features // 8)
        axes[0, 2].set_xticks(np.arange(0, n_features, tick_step))
        axes[0, 2].set_yticks(np.arange(0, n_features, tick_step))
        axes[0, 2].set_xticklabels([f'F{i}' for i in range(0, n_features, tick_step)], fontsize=8)
        axes[0, 2].set_yticklabels([f'F{i}' for i in range(0, n_features, tick_step)], fontsize=8)
        plt.colorbar(im, ax=axes[0, 2], shrink=0.8)
        
        # Temporal dynamics
        temp_dyn = results['temporal_dynamics']
        axes[1, 0].bar(feature_indices, temp_dyn['autocorrelation_difference'], 
                       color='#d62728', alpha=0.7)
        axes[1, 0].set_title('Autocorrelation\nDifference', fontsize=12)
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Difference')
        axes[1, 0].set_xticks(feature_indices[::label_step])
        axes[1, 0].set_xticklabels([f'F{i}' for i in feature_indices[::label_step]], rotation=45, fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Financial metrics (volatility difference)
        fin_metrics = results['financial_metrics']
        axes[1, 1].bar(feature_indices, fin_metrics['volatility_difference'], 
                       color='#9467bd', alpha=0.7)
        axes[1, 1].set_title('Volatility\nDifference', fontsize=12)
        axes[1, 0].set_xlabel('Features')
        axes[1, 1].set_ylabel('Difference')
        axes[1, 1].set_xticks(feature_indices[::label_step])
        axes[1, 1].set_xticklabels([f'F{i}' for i in feature_indices[::label_step]], rotation=45, fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Feature importance
        feat_imp = results['feature_importance']['feature_importance']
        axes[1, 2].bar(feature_indices, feat_imp, color='#8c564b', alpha=0.7)
        axes[1, 2].set_title('Feature Importance\n(F-statistic)', fontsize=12)
        axes[1, 2].set_xlabel('Features')
        axes[1, 2].set_ylabel('F-statistic')
        axes[1, 2].set_xticks(feature_indices[::label_step])
        axes[1, 2].set_xticklabels([f'F{i}' for i in feature_indices[::label_step]], rotation=45, fontsize=8)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Optimize layout for tight screens
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        plt.show()
        
        # Print overall quality score with better formatting
        overall_score = results['overall_quality']
        print(f"\n{'='*60}")
        print(f"🎯 TREASURY GAN EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"📊 Overall Quality Score: {overall_score:.4f} (0-1 scale)")
        
        if overall_score > 0.8:
            print(f"🏆 EXCELLENT synthetic data quality!")
        elif overall_score > 0.6:
            print(f"✅ GOOD synthetic data quality.")
        elif overall_score > 0.4:
            print(f"⚠️  MODERATE synthetic data quality.")
        else:
            print(f"❌ POOR synthetic data quality - consider retraining.")
        
        print(f"\n📈 Key Metrics:")
        print(f"   • Distribution Similarity: {np.mean(results['distribution_similarity']['ks_statistics']):.4f}")
        print(f"   • Correlation Preservation: {results['correlation_analysis']['correlation_mse']:.4f}")
        print(f"   • Temporal Dynamics: {np.mean(results['temporal_dynamics']['autocorrelation_difference']):.4f}")
        print(f"{'='*60}")

def evaluate_treasury_gan(real_data: np.ndarray, synthetic_data: np.ndarray, 
                         save_plots: bool = True) -> Dict[str, Dict]:
    """
    Convenience function to run complete evaluation.
    
    Args:
        real_data: Real treasury data
        synthetic_data: Synthetic treasury data
        save_plots: Whether to save evaluation plots
        
    Returns:
        Complete evaluation results
    """
    evaluator = TreasuryDataEvaluator(real_data, synthetic_data)
    results = evaluator.comprehensive_evaluation()
    
    if save_plots:
        evaluator.plot_evaluation_results(results, 'evaluation_results.png')
    
    return results 