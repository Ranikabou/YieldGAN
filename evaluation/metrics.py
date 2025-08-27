"""
Evaluation metrics for Treasury GAN models.
Implements various metrics to assess the quality of generated synthetic data.
"""

import numpy as np
# import matplotlib.pyplot as plt  # REMOVED: Matplotlib causes figure generation during evaluation
# import seaborn as sns            # REMOVED: Not needed without matplotlib
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def evaluate_treasury_gan(real_data: np.ndarray, synthetic_data: np.ndarray, 
                         save_plots: bool = True) -> Dict[str, Any]:
    """
    Comprehensive evaluation of Treasury GAN performance.
    
    Args:
        real_data: Real treasury data
        synthetic_data: Generated synthetic data
        save_plots: Whether to save evaluation plots
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Starting GAN evaluation...")
    
    # Ensure data has the same shape
    if real_data.shape != synthetic_data.shape:
        logger.warning(f"Data shape mismatch: real {real_data.shape}, synthetic {synthetic_data.shape}")
        # Reshape synthetic data to match real data
        synthetic_data = synthetic_data[:real_data.shape[0]]
    
    results = {}
    
    # 1. Basic Statistics
    results['basic_stats'] = calculate_basic_statistics(real_data, synthetic_data)
    
    # 2. Distribution Similarity
    results['distribution_metrics'] = calculate_distribution_metrics(real_data, synthetic_data)
    
    # 3. Correlation Analysis
    results['correlation_metrics'] = calculate_correlation_metrics(real_data, synthetic_data)
    
    # 4. Time Series Metrics
    results['timeseries_metrics'] = calculate_timeseries_metrics(real_data, synthetic_data)
    
    # 5. Feature-wise Analysis
    results['feature_analysis'] = analyze_features(real_data, synthetic_data)
    
    # Generate and save plots if requested
    if save_plots:
        results['plots'] = generate_evaluation_plots(real_data, synthetic_data)
    
    logger.info("GAN evaluation completed successfully")
    return results

def calculate_basic_statistics(real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, Any]:
    """Calculate basic statistical measures."""
    stats_dict = {}
    
    # Mean and standard deviation
    stats_dict['real_mean'] = np.mean(real_data, axis=0).tolist()
    stats_dict['real_std'] = np.std(real_data, axis=0).tolist()
    stats_dict['synthetic_mean'] = np.mean(synthetic_data, axis=0).tolist()
    stats_dict['synthetic_std'] = np.std(synthetic_data, axis=0).tolist()
    
    # Min and max values
    stats_dict['real_min'] = np.min(real_data, axis=0).tolist()
    stats_dict['real_max'] = np.max(real_data, axis=0).tolist()
    stats_dict['synthetic_min'] = np.min(synthetic_data, axis=0).tolist()
    stats_dict['synthetic_max'] = np.max(synthetic_data, axis=0).tolist()
    
    # Mean absolute difference
    mean_diff = np.mean(np.abs(real_data - synthetic_data), axis=0)
    stats_dict['mean_absolute_difference'] = mean_diff.tolist()
    
    return stats_dict

def calculate_distribution_metrics(real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, Any]:
    """Calculate distribution similarity metrics."""
    metrics = {}
    
    # Kolmogorov-Smirnov test for each feature
    ks_stats = []
    ks_pvalues = []
    
    for i in range(real_data.shape[1]):
        ks_stat, p_value = stats.ks_2samp(real_data[:, i], synthetic_data[:, i])
        ks_stats.append(ks_stat)
        ks_pvalues.append(p_value)
    
    metrics['ks_statistics'] = ks_stats
    metrics['ks_pvalues'] = ks_pvalues
    
    # Wasserstein distance (Earth Mover's Distance)
    wasserstein_distances = []
    for i in range(real_data.shape[1]):
        wd = stats.wasserstein_distance(real_data[:, i], synthetic_data[:, i])
        wasserstein_distances.append(wd)
    
    metrics['wasserstein_distances'] = wasserstein_distances
    
    # Jensen-Shannon divergence
    js_divergences = []
    for i in range(real_data.shape[1]):
        # Create histograms for comparison
        real_hist, _ = np.histogram(real_data[:, i], bins=50, density=True)
        syn_hist, _ = np.histogram(synthetic_data[:, i], bins=50, density=True)
        
        # Normalize histograms
        real_hist = real_hist / np.sum(real_hist)
        syn_hist = syn_hist / np.sum(syn_hist)
        
        # Calculate JS divergence
        m = 0.5 * (real_hist + syn_hist)
        js_div = 0.5 * (stats.entropy(real_hist, m) + stats.entropy(syn_hist, m))
        js_divergences.append(js_div)
    
    metrics['jensen_shannon_divergences'] = js_divergences
    
    return metrics

def calculate_correlation_metrics(real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, Any]:
    """Calculate correlation-based metrics."""
    metrics = {}
    
    # Feature correlation matrices
    real_corr = np.corrcoef(real_data.T)
    synthetic_corr = np.corrcoef(synthetic_data.T)
    
    metrics['real_correlation_matrix'] = real_corr.tolist()
    metrics['synthetic_correlation_matrix'] = synthetic_corr.tolist()
    
    # Correlation matrix difference
    corr_diff = np.abs(real_corr - synthetic_corr)
    metrics['correlation_difference'] = corr_diff.tolist()
    metrics['mean_correlation_difference'] = np.mean(corr_diff).item()
    
    # Temporal correlation (if data has time dimension)
    if len(real_data.shape) > 2:
        # Calculate lag-1 autocorrelation for each feature
        real_autocorr = []
        syn_autocorr = []
        
        for i in range(real_data.shape[2]):  # features
            real_feature = real_data[:, :, i].flatten()
            syn_feature = synthetic_data[:, :, i].flatten()
            
            if len(real_feature) > 1:
                real_ac = np.corrcoef(real_feature[:-1], real_feature[1:])[0, 1]
                syn_ac = np.corrcoef(syn_feature[:-1], syn_feature[1:])[0, 1]
                
                real_autocorr.append(real_ac if not np.isnan(real_ac) else 0)
                syn_autocorr.append(syn_ac if not np.isnan(syn_ac) else 0)
        
        metrics['real_autocorrelation'] = real_autocorr
        metrics['synthetic_autocorrelation'] = syn_autocorr
    
    return metrics

def calculate_timeseries_metrics(real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, Any]:
    """Calculate time series specific metrics."""
    metrics = {}
    
    # Reshape data if needed
    if len(real_data.shape) == 3:
        # (samples, sequence_length, features)
        real_reshaped = real_data.reshape(-1, real_data.shape[-1])
        syn_reshaped = synthetic_data.reshape(-1, synthetic_data.shape[-1])
    else:
        real_reshaped = real_data
        syn_reshaped = synthetic_data
    
    # Mean squared error
    mse = mean_squared_error(real_reshaped, syn_reshaped)
    metrics['mean_squared_error'] = mse
    
    # Mean absolute error
    mae = mean_absolute_error(real_reshaped, syn_reshaped)
    metrics['mean_absolute_error'] = mae
    
    # Root mean squared error
    rmse = np.sqrt(mse)
    metrics['root_mean_squared_error'] = rmse
    
    # R-squared score
    from sklearn.metrics import r2_score
    r2 = r2_score(real_reshaped, syn_reshaped)
    metrics['r2_score'] = r2
    
    return metrics

def analyze_features(real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, Any]:
    """Analyze individual features."""
    analysis = {}
    
    # Feature-wise statistics
    feature_stats = []
    
    for i in range(real_data.shape[1]):
        real_feature = real_data[:, i]
        syn_feature = synthetic_data[:, i]
        
        feature_stat = {
            'feature_index': i,
            'real_mean': np.mean(real_feature).item(),
            'real_std': np.std(real_feature).item(),
            'synthetic_mean': np.mean(syn_feature).item(),
            'synthetic_std': np.std(syn_feature).item(),
            'mean_difference': np.mean(real_feature - syn_feature).item(),
            'std_difference': np.std(real_feature) - np.std(syn_feature).item()
        }
        
        feature_stats.append(feature_stat)
    
    analysis['feature_statistics'] = feature_stats
    
    # Overall quality score (lower is better)
    quality_scores = []
    for i in range(real_data.shape[1]):
        real_feature = real_data[:, i]
        syn_feature = synthetic_data[:, i]
        
        # Combine multiple metrics for quality score
        mean_diff = np.abs(np.mean(real_feature) - np.mean(syn_feature))
        std_diff = np.abs(np.std(real_feature) - np.std(syn_feature))
        ks_stat, _ = stats.ks_2samp(real_feature, syn_feature)
        
        # Normalize and combine
        quality_score = (mean_diff + std_diff + ks_stat) / 3
        quality_scores.append(quality_score)
    
    analysis['quality_scores'] = quality_scores
    analysis['overall_quality'] = np.mean(quality_scores).item()
    
    return analysis

def generate_evaluation_plots(real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, str]:
    """Generate and save evaluation plots - DISABLED to prevent matplotlib figure generation."""
    plots = {}
    
    try:
        # Create plots directory
        import os
        os.makedirs('results/plots', exist_ok=True)
        
        logger.info("📊 Evaluation plots generation disabled for SSE dashboard compatibility")
        logger.info(f"📈 Evaluation data available: Real data shape: {real_data.shape}, "
                   f"Synthetic data shape: {synthetic_data.shape}")
        
        # REMOVED: All matplotlib plotting to prevent figures from appearing during evaluation
        # All evaluation visualization is now handled by the web dashboard
        
        # Return empty plots dictionary to maintain compatibility
        plots['distribution_comparison'] = 'results/plots/distribution_comparison.png (disabled)'
        plots['correlation_comparison'] = 'results/plots/correlation_comparison.png (disabled)'
        plots['timeseries_comparison'] = 'results/plots/timeseries_comparison.png (disabled)'
        
        logger.info("Evaluation plots generation completed (matplotlib disabled)")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        plots['error'] = str(e)
    
    return plots 