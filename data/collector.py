"""
Data collection script for treasury curve GAN project.
Fetches treasury data and simulates order book data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os
import sys
from typing import Dict, List, Optional
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import TreasuryDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_treasury_data_detailed(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch detailed treasury data including yields, prices, and volumes.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date for data collection
        
    Returns:
        DataFrame with treasury data
    """
    logger.info(f"Fetching treasury data from {start_date} to {end_date}")
    
    # Treasury instruments to fetch - using working ticker symbols
    instruments = {
        '2Y': '^UST2YR',
        '5Y': '^UST5YR', 
        '10Y': '^UST10YR',
        '30Y': '^UST30YR',
        'SOFR': '^SOFR'
    }
    
    data = {}
    
    for instrument_name, ticker in instruments.items():
        try:
            logger.info(f"Fetching data for {instrument_name} ({ticker})")
            
            # Download data
            ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not ticker_data.empty:
                # Calculate additional metrics
                ticker_data['Yield'] = ticker_data['Close']
                ticker_data['Price'] = 100 / (1 + ticker_data['Close']/100)  # Convert yield to price
                ticker_data['Volume'] = ticker_data['Volume'].fillna(0)
                
                # Calculate daily returns
                ticker_data['Returns'] = ticker_data['Close'].pct_change()
                
                # Calculate volatility (rolling 20-day)
                ticker_data['Volatility'] = ticker_data['Returns'].rolling(window=20).std()
                
                # Store with instrument prefix
                for col in ticker_data.columns:
                    data[f"{instrument_name}_{col}"] = ticker_data[col]
                    
                logger.info(f"Successfully fetched {len(ticker_data)} data points for {instrument_name}")
            else:
                logger.warning(f"No data found for {instrument_name}")
                
        except Exception as e:
            logger.error(f"Error fetching data for {instrument_name}: {e}")
    
    df = pd.DataFrame(data)
    
    # If no data was fetched, create synthetic data for demonstration
    if df.empty:
        logger.warning("No real data fetched, creating synthetic data for demonstration")
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        synthetic_data = {}
        
        for instrument_name in instruments.keys():
            # Generate realistic synthetic yields
            base_yield = {
                '2Y': 4.5, '5Y': 4.2, '10Y': 4.0, '30Y': 3.8, 'SOFR': 5.3
            }.get(instrument_name, 4.0)
            
            # Add some realistic variation
            np.random.seed(42)  # For reproducible results
            yields = base_yield + np.random.normal(0, 0.1, len(date_range))
            synthetic_data[f"{instrument_name}_Yield"] = yields
            synthetic_data[f"{instrument_name}_Price"] = 100 / (1 + yields/100)
            synthetic_data[f"{instrument_name}_Volume"] = np.random.exponential(1000, len(date_range))
            synthetic_data[f"{instrument_name}_Returns"] = np.random.normal(0, 0.01, len(date_range))
            synthetic_data[f"{instrument_name}_Volatility"] = np.random.exponential(0.01, len(date_range))
        
        df = pd.DataFrame(synthetic_data, index=date_range)
        logger.info("Created synthetic treasury data for demonstration")
    
    df = df.dropna()
    
    logger.info(f"Total data shape: {df.shape}")
    return df

def simulate_market_microstructure(treasury_data: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate realistic market microstructure including bid-ask spreads, volumes, and order book dynamics.
    
    Args:
        treasury_data: DataFrame with treasury yields and prices
        
    Returns:
        DataFrame with simulated order book data
    """
    logger.info("Simulating market microstructure")
    
    order_book_data = []
    
    for timestamp, row in treasury_data.iterrows():
        for instrument in ['2Y', '5Y', '10Y', '30Y', 'SOFR']:
            # Get base yield for this instrument
            yield_col = f"{instrument}_Yield"
            if yield_col not in row.index:
                continue
                
            base_yield = row[yield_col]
            base_price = row.get(f"{instrument}_Price", 100 / (1 + base_yield/100))
            
            # Market conditions affect spreads
            volatility = row.get(f"{instrument}_Volatility", 0.01)
            volume = row.get(f"{instrument}_Volume", 1000)
            
            # Simulate 5 levels of order book
            for level in range(5):
                # Level-dependent spreads (tighter at top levels)
                level_multiplier = (level + 1) * 0.5
                
                # Volatility affects spread width
                spread_basis = max(0.001, volatility * level_multiplier)
                
                # Bid side (lower yields = higher prices)
                bid_spread = spread_basis * (1 + np.random.normal(0, 0.1))
                bid_yield = base_yield + bid_spread
                bid_price = 100 / (1 + bid_yield/100)
                
                # Volume decreases with level (more liquidity at top)
                volume_multiplier = max(0.1, (5 - level) / 5)
                bid_volume = max(100, volume * volume_multiplier * np.random.exponential(1))
                
                # Ask side (higher yields = lower prices)
                ask_spread = spread_basis * (1 + np.random.normal(0, 0.1))
                ask_yield = base_yield - ask_spread
                ask_price = 100 / (1 + ask_yield/100)
                ask_volume = max(100, volume * volume_multiplier * np.random.exponential(1))
                
                # Mid price and spread
                mid_price = (bid_price + ask_price) / 2
                spread = ask_yield - bid_yield
                
                # Order book imbalance
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                
                order_book_data.append({
                    'timestamp': timestamp,
                    'instrument': instrument,
                    'level': level,
                    'bid_price': bid_price,
                    'bid_volume': bid_volume,
                    'ask_price': ask_price,
                    'ask_volume': ask_volume,
                    'mid_price': mid_price,
                    'spread': spread,
                    'imbalance': imbalance,
                    'base_yield': base_yield,
                    'volatility': volatility,
                    'base_volume': volume
                })
    
    df = pd.DataFrame(order_book_data)
    logger.info(f"Generated {len(df)} order book records")
    return df

def create_enhanced_features(order_book_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create enhanced features for GAN training.
    
    Args:
        order_book_data: DataFrame with order book data
        
    Returns:
        DataFrame with enhanced features
    """
    logger.info("Creating enhanced features")
    
    # Pivot to get features for each timestamp
    features = order_book_data.pivot_table(
        index='timestamp',
        columns=['instrument', 'level'],
        values=['bid_price', 'bid_volume', 'ask_price', 'ask_volume', 'spread', 'imbalance'],
        aggfunc='first'
    )
    
    # Flatten column names
    features.columns = [f"{col[0]}_{col[1]}_{col[2]}" for col in features.columns]
    
    # Fill missing values
    features = features.fillna(method='ffill').fillna(method='bfill')
    
    # Add derived features
    for instrument in ['2Y', '5Y', '10Y', '30Y', 'SOFR']:
        # Yield curve slope (10Y - 2Y)
        if f'2Y_Yield_0' in features.columns and f'10Y_Yield_0' in features.columns:
            features[f'{instrument}_curve_slope'] = features[f'10Y_Yield_0'] - features[f'2Y_Yield_0']
        
        # Volume-weighted average spread
        spread_cols = [col for col in features.columns if 'spread' in col and instrument in col]
        volume_cols = [col for col in features.columns if 'bid_volume' in col and instrument in col]
        
        if spread_cols and volume_cols:
            features[f'{instrument}_vwap_spread'] = (
                features[spread_cols].multiply(features[volume_cols]).sum(axis=1) / 
                features[volume_cols].sum(axis=1)
            )
    
    # Add market-wide features
    features['total_volume'] = features[[col for col in features.columns if 'volume' in col]].sum(axis=1)
    features['avg_spread'] = features[[col for col in features.columns if 'spread' in col]].mean(axis=1)
    
    logger.info(f"Created feature matrix with shape {features.shape}")
    return features

def save_data(data: pd.DataFrame, filename: str, data_dir: str = 'data') -> None:
    """
    Save data to file.
    
    Args:
        data: DataFrame to save
        filename: Output filename
        data_dir: Directory to save data
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    if filename.endswith('.csv'):
        data.to_csv(filepath)
    elif filename.endswith('.parquet'):
        data.to_parquet(filepath)
    elif filename.endswith('.pkl'):
        data.to_pickle(filepath)
    else:
        data.to_csv(filepath + '.csv')
    
    logger.info(f"Data saved to {filepath}")

def main():
    """Main data collection function."""
    parser = argparse.ArgumentParser(description='Collect treasury data for GAN training')
    parser.add_argument('--start-date', type=str, default='2022-01-01', 
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-01-01', 
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-format', type=str, default='parquet',
                       choices=['csv', 'parquet', 'pkl'], help='Output file format')
    parser.add_argument('--enhance-features', action='store_true',
                       help='Create enhanced features')
    
    args = parser.parse_args()
    
    # Fetch treasury data
    treasury_data = fetch_treasury_data_detailed(args.start_date, args.end_date)
    
    if treasury_data.empty:
        logger.error("No treasury data collected. Exiting.")
        return
    
    # Save raw treasury data
    save_data(treasury_data, f'treasury_data_{args.start_date}_{args.end_date}.{args.output_format}')
    
    # Simulate order book data
    order_book_data = simulate_market_microstructure(treasury_data)
    save_data(order_book_data, f'order_book_data_{args.start_date}_{args.end_date}.{args.output_format}')
    
    # Create enhanced features if requested
    if args.enhance_features:
        enhanced_features = create_enhanced_features(order_book_data)
        save_data(enhanced_features, f'enhanced_features_{args.start_date}_{args.end_date}.{args.output_format}')
    
    # Create processed data for GAN training
    processor = TreasuryDataProcessor(
        instruments=['2Y', '5Y', '10Y', '30Y', 'SOFR'],
        sequence_length=100
    )
    
    sequences, targets, scaler = processor.prepare_data(args.start_date, args.end_date)
    
    # Save processed data
    np.save('data/sequences.npy', sequences)
    np.save('data/targets.npy', targets)
    
    logger.info("Data collection completed successfully!")
    logger.info(f"Sequences shape: {sequences.shape}")
    logger.info(f"Targets shape: {targets.shape}")

if __name__ == "__main__":
    main() 