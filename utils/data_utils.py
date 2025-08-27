"""
Data utilities for GAN training on treasury data.
Handles data loading, preprocessing, and data loader creation.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TreasuryDataProcessor:
    """
    Processes treasury data for GAN training.
    """
    
    def __init__(self, instruments: list, sequence_length: int):
        self.instruments = instruments
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
    def prepare_data(self, start_date: str, end_date: str) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Prepare data for training.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Tuple of (sequences, targets, scaler)
        """
        # Generate synthetic treasury data for demonstration
        # In production, this would fetch real data from APIs or databases
        
        logger.info(f"Generating synthetic treasury data from {start_date} to {end_date}")
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        num_days = len(dates)
        
        # Generate synthetic treasury yields (2Y, 5Y, 10Y, 30Y)
        np.random.seed(42)  # For reproducible results
        
        # Base yields with realistic ranges
        base_yields = {
            '2Y': 2.5,
            '5Y': 3.0,
            '10Y': 3.5,
            '30Y': 4.0
        }
        
        # Generate daily yields with some correlation and noise
        yields_data = {}
        for instrument, base_yield in base_yields.items():
            # Add trend and seasonal components
            trend = np.linspace(0, 0.5, num_days)  # Gradual increase
            seasonal = 0.1 * np.sin(2 * np.pi * np.arange(num_days) / 365)  # Annual cycle
            noise = 0.05 * np.random.randn(num_days)  # Daily noise
            
            yields_data[instrument] = base_yield + trend + seasonal + noise
        
        # Create features matrix
        features = np.column_stack([yields_data[inst] for inst in self.instruments])
        
        # Add derived features
        spreads = {
            '10Y-2Y': yields_data['10Y'] - yields_data['2Y'],
            '30Y-10Y': yields_data['30Y'] - yields_data['10Y'],
            '5Y-2Y': yields_data['5Y'] - yields_data['2Y']
        }
        
        # Add spreads to features
        for spread_name, spread_values in spreads.items():
            features = np.column_stack([features, spread_values])
        
        # Add volatility features (rolling standard deviation)
        volatility_window = 20
        volatility_features = []
        for i in range(features.shape[1]):
            vol = pd.Series(features[:, i]).rolling(window=volatility_window).std().fillna(0)
            volatility_features.append(vol.values)
        
        features = np.column_stack([features, np.column_stack(volatility_features)])
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(features_normalized)):
            sequence = features_normalized[i-self.sequence_length:i]
            target = features_normalized[i]
            
            sequences.append(sequence)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"Generated data: {sequences.shape[0]} sequences of shape {sequences.shape[1:]} with {targets.shape[1]} features")
        
        return sequences, targets, self.scaler

def create_data_loaders(sequences: np.ndarray, targets: np.ndarray, 
                        batch_size: int = 32, train_split: float = 0.7, 
                        val_split: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        sequences: Input sequences
        targets: Target values
        batch_size: Batch size for training
        train_split: Training set proportion
        val_split: Validation set proportion
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Calculate split indices
    total_samples = len(sequences)
    train_end = int(total_samples * train_split)
    val_end = int(total_samples * (train_split + val_split))
    
    # Split data
    train_sequences = sequences[:train_end]
    train_targets = targets[:train_end]
    
    val_sequences = sequences[train_end:val_end]
    val_targets = targets[train_end:val_end]
    
    test_sequences = sequences[val_end:]
    test_targets = targets[val_end:]
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(train_sequences),
        torch.FloatTensor(train_targets)
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(val_sequences),
        torch.FloatTensor(val_targets)
    )
    
    test_dataset = TensorDataset(
        torch.FloatTensor(test_sequences),
        torch.FloatTensor(test_targets)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_loader.dataset)} samples")
    logger.info(f"  Validation: {len(val_loader.dataset)} samples")
    logger.info(f"  Test: {len(test_loader.dataset)} samples")
    
    return train_loader, val_loader, test_loader 