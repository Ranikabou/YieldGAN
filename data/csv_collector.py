"""
CSV data collector for Treasury GAN training.
Handles loading and processing of CSV data files.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class CSVDataCollector:
    """
    Collects and processes CSV data for GAN training.
    """
    
    def __init__(self, csv_directory: str):
        self.csv_directory = csv_directory
        self.scaler = StandardScaler()
        
    def collect_and_process(self, sequence_length: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect and process CSV data for training.
        
        Args:
            sequence_length: Length of sequences for training
            
        Returns:
            Tuple of (sequences, targets)
        """
        logger.info(f"Processing CSV data from {self.csv_directory}")
        
        # List all CSV files
        csv_files = [f for f in os.listdir(self.csv_directory) if f.endswith('.csv')]
        logger.info(f"Found CSV files: {csv_files}")
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_directory}")
        
        # Load and process each CSV file
        data_frames = {}
        for csv_file in csv_files:
            file_path = os.path.join(self.csv_directory, csv_file)
            try:
                df = pd.read_csv(file_path)
                data_frames[csv_file] = df
                logger.info(f"Loaded {csv_file}: {df.shape}")
            except Exception as e:
                logger.warning(f"Could not load {csv_file}: {e}")
        
        if not data_frames:
            raise ValueError("No CSV files could be loaded successfully")
        
        # Process the data
        sequences, targets = self._process_csv_data(data_frames, sequence_length)
        
        return sequences, targets
    
    def _process_csv_data(self, data_frames: dict, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process CSV data into sequences and targets.
        
        Args:
            data_frames: Dictionary of loaded CSV dataframes
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (sequences, targets)
        """
        # For demonstration, we'll create synthetic data if no real CSV data is provided
        # In production, this would process the actual CSV data
        
        # Check if we have meaningful data
        has_real_data = False
        for name, df in data_frames.items():
            if len(df) > sequence_length and df.shape[1] > 1:
                has_real_data = True
                break
        
        if not has_real_data:
            logger.info("No suitable CSV data found, generating synthetic data for demonstration")
            return self._generate_synthetic_data(sequence_length)
        
        # Process real CSV data
        logger.info("Processing real CSV data...")
        
        # Combine all dataframes
        combined_data = []
        for name, df in data_frames.items():
            if len(df) > sequence_length:
                # Select numeric columns only
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    combined_data.append(df[numeric_cols].values)
        
        if not combined_data:
            logger.info("No suitable numeric data found, generating synthetic data")
            return self._generate_synthetic_data(sequence_length)
        
        # Concatenate all data
        all_data = np.concatenate(combined_data, axis=1)
        
        # Remove rows with NaN values
        all_data = all_data[~np.isnan(all_data).any(axis=1)]
        
        if len(all_data) < sequence_length + 1:
            logger.info("Insufficient data after cleaning, generating synthetic data")
            return self._generate_synthetic_data(sequence_length)
        
        # Normalize data
        all_data_normalized = self.scaler.fit_transform(all_data)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(all_data_normalized)):
            sequence = all_data_normalized[i-sequence_length:i]
            target = all_data_normalized[i]
            
            sequences.append(sequence)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"Processed CSV data: {sequences.shape[0]} sequences of shape {sequences.shape[1:]}")
        
        return sequences, targets
    
    def _generate_synthetic_data(self, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic treasury data for demonstration.
        
        Args:
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (sequences, targets)
        """
        logger.info("Generating synthetic treasury data for demonstration")
        
        # Generate synthetic treasury yields (2Y, 5Y, 10Y, 30Y)
        np.random.seed(42)  # For reproducible results
        
        # Parameters
        num_days = 1000
        num_features = 7  # 4 yields + 3 spreads
        
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
        features = np.column_stack([yields_data[inst] for inst in base_yields.keys()])
        
        # Add derived features (spreads)
        spreads = {
            '10Y-2Y': yields_data['10Y'] - yields_data['2Y'],
            '30Y-10Y': yields_data['30Y'] - yields_data['10Y'],
            '5Y-2Y': yields_data['5Y'] - yields_data['2Y']
        }
        
        # Add spreads to features
        for spread_name, spread_values in spreads.items():
            features = np.column_stack([features, spread_values])
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(features_normalized)):
            sequence = features_normalized[i-sequence_length:i]
            target = features_normalized[i]
            
            sequences.append(sequence)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"Generated synthetic data: {sequences.shape[0]} sequences of shape {sequences.shape[1:]} with {targets.shape[1]} features")
        
        return sequences, targets
    
    def save_processed_data(self, sequences: np.ndarray, targets: np.ndarray, 
                           output_dir: str = 'data/processed'):
        """
        Save processed data to files.
        
        Args:
            sequences: Input sequences
            targets: Target values
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, 'sequences.npy'), sequences)
        np.save(os.path.join(output_dir, 'targets.npy'), targets)
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        
        logger.info(f"Processed data saved to {output_dir}")
    
    def load_processed_data(self, data_dir: str = 'data/processed') -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Load previously processed data.
        
        Args:
            data_dir: Directory containing processed data
            
        Returns:
            Tuple of (sequences, targets, scaler)
        """
        sequences_path = os.path.join(data_dir, 'sequences.npy')
        targets_path = os.path.join(data_dir, 'targets.npy')
        scaler_path = os.path.join(data_dir, 'scaler.pkl')
        
        if not all(os.path.exists(p) for p in [sequences_path, targets_path, scaler_path]):
            raise FileNotFoundError(f"Processed data not found in {data_dir}")
        
        sequences = np.load(sequences_path)
        targets = np.load(targets_path)
        
        import joblib
        scaler = joblib.load(scaler_path)
        
        logger.info(f"Loaded processed data: {sequences.shape[0]} sequences")
        
        return sequences, targets, scaler 