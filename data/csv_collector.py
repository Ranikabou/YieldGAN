#!/usr/bin/env python3
"""
CSV-based data collector for treasury curve GAN project.
Loads data from CSV files instead of fetching from APIs.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVDataCollector:
    """
    Collects and processes treasury data from CSV files.
    """
    
    def __init__(self, data_directory: str = "data/csv"):
        """
        Initialize CSV data collector.
        
        Args:
            data_directory: Directory containing CSV data files
        """
        self.data_directory = data_directory
        self.required_columns = {
            'treasury': ['date', '2Y', '5Y', '10Y', '30Y', 'SOFR'],
            'order_book': ['timestamp', 'instrument', 'level', 'bid_price', 'bid_size', 'ask_price', 'ask_size'],
            'features': ['date', 'feature_1', 'feature_2', 'feature_3']  # Customize based on your CSV structure
        }
        
    def load_csv_data(self, file_pattern: str = "*.csv") -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from the data directory.
        
        Args:
            file_pattern: Glob pattern to match CSV files
            
        Returns:
            Dictionary mapping file types to DataFrames
        """
        csv_files = glob.glob(os.path.join(self.data_directory, file_pattern))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.data_directory}")
            return {}
        
        data = {}
        
        for file_path in csv_files:
            try:
                file_name = os.path.basename(file_path)
                logger.info(f"Loading CSV file: {file_name}")
                
                # Load CSV with flexible parsing
                df = pd.read_csv(file_path, parse_dates=True, infer_datetime_format=True)
                
                # Determine file type based on filename or content
                file_type = self._identify_file_type(file_name, df.columns)
                
                if file_type:
                    data[file_type] = df
                    logger.info(f"Successfully loaded {file_type} data: {df.shape}")
                else:
                    logger.warning(f"Could not identify type for {file_name}")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return data
    
    def _identify_file_type(self, filename: str, columns: List[str]) -> Optional[str]:
        """
        Identify the type of CSV file based on filename or columns.
        
        Args:
            filename: Name of the CSV file
            columns: Column names in the CSV
            
        Returns:
            File type identifier or None if unknown
        """
        filename_lower = filename.lower()
        columns_lower = [col.lower() for col in columns]
        
        # Check for treasury yield data
        if any(col in columns_lower for col in ['2y', '5y', '10y', '30y', 'sofr', 'yield']):
            return 'treasury'
        
        # Check for order book data
        if any(col in columns_lower for col in ['bid', 'ask', 'level', 'order_book']):
            return 'order_book'
        
        # Check for feature data
        if any(col in columns_lower for col in ['feature', 'indicator', 'metric']):
            return 'features'
        
        # Check filename patterns
        if 'treasury' in filename_lower or 'yield' in filename_lower:
            return 'treasury'
        elif 'order' in filename_lower or 'book' in filename_lower:
            return 'order_book'
        elif 'feature' in filename_lower or 'indicator' in filename_lower:
            return 'features'
        
        return None
    
    def process_treasury_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process treasury yield data from CSV.
        
        Args:
            df: Raw treasury DataFrame
            
        Returns:
            Processed treasury DataFrame
        """
        logger.info("Processing treasury data...")
        
        # Ensure date column is datetime
        date_col = self._find_date_column(df.columns)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
        
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if '2y' in col_lower or '2_y' in col_lower:
                column_mapping[col] = '2Y_Yield'
            elif '5y' in col_lower or '5_y' in col_lower:
                column_mapping[col] = '5Y_Yield'
            elif '10y' in col_lower or '10_y' in col_lower:
                column_mapping[col] = '10Y_Yield'
            elif '30y' in col_lower or '30_y' in col_lower:
                column_mapping[col] = '30Y_Yield'
            elif 'sofr' in col_lower:
                column_mapping[col] = 'SOFR_Yield'
            elif 'date' in col_lower or 'time' in col_lower:
                column_mapping[col] = 'Date'
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Calculate additional metrics if not present
        yield_cols = [col for col in df.columns if 'Yield' in col]
        
        for col in yield_cols:
            # Convert yield to price if not present
            price_col = col.replace('Yield', 'Price')
            if price_col not in df.columns:
                df[price_col] = 100 / (1 + df[col]/100)
            
            # Calculate returns if not present
            returns_col = col.replace('Yield', 'Returns')
            if returns_col not in df.columns:
                df[returns_col] = df[col].pct_change()
            
            # Calculate volatility if not present
            vol_col = col.replace('Yield', 'Volatility')
            if vol_col not in df.columns:
                df[vol_col] = df[returns_col].rolling(window=20).std()
        
        logger.info(f"Processed treasury data: {df.shape}")
        return df
    
    def process_order_book_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process order book data from CSV.
        
        Args:
            df: Raw order book DataFrame
        
        Returns:
            Processed order book DataFrame
        """
        logger.info("Processing order book data...")
        
        # Ensure timestamp column is datetime
        timestamp_col = self._find_timestamp_column(df.columns)
        if timestamp_col:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'bid' in col_lower and 'price' in col_lower:
                column_mapping[col] = 'bid_price'
            elif 'bid' in col_lower and ('size' in col_lower or 'volume' in col_lower):
                column_mapping[col] = 'bid_size'
            elif 'ask' in col_lower and 'price' in col_lower:
                column_mapping[col] = 'ask_price'
            elif 'ask' in col_lower and ('size' in col_lower or 'volume' in col_lower):
                column_mapping[col] = 'ask_size'
            elif 'level' in col_lower:
                column_mapping[col] = 'level'
            elif 'instrument' in col_lower or 'symbol' in col_lower:
                column_mapping[col] = 'instrument'
            elif 'time' in col_lower or 'date' in col_lower:
                column_mapping[col] = 'timestamp'
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Calculate spread if not present
        if 'bid_price' in df.columns and 'ask_price' in df.columns:
            df['spread'] = df['ask_price'] - df['bid_price']
            df['spread_bps'] = (df['spread'] / df['bid_price']) * 10000
        
        logger.info(f"Processed order book data: {df.shape}")
        return df
    
    def _find_date_column(self, columns: List[str]) -> Optional[str]:
        """Find the date column in the DataFrame."""
        date_patterns = ['date', 'time', 'timestamp', 'datetime']
        for col in columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                return col
        return None
    
    def _find_timestamp_column(self, columns: List[str]) -> Optional[str]:
        """Find the timestamp column in the DataFrame."""
        timestamp_patterns = ['timestamp', 'time', 'datetime', 'date']
        for col in columns:
            if any(pattern in col.lower() for pattern in timestamp_patterns):
                return col
        return None
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for GAN training.
        
        Args:
            df: Processed DataFrame
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (sequences, targets)
        """
        logger.info(f"Creating sequences with length {sequence_length}")
        
        # Remove date/timestamp columns for numerical processing
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_data = df[numeric_cols].values
        
        # Handle missing values
        numeric_data = np.nan_to_num(numeric_data, nan=0.0)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(numeric_data) - sequence_length):
            sequence = numeric_data[i:i + sequence_length]
            target = numeric_data[i + sequence_length]
            
            sequences.append(sequence)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"Created sequences: {sequences.shape}, targets: {targets.shape}")
        return sequences, targets
    
    def save_processed_data(self, data: Dict[str, pd.DataFrame], output_dir: str = "data"):
        """
        Save processed data to files.
        
        Args:
            data: Dictionary of processed DataFrames
            output_dir: Output directory for processed data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for data_type, df in data.items():
            output_path = os.path.join(output_dir, f"{data_type}_processed.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {data_type} data to {output_path}")
    
    def collect_and_process(self, sequence_length: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main method to collect and process CSV data.
        
        Args:
            sequence_length: Length of sequences for GAN training
            
        Returns:
            Tuple of (sequences, targets) for GAN training
        """
        logger.info("Starting CSV data collection and processing...")
        
        # Load CSV files
        raw_data = self.load_csv_data()
        
        if not raw_data:
            raise ValueError("No CSV data files found or loaded")
        
        # Process each data type
        processed_data = {}
        
        if 'treasury' in raw_data:
            processed_data['treasury'] = self.process_treasury_data(raw_data['treasury'])
        
        if 'order_book' in raw_data:
            processed_data['order_book'] = self.process_order_book_data(raw_data['order_book'])
        
        if 'features' in raw_data:
            processed_data['features'] = raw_data['features']  # Keep as-is for now
        
        # Combine all data
        combined_df = self._combine_data(processed_data)
        
        # Create sequences
        sequences, targets = self.create_sequences(combined_df, sequence_length)
        
        # Save processed data
        self.save_processed_data(processed_data)
        
        logger.info("CSV data collection and processing completed successfully!")
        return sequences, targets
    
    def _combine_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine different data types into a single DataFrame.
        
        Args:
            data_dict: Dictionary of processed DataFrames
            
        Returns:
            Combined DataFrame
        """
        if not data_dict:
            return pd.DataFrame()
        
        # Start with treasury data if available
        if 'treasury' in data_dict:
            combined = data_dict['treasury'].copy()
        else:
            combined = pd.DataFrame()
        
        # Add other data types
        for data_type, df in data_dict.items():
            if data_type == 'treasury':
                continue
            
            # Merge on date/timestamp if possible
            if not combined.empty and 'Date' in combined.columns:
                date_col = self._find_date_column(df.columns)
                if date_col:
                    combined = combined.merge(df, left_on='Date', right_on=date_col, how='outer')
                else:
                    # Concatenate if no common date column
                    combined = pd.concat([combined, df], axis=1)
            else:
                combined = pd.concat([combined, df], axis=1)
        
        return combined

def main():
    """Main function for testing CSV data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect and process CSV data for GAN training')
    parser.add_argument('--data-dir', type=str, default='data/csv',
                       help='Directory containing CSV files')
    parser.add_argument('--sequence-length', type=int, default=100,
                       help='Length of sequences for GAN training')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # Create collector
    collector = CSVDataCollector(args.data_dir)
    
    try:
        # Collect and process data
        sequences, targets = collector.collect_and_process(args.sequence_length)
        
        # Save sequences and targets
        np.save(os.path.join(args.output_dir, 'sequences.npy'), sequences)
        np.save(os.path.join(args.output_dir, 'targets.npy'), targets)
        
        print(f"✅ Successfully processed CSV data!")
        print(f"📊 Sequences shape: {sequences.shape}")
        print(f"🎯 Targets shape: {targets.shape}")
        
    except Exception as e:
        print(f"❌ Error processing CSV data: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 