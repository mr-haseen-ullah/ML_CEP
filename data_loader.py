"""
Data Loading and Preprocessing Module
Adaptive Micro-Grid Segmentation - CEP Solution

This module handles:
- Loading UCI Appliances Energy Prediction dataset
- Feature extraction and selection
- Data normalization
- Train/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import os


class DataLoader:
    """
    Handles loading and preprocessing of energy consumption data.
    """
    
    def __init__(self, filepath: str = 'energydata_complete.csv'):
        """
        Initialize DataLoader with dataset path.
        
        Args:
            filepath: Path to the UCI Appliances Energy dataset
        """
        self.filepath = filepath
        self.scaler = StandardScaler()
        self.feature_names = None
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Returns:
            DataFrame containing the complete dataset
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Dataset not found at {self.filepath}. "
                "Please download 'energydata_complete.csv' from UCI ML Repository."
            )
        
        self.data = pd.read_csv(self.filepath)
        print(f"Dataset loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        return self.data
    
    def explore_data(self) -> dict:
        """
        Perform basic exploratory data analysis.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self.data is None:
            self.load_data()
        
        # Only compute correlation on numeric columns to avoid ValueError with datetime
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        stats = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'target_stats': self.data['Appliances'].describe().to_dict(),
            'correlation_with_target': numeric_data.corr()['Appliances'].sort_values(ascending=False).to_dict()
        }
        
        print("\n=== Dataset Overview ===")
        print(f"Shape: {stats['shape']}")
        print(f"\nTarget (Appliances) Statistics:")
        print(f"  Mean: {stats['target_stats']['mean']:.2f} Wh")
        print(f"  Std: {stats['target_stats']['std']:.2f} Wh")
        print(f"  Min: {stats['target_stats']['min']:.2f} Wh")
        print(f"  Max: {stats['target_stats']['max']:.2f} Wh")
        
        return stats
    
    def prepare_features(self, exclude_cols: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable.
        
        Args:
            exclude_cols: Columns to exclude from features (e.g., 'date', 'lights')
            
        Returns:
            Tuple of (features_df, target_series)
        """
        if self.data is None:
            self.load_data()
        
        # Default columns to exclude
        if exclude_cols is None:
            exclude_cols = ['date', 'Appliances']
        else:
            exclude_cols = list(set(exclude_cols + ['Appliances']))
        
        # Separate features and target
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        X = self.data[feature_cols]
        y = self.data['Appliances']
        
        self.feature_names = feature_cols
        
        print(f"\nFeatures selected: {len(feature_cols)}")
        print(f"Feature columns: {feature_cols[:5]}... (showing first 5)")
        
        return X, y
    
    def normalize_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Normalize features using StandardScaler.
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler (True for training data, False for test data)
            
        Returns:
            Normalized feature array
        """
        if fit:
            X_normalized = self.scaler.fit_transform(X)
            print("Features normalized (fitted scaler)")
        else:
            X_normalized = self.scaler.transform(X)
            print("Features normalized (using existing scaler)")
        
        return X_normalized
    
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nData split:")
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def get_full_pipeline(
        self, 
        test_size: float = 0.2, 
        exclude_cols: Optional[list] = None,
        random_state: int = 42
    ) -> dict:
        """
        Execute complete data loading and preprocessing pipeline.
        
        Args:
            test_size: Proportion of data for testing
            exclude_cols: Columns to exclude from features
            random_state: Random seed
            
        Returns:
            Dictionary containing all processed data and metadata
        """
        print("=== Starting Full Data Pipeline ===\n")
        
        # Load data
        self.load_data()
        
        # Explore
        stats = self.explore_data()
        
        # Prepare features
        X, y = self.prepare_features(exclude_cols=exclude_cols)
        
        # Normalize
        X_normalized = self.normalize_features(X, fit=True)
        
        # Split
        X_train, X_test, y_train, y_test = self.split_data(
            X_normalized, y.values, test_size=test_size, random_state=random_state
        )
        
        print("\n=== Pipeline Complete ===\n")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'stats': stats
        }


def create_sample_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample synthetic data for testing (if real dataset is not available).
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (X, y)
    """
    np.random.seed(42)
    
    # Generate features: temperature, humidity, etc.
    X = np.random.randn(n_samples, 10)
    
    # Generate target with some patterns
    y = 50 + 20 * X[:, 0] + 10 * X[:, 1] + np.random.randn(n_samples) * 5
    y = np.maximum(y, 0)  # Energy consumption cannot be negative
    
    print(f"Generated synthetic data: {n_samples} samples, 10 features")
    
    return X, y


if __name__ == "__main__":
    """
    Demo usage of DataLoader
    """
    # Initialize loader
    loader = DataLoader('energydata_complete.csv')
    
    try:
        # Run full pipeline
        data = loader.get_full_pipeline(test_size=0.2)
        
        print("\n=== Data Ready for Training ===")
        print(f"X_train shape: {data['X_train'].shape}")
        print(f"y_train shape: {data['y_train'].shape}")
        print(f"Number of features: {len(data['feature_names'])}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nGenerating synthetic data for demonstration...")
        
        # Use synthetic data
        X, y = create_sample_data(1000)
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y, test_size=0.2, random_state=42
        )
        
        print(f"\nSynthetic data ready:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
