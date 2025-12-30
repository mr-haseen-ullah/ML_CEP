"""
Hybrid Predictor Module
Adaptive Micro-Grid Segmentation - CEP Solution

This module implements the HybridPredictor class that:
1. Takes an input vector x
2. Determines its cluster
3. Applies the appropriate Ridge regression model
4. Returns predicted energy usage
"""

import numpy as np
import pickle
from typing import Optional, Tuple
from clustering import ClusteringEngine
from ridge_regression import ClusterRidgeRegression


class HybridPredictor:
    """
    Hybrid prediction system combining clustering and regression.
    
    Architecture:
        Input → Cluster Assignment → Cluster-Specific Ridge Model → Prediction
    """
    
    def __init__(
        self,
        clustering_model: ClusteringEngine,
        regression_models: ClusterRidgeRegression
    ):
        """
        Initialize HybridPredictor with trained models.
        
        Args:
            clustering_model: Trained ClusteringEngine
            regression_models: Trained ClusterRidgeRegression
        """
        self.clustering_model = clustering_model
        self.regression_models = regression_models
        self.n_clusters = regression_models.n_clusters
        
        # Validate models are trained
        if self.clustering_model.model is None:
            raise ValueError("Clustering model not trained")
        
        for k, model in enumerate(self.regression_models.models):
            if model.weights is None:
                raise ValueError(f"Ridge model for cluster {k} not trained")
        
        print(f"HybridPredictor initialized with {self.n_clusters} clusters")
    
    def predict_single(self, x: np.ndarray, return_cluster: bool = False):
        """
        Predict energy consumption for a single input.
        
        Args:
            x: Feature vector (n_features,)
            return_cluster: Whether to return cluster assignment
            
        Returns:
            Prediction (and cluster ID if return_cluster=True)
        """
        # Ensure x is 2D for sklearn compatibility
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Step 1: Determine cluster
        cluster_id = self.clustering_model.predict(x)[0]
        
        # Step 2: Apply cluster-specific regression model
        prediction = self.regression_models.predict_cluster(x, cluster_id)[0]
        
        if return_cluster:
            return prediction, cluster_id
        return prediction
    
    def predict(self, X: np.ndarray, return_clusters: bool = False):
        """
        Predict energy consumption for multiple inputs.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            return_clusters: Whether to return cluster assignments
            
        Returns:
            Predictions (and cluster IDs if return_clusters=True)
        """
        # Step 1: Determine clusters for all inputs
        cluster_labels = self.clustering_model.predict(X)
        
        # Step 2: Apply appropriate regression model for each sample
        predictions = np.zeros(X.shape[0])
        
        for k in range(self.n_clusters):
            # Get samples belonging to cluster k
            mask = cluster_labels == k
            
            if np.sum(mask) > 0:
                # Predict using cluster k's model
                predictions[mask] = self.regression_models.predict_cluster(
                    X[mask], k
                )
        
        if return_clusters:
            return predictions, cluster_labels
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate hybrid model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions, cluster_labels = self.predict(X, return_clusters=True)
        
        # Overall metrics
        rmse = np.sqrt(np.mean((y - predictions) ** 2))
        mae = np.mean(np.abs(y - predictions))
        r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        # Per-cluster metrics
        cluster_metrics = []
        for k in range(self.n_clusters):
            mask = cluster_labels == k
            if np.sum(mask) > 0:
                y_k = y[mask]
                pred_k = predictions[mask]
                
                rmse_k = np.sqrt(np.mean((y_k - pred_k) ** 2))
                mae_k = np.mean(np.abs(y_k - pred_k))
                
                cluster_metrics.append({
                    'cluster_id': k,
                    'n_samples': np.sum(mask),
                    'rmse': rmse_k,
                    'mae': mae_k
                })
        
        return {
            'overall': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'per_cluster': cluster_metrics
        }
    
    def explain_prediction(self, x: np.ndarray) -> dict:
        """
        Explain prediction for a single input.
        
        Args:
            x: Feature vector
            
        Returns:
            Dictionary with prediction explanation
        """
        prediction, cluster_id = self.predict_single(x, return_cluster=True)
        
        # Get model weights
        model = self.regression_models.models[cluster_id]
        
        # Feature contributions
        contributions = x * model.weights
        
        explanation = {
            'prediction': prediction,
            'cluster_id': cluster_id,
            'bias': model.bias,
            'feature_contributions': contributions,
            'top_features': np.argsort(np.abs(contributions))[::-1][:5]  # Top 5
        }
        
        return explanation
    
    def save(self, filepath: str):
        """
        Save HybridPredictor to file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'clustering_model': self.clustering_model,
            'regression_models': self.regression_models,
            'n_clusters': self.n_clusters
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"HybridPredictor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'HybridPredictor':
        """
        Load HybridPredictor from file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded HybridPredictor
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = HybridPredictor(
            clustering_model=model_data['clustering_model'],
            regression_models=model_data['regression_models']
        )
        
        print(f"HybridPredictor loaded from {filepath}")
        
        return predictor


class GlobalPredictor:
    """
    Baseline: Single global Ridge regression model (for comparison).
    """
    
    def __init__(self, lambda_param: float = 1.0):
        """
        Initialize global predictor.
        
        Args:
            lambda_param: L2 regularization parameter
        """
        from ridge_regression import RidgeRegression
        self.model = RidgeRegression(lambda_param=lambda_param)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train global model.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        print("\n=== Training Global Ridge Model ===")
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using global model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate global model.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with metrics
        """
        predictions = self.predict(X)
        
        rmse = np.sqrt(np.mean((y - predictions) ** 2))
        mae = np.mean(np.abs(y - predictions))
        r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        return {
            'overall': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        }


def compare_models(
    hybrid_predictor: HybridPredictor,
    global_predictor: GlobalPredictor,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Compare Hybrid vs Global model performance.
    
    Args:
        hybrid_predictor: Trained HybridPredictor
        global_predictor: Trained GlobalPredictor
        X_test: Test feature matrix
        y_test: Test target values
        
    Returns:
        Comparison dictionary
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON: HYBRID vs GLOBAL")
    print("="*60)
    
    # Evaluate both models
    hybrid_metrics = hybrid_predictor.evaluate(X_test, y_test)
    global_metrics = global_predictor.evaluate(X_test, y_test)
    
    # Calculate improvements
    rmse_improvement = (
        (global_metrics['overall']['rmse'] - hybrid_metrics['overall']['rmse']) 
        / global_metrics['overall']['rmse'] * 100
    )
    
    mae_improvement = (
        (global_metrics['overall']['mae'] - hybrid_metrics['overall']['mae']) 
        / global_metrics['overall']['mae'] * 100
    )
    
    # Print comparison
    print("\nGLOBAL MODEL:")
    print(f"  RMSE: {global_metrics['overall']['rmse']:.4f}")
    print(f"  MAE:  {global_metrics['overall']['mae']:.4f}")
    print(f"  R²:   {global_metrics['overall']['r2']:.4f}")
    
    print("\nHYBRID MODEL:")
    print(f"  RMSE: {hybrid_metrics['overall']['rmse']:.4f}")
    print(f"  MAE:  {hybrid_metrics['overall']['mae']:.4f}")
    print(f"  R²:   {hybrid_metrics['overall']['r2']:.4f}")
    
    print("\nIMPROVEMENT:")
    print(f"  RMSE: {rmse_improvement:+.2f}%")
    print(f"  MAE:  {mae_improvement:+.2f}%")
    
    if rmse_improvement > 0:
        print("\n✓ Hybrid model outperforms global model!")
    else:
        print("\n✗ Global model performs better (unexpected)")
    
    print("\nPER-CLUSTER PERFORMANCE:")
    for cluster_metric in hybrid_metrics['per_cluster']:
        k = cluster_metric['cluster_id']
        print(f"  Cluster {k}: RMSE={cluster_metric['rmse']:.4f}, n={cluster_metric['n_samples']}")
    
    print("="*60)
    
    return {
        'hybrid': hybrid_metrics,
        'global': global_metrics,
        'improvement': {
            'rmse_percent': rmse_improvement,
            'mae_percent': mae_improvement
        }
    }


if __name__ == "__main__":
    """
    Demo usage of HybridPredictor
    """
    from data_loader import create_sample_data
    from sklearn.preprocessing import StandardScaler
    
    # Generate sample data
    X, y = create_sample_data(n_samples=1000)
    
    # Normalize
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Split
    split_idx = 800
    X_train, X_test = X_normalized[:split_idx], X_normalized[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train clustering
    clustering = ClusteringEngine(method='gmm', random_state=42)
    clustering.find_optimal_k(X_train, k_range=range(2, 6), criterion='bic')
    clustering.fit(X_train)
    
    # Get cluster labels
    train_labels = clustering.predict(X_train)
    
    # Train cluster-wise Ridge models
    from ridge_regression import ClusterRidgeRegression
    regression_models = ClusterRidgeRegression(
        n_clusters=clustering.optimal_k,
        lambda_param=1.0
    )
    regression_models.fit(X_train, y_train, train_labels)
    
    # Create HybridPredictor
    hybrid = HybridPredictor(clustering, regression_models)
    
    # Train GlobalPredictor for comparison
    global_model = GlobalPredictor(lambda_param=1.0)
    global_model.fit(X_train, y_train)
    
    # Compare
    comparison = compare_models(hybrid, global_model, X_test, y_test)
    
    # Test single prediction
    print("\n=== Single Prediction Example ===")
    x_sample = X_test[0]
    prediction, cluster = hybrid.predict_single(x_sample, return_cluster=True)
    print(f"Input: {x_sample[:3]}... (showing first 3 features)")
    print(f"Predicted energy: {prediction:.2f} Wh")
    print(f"Assigned to cluster: {cluster}")
    print(f"Actual energy: {y_test[0]:.2f} Wh")
