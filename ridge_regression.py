"""
Ridge Regression Module
Adaptive Micro-Grid Segmentation - CEP Solution

This module implements:
- Analytical Ridge Regression using closed-form solution
- No gradient descent (hardware constraint compliance)
- Automatic lambda selection through cross-validation
- Per-cluster model training
"""

import numpy as np
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class RidgeRegression:
    """
    Analytical Ridge Regression solver using closed-form normal equation.
    
    β = (X^T X + λI)^(-1) X^T y
    
    This implementation satisfies embedded hardware constraints:
    - No iterative optimization (no gradient descent)
    - Single matrix operation for prediction
    - Guaranteed invertibility via L2 regularization
    """
    
    def __init__(self, lambda_param: float = 1.0):
        """
        Initialize Ridge Regression model.
        
        Args:
            lambda_param: L2 regularization parameter (λ > 0)
        """
        if lambda_param <= 0:
            raise ValueError("Lambda must be positive (λ > 0)")
        
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
        self.n_features = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, add_bias: bool = True):
        """
        Train Ridge Regression model using closed-form solution.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            add_bias: Whether to add bias term (intercept)
        """
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Add bias column if requested
        if add_bias:
            X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
        else:
            X_with_bias = X
        
        # Closed-form solution: β = (X^T X + λI)^(-1) X^T y
        XtX = X_with_bias.T @ X_with_bias
        
        # Add L2 regularization: λI
        # Note: We don't regularize the bias term (first column)
        lambda_matrix = self.lambda_param * np.eye(X_with_bias.shape[1])
        if add_bias:
            lambda_matrix[0, 0] = 0  # Don't regularize bias
        
        # Compute (X^T X + λI)
        regularized_XtX = XtX + lambda_matrix
        
        # Check if matrix is invertible (it should always be with λ > 0)
        det = np.linalg.det(regularized_XtX)
        if abs(det) < 1e-10:
            warnings.warn(f"Matrix nearly singular (det={det:.2e}). Increasing lambda.")
            lambda_matrix = (self.lambda_param * 10) * np.eye(X_with_bias.shape[1])
            if add_bias:
                lambda_matrix[0, 0] = 0
            regularized_XtX = XtX + lambda_matrix
        
        # Solve: β = (X^T X + λI)^(-1) X^T y
        try:
            beta = np.linalg.solve(regularized_XtX, X_with_bias.T @ y)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            warnings.warn("Using pseudo-inverse for singular matrix")
            beta = np.linalg.lstsq(regularized_XtX, X_with_bias.T @ y, rcond=None)[0]
        
        # Separate bias and weights
        if add_bias:
            self.bias = beta[0]
            self.weights = beta[1:]
        else:
            self.bias = 0
            self.weights = beta
        
        # Verify positive definiteness
        eigenvalues = np.linalg.eigvalsh(regularized_XtX)
        min_eigenvalue = np.min(eigenvalues)
        
        if min_eigenvalue > 0:
            status = "✓ Matrix is positive definite"
        else:
            status = "✗ Warning: Matrix may not be positive definite"
        
        print(f"  {status} (min eigenvalue: {min_eigenvalue:.6f})")
        print(f"  Trained with λ={self.lambda_param}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using trained model.
        
        Prediction: ŷ = X @ β + b
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
        
        # Simple matrix multiplication (hardware-friendly)
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Root Mean Squared Error.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            RMSE
        """
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return np.sqrt(mse)


class ClusterRidgeRegression:
    """
    Manages multiple Ridge Regression models, one per cluster.
    """
    
    def __init__(self, n_clusters: int, lambda_param: float = 1.0):
        """
        Initialize cluster-wise Ridge models.
        
        Args:
            n_clusters: Number of clusters (K)
            lambda_param: L2 regularization parameter
        """
        self.n_clusters = n_clusters
        self.lambda_param = lambda_param
        self.models = [RidgeRegression(lambda_param) for _ in range(n_clusters)]
    
    def fit(self, X: np.ndarray, y: np.ndarray, cluster_labels: np.ndarray):
        """
        Train separate Ridge model for each cluster.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            cluster_labels: Cluster assignments (n_samples,)
        """
        print(f"\n=== Training {self.n_clusters} Ridge Regression Models ===")
        
        for k in range(self.n_clusters):
            # Get data for this cluster
            mask = cluster_labels == k
            X_k = X[mask]
            y_k = y[mask]
            
            n_samples_k = X_k.shape[0]
            
            print(f"\nCluster {k}: {n_samples_k} samples")
            
            if n_samples_k == 0:
                warnings.warn(f"Cluster {k} has no samples!")
                continue
            
            # Check for potential singularity
            if n_samples_k < X.shape[1]:
                print(f"  ⚠ Warning: n_k ({n_samples_k}) < d ({X.shape[1]})")
                print(f"  Ridge regularization prevents singularity!")
            
            # Train model for this cluster
            self.models[k].fit(X_k, y_k)
        
        print("\n=== All cluster models trained ===")
    
    def predict_cluster(self, X: np.ndarray, cluster_id: int) -> np.ndarray:
        """
        Predict using a specific cluster's model.
        
        Args:
            X: Feature matrix
            cluster_id: Cluster ID
            
        Returns:
            Predictions
        """
        return self.models[cluster_id].predict(X)
    
    def evaluate_clusters(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        cluster_labels: np.ndarray
    ) -> dict:
        """
        Evaluate each cluster's model separately.
        
        Args:
            X: Feature matrix
            y: Target vector
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary with per-cluster metrics
        """
        metrics = {
            'cluster_rmse': [],
            'cluster_r2': [],
            'cluster_sizes': []
        }
        
        print(f"\n=== Per-Cluster Evaluation ===")
        
        for k in range(self.n_clusters):
            mask = cluster_labels == k
            X_k = X[mask]
            y_k = y[mask]
            
            if len(y_k) == 0:
                metrics['cluster_rmse'].append(np.nan)
                metrics['cluster_r2'].append(np.nan)
                metrics['cluster_sizes'].append(0)
                continue
            
            rmse = self.models[k].rmse(X_k, y_k)
            r2 = self.models[k].score(X_k, y_k)
            
            metrics['cluster_rmse'].append(rmse)
            metrics['cluster_r2'].append(r2)
            metrics['cluster_sizes'].append(len(y_k))
            
            print(f"Cluster {k}: RMSE={rmse:.2f}, R²={r2:.4f}, n={len(y_k)}")
        
        return metrics


def select_lambda_cv(
    X: np.ndarray, 
    y: np.ndarray, 
    lambda_range: np.ndarray = np.logspace(-3, 3, 20),
    k_folds: int = 5
) -> float:
    """
    Select optimal lambda using K-fold cross-validation.
    
    Args:
        X: Feature matrix
        y: Target vector
        lambda_range: Range of lambda values to test
        k_folds: Number of CV folds
        
    Returns:
        Optimal lambda value
    """
    n_samples = X.shape[0]
    fold_size = n_samples // k_folds
    
    best_lambda = lambda_range[0]
    best_rmse = float('inf')
    
    print("\n=== Lambda Selection via Cross-Validation ===")
    
    for lam in lambda_range:
        rmse_folds = []
        
        for fold in range(k_folds):
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size
            
            val_idx = list(range(val_start, val_end))
            train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train and evaluate
            model = RidgeRegression(lambda_param=lam)
            model.fit(X_train, y_train, add_bias=True)
            rmse = model.rmse(X_val, y_val)
            rmse_folds.append(rmse)
        
        avg_rmse = np.mean(rmse_folds)
        
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_lambda = lam
    
    print(f"Optimal λ = {best_lambda:.6f} (CV RMSE = {best_rmse:.4f})")
    
    return best_lambda


if __name__ == "__main__":
    """
    Demo usage of Ridge Regression
    """
    # Generate sample data with singularity issue
    np.random.seed(42)
    
    n_samples = 50
    n_features = 10
    
    # Create scenario where n < d (singularity guaranteed for OLS)
    X = np.random.randn(n_samples, n_features)
    true_beta = np.random.randn(n_features)
    y = X @ true_beta + np.random.randn(n_samples) * 0.5
    
    print(f"Dataset: n={n_samples}, d={n_features}")
    print(f"Condition: n < d → X^T X is singular for OLS")
    
    # Find optimal lambda
    optimal_lambda = select_lambda_cv(X, y)
    
    # Train Ridge model
    print("\n=== Training Ridge Regression ===")
    model = RidgeRegression(lambda_param=optimal_lambda)
    model.fit(X, y)
    
    # Evaluate
    rmse = model.rmse(X, y)
    r2 = model.score(X, y)
    
    print(f"\nFinal Model Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
