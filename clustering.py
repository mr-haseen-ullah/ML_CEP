"""
Clustering Module
Adaptive Micro-Grid Segmentation - CEP Solution

This module implements:
- Gaussian Mixture Models (GMM) clustering
- K-Means clustering (alternative)
- Elbow Method for optimal K selection
- BIC/AIC scoring for model selection
- Cluster visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


class ClusteringEngine:
    """
    Handles unsupervised learning for detecting campus operating modes.
    """
    
    def __init__(self, method: str = 'gmm', random_state: int = 42):
        """
        Initialize clustering engine.
        
        Args:
            method: 'gmm' for Gaussian Mixture Models or 'kmeans' for K-Means
            random_state: Random seed for reproducibility
        """
        self.method = method.lower()
        self.random_state = random_state
        self.model = None
        self.optimal_k = None
        self.scores = {}
        
        if self.method not in ['gmm', 'kmeans']:
            raise ValueError("Method must be 'gmm' or 'kmeans'")
    
    def find_optimal_k(
        self, 
        X: np.ndarray, 
        k_range: range = range(2, 11),
        criterion: str = 'bic'
    ) -> int:
        """
        Find optimal number of clusters using Elbow Method or BIC/AIC.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            k_range: Range of K values to test
            criterion: 'elbow', 'bic', or 'aic' (for GMM only)
            
        Returns:
            Optimal K value
        """
        print(f"\n=== Finding Optimal K using {criterion.upper()} ===")
        
        inertias = []
        bic_scores = []
        aic_scores = []
        
        for k in k_range:
            if self.method == 'gmm':
                model = GaussianMixture(
                    n_components=k, 
                    random_state=self.random_state,
                    covariance_type='full'
                )
                model.fit(X)
                
                bic_scores.append(model.bic(X))
                aic_scores.append(model.aic(X))
                
                # For GMM, use negative log-likelihood as "inertia"
                inertias.append(-model.score(X) * X.shape[0])
                
            else:  # kmeans
                model = KMeans(
                    n_clusters=k, 
                    random_state=self.random_state,
                    n_init=10
                )
                model.fit(X)
                inertias.append(model.inertia_)
        
        # Store scores
        self.scores = {
            'k_values': list(k_range),
            'inertias': inertias,
            'bic': bic_scores if self.method == 'gmm' else None,
            'aic': aic_scores if self.method == 'gmm' else None
        }
        
        # Determine optimal K
        if criterion == 'bic' and self.method == 'gmm':
            optimal_idx = np.argmin(bic_scores)
            self.optimal_k = list(k_range)[optimal_idx]
            print(f"Optimal K based on BIC: {self.optimal_k}")
            
        elif criterion == 'aic' and self.method == 'gmm':
            optimal_idx = np.argmin(aic_scores)
            self.optimal_k = list(k_range)[optimal_idx]
            print(f"Optimal K based on AIC: {self.optimal_k}")
            
        else:  # elbow method
            # Use elbow detection (largest second derivative)
            if len(inertias) >= 3:
                second_derivatives = np.diff(np.diff(inertias))
                optimal_idx = np.argmax(second_derivatives) + 1
                self.optimal_k = list(k_range)[optimal_idx]
                print(f"Optimal K based on Elbow Method: {self.optimal_k}")
            else:
                self.optimal_k = list(k_range)[len(k_range)//2]
                print(f"Using middle K value: {self.optimal_k}")
        
        return self.optimal_k
    
    def plot_elbow(self, save_path: Optional[str] = None):
        """
        Plot Elbow curve and BIC/AIC scores.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.scores:
            raise ValueError("Run find_optimal_k() first")
        
        k_values = self.scores['k_values']
        
        if self.method == 'gmm' and self.scores['bic'] is not None:
            # Plot BIC and AIC
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # BIC plot
            axes[0].plot(k_values, self.scores['bic'], 'bo-', linewidth=2, markersize=8)
            axes[0].axvline(self.optimal_k, color='r', linestyle='--', label=f'Optimal K={self.optimal_k}')
            axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
            axes[0].set_ylabel('BIC Score', fontsize=12)
            axes[0].set_title('BIC Score vs Number of Clusters', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # AIC plot
            axes[1].plot(k_values, self.scores['aic'], 'go-', linewidth=2, markersize=8)
            axes[1].axvline(self.optimal_k, color='r', linestyle='--', label=f'Optimal K={self.optimal_k}')
            axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
            axes[1].set_ylabel('AIC Score', fontsize=12)
            axes[1].set_title('AIC Score vs Number of Clusters', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
        else:
            # Plot inertia (Elbow curve)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(k_values, self.scores['inertias'], 'bo-', linewidth=2, markersize=8)
            ax.axvline(self.optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal K={self.optimal_k}')
            ax.set_xlabel('Number of Clusters (K)', fontsize=12)
            ax.set_ylabel('Inertia', fontsize=12)
            ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def fit(self, X: np.ndarray, n_clusters: Optional[int] = None):
        """
        Fit clustering model to data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            n_clusters: Number of clusters (if None, uses optimal_k)
        """
        if n_clusters is None:
            if self.optimal_k is None:
                raise ValueError("Specify n_clusters or run find_optimal_k() first")
            n_clusters = self.optimal_k
        
        print(f"\nTraining {self.method.upper()} with K={n_clusters} clusters...")
        
        if self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=n_clusters,
                random_state=self.random_state,
                covariance_type='full',
                max_iter=200
            )
        else:  # kmeans
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
        
        self.model.fit(X)
        print("Clustering model trained successfully!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Cluster labels (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.model.predict(X)
    
    def get_cluster_info(self, X: np.ndarray, labels: np.ndarray) -> dict:
        """
        Get information about each cluster.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary with cluster statistics
        """
        n_clusters = len(np.unique(labels))
        
        info = {
            'n_clusters': n_clusters,
            'cluster_sizes': [],
            'cluster_percentages': []
        }
        
        print(f"\n=== Cluster Information ===")
        for k in range(n_clusters):
            mask = labels == k
            size = np.sum(mask)
            percentage = (size / len(labels)) * 100
            
            info['cluster_sizes'].append(size)
            info['cluster_percentages'].append(percentage)
            
            print(f"Cluster {k}: {size} samples ({percentage:.1f}%)")
        
        return info
    
    def visualize_clusters(
        self, 
        X: np.ndarray, 
        labels: np.ndarray,
        title: str = "Cluster Visualization",
        save_path: Optional[str] = None
    ):
        """
        Visualize clusters using PCA projection to 2D.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        # Project to 2D using PCA
        pca = PCA(n_components=2, random_state=self.random_state)
        X_2d = pca.fit_transform(X)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot each cluster
        n_clusters = len(np.unique(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for k in range(n_clusters):
            mask = labels == k
            plt.scatter(
                X_2d[mask, 0], 
                X_2d[mask, 1],
                c=[colors[k]], 
                label=f'Cluster {k} (n={np.sum(mask)})',
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    """
    Demo usage of ClusteringEngine
    """
    # Generate sample data
    np.random.seed(42)
    
    # Create 3 clusters with different patterns (simulating day/night/weekend)
    n_samples_per_cluster = 300
    
    # Day cluster: high temperature, high load
    day = np.random.randn(n_samples_per_cluster, 10) + np.array([2, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Night cluster: low temperature, low load
    night = np.random.randn(n_samples_per_cluster, 10) + np.array([-2, -2, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Weekend cluster: moderate temperature, low load
    weekend = np.random.randn(n_samples_per_cluster, 10) + np.array([0, -1, 0, 0, 0, 0, 0, 0, 0, 0])
    
    X_sample = np.vstack([day, night, weekend])
    
    # Initialize clustering engine
    engine = ClusteringEngine(method='gmm', random_state=42)
    
    # Find optimal K
    optimal_k = engine.find_optimal_k(X_sample, k_range=range(2, 8), criterion='bic')
    
    # Plot elbow/BIC curves
    engine.plot_elbow()
    
    # Fit model
    engine.fit(X_sample)
    
    # Predict clusters
    labels = engine.predict(X_sample)
    
    # Get cluster info
    info = engine.get_cluster_info(X_sample, labels)
    
    # Visualize
    engine.visualize_clusters(X_sample, labels, title="Sample Cluster Visualization")
