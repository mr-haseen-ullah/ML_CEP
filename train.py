"""
Training Pipeline
Adaptive Micro-Grid Segmentation - CEP Solution

This script orchestrates the complete training process:
1. Load and preprocess data
2. Find optimal K using Elbow/BIC
3. Train clustering model
4. Train Ridge models per cluster
5. Save trained models
6. Generate training metrics
"""

import numpy as np
import os
import sys
from data_loader import DataLoader, create_sample_data
from clustering import ClusteringEngine
from ridge_regression import ClusterRidgeRegression, select_lambda_cv
from hybrid_predictor import HybridPredictor, GlobalPredictor
import pickle


def train_hybrid_system(
    data_path: str = 'energydata_complete.csv',
    test_size: float = 0.2,
    clustering_method: str = 'gmm',
    k_range: range = range(2, 8),
    lambda_param: float = None,
    save_models: bool = True,
    output_dir: str = 'models'
):
    """
    Complete training pipeline for hybrid prediction system.
    
    Args:
        data_path: Path to dataset
        test_size: Proportion of data for testing
        clustering_method: 'gmm' or 'kmeans'
        k_range: Range of K values to test
        lambda_param: Ridge regularization parameter (None for auto-selection)
        save_models: Whether to save trained models
        output_dir: Directory to save models
        
    Returns:
        Dictionary containing trained models and metrics
    """
    print("="*70)
    print("ADAPTIVE MICRO-GRID SEGMENTATION - TRAINING PIPELINE")
    print("="*70)
    
    # ==================== STEP 1: DATA LOADING ====================
    print("\n[STEP 1/6] Loading and Preprocessing Data...")
    
    loader = DataLoader(data_path)
    
    try:
        data = loader.get_full_pipeline(test_size=test_size)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        feature_names = data['feature_names']
        
    except FileNotFoundError:
        print(f"\nDataset not found at {data_path}")
        print("Using synthetic data for demonstration...")
        
        X, y = create_sample_data(n_samples=2000)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    # ==================== STEP 2: OPTIMAL K SELECTION ====================
    print("\n[STEP 2/6] Finding Optimal Number of Clusters...")
    
    clustering = ClusteringEngine(method=clustering_method, random_state=42)
    
    criterion = 'bic' if clustering_method == 'gmm' else 'elbow'
    optimal_k = clustering.find_optimal_k(X_train, k_range=k_range, criterion=criterion)
    
    print(f"\n✓ Optimal K selected: {optimal_k}")
    
    # Save elbow plot
    if save_models:
        os.makedirs(output_dir, exist_ok=True)
        clustering.plot_elbow(save_path=f'{output_dir}/elbow_curve.png')
    
    # ==================== STEP 3: CLUSTERING ====================
    print("\n[STEP 3/6] Training Clustering Model...")
    
    clustering.fit(X_train, n_clusters=optimal_k)
    train_labels = clustering.predict(X_train)
    
    # Get cluster information
    cluster_info = clustering.get_cluster_info(X_train, train_labels)
    
    # Visualize clusters
    if save_models:
        clustering.visualize_clusters(
            X_train, 
            train_labels,
            title=f"{clustering_method.upper()} Clustering (K={optimal_k})",
            save_path=f'{output_dir}/clusters_visualization.png'
        )
    
    # ==================== STEP 4: LAMBDA SELECTION ====================
    print("\n[STEP 4/6] Selecting Ridge Regularization Parameter...")
    
    if lambda_param is None:
        # Auto-select lambda using cross-validation on a subset
        subset_size = min(500, X_train.shape[0])
        lambda_param = select_lambda_cv(
            X_train[:subset_size], 
            y_train[:subset_size],
            lambda_range=np.logspace(-2, 2, 15)
        )
    else:
        print(f"Using provided λ = {lambda_param}")
    
    # ==================== STEP 5: RIDGE REGRESSION ====================
    print("\n[STEP 5/6] Training Cluster-Wise Ridge Regression Models...")
    
    regression_models = ClusterRidgeRegression(
        n_clusters=optimal_k,
        lambda_param=lambda_param
    )
    
    regression_models.fit(X_train, y_train, train_labels)
    
    # Evaluate per-cluster performance
    train_metrics = regression_models.evaluate_clusters(X_train, y_train, train_labels)
    
    # ==================== STEP 6: HYBRID SYSTEM ====================
    print("\n[STEP 6/6] Creating Hybrid Prediction System...")
    
    hybrid_predictor = HybridPredictor(clustering, regression_models)
    
    # Train global baseline
    print("\n[BASELINE] Training Global Ridge Model...")
    global_predictor = GlobalPredictor(lambda_param=lambda_param)
    global_predictor.fit(X_train, y_train)
    
    # ==================== EVALUATION ====================
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    
    from hybrid_predictor import compare_models
    comparison = compare_models(hybrid_predictor, global_predictor, X_test, y_test)
    
    # ==================== SAVE MODELS ====================
    if save_models:
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save hybrid predictor
        hybrid_predictor.save(f'{output_dir}/hybrid_predictor.pkl')
        
        # Save global predictor
        with open(f'{output_dir}/global_predictor.pkl', 'wb') as f:
            pickle.dump(global_predictor, f)
        print(f"Global predictor saved to {output_dir}/global_predictor.pkl")
        
        # Save metadata
        metadata = {
            'optimal_k': optimal_k,
            'lambda_param': lambda_param,
            'feature_names': feature_names,
            'cluster_info': cluster_info,
            'comparison': comparison,
            'clustering_method': clustering_method
        }
        
        with open(f'{output_dir}/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to {output_dir}/metadata.pkl")
        
        print("\n✓ All models and results saved!")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Clustering: {clustering_method.upper()}")
    print(f"  Number of clusters: {optimal_k}")
    print(f"  Ridge λ: {lambda_param:.6f}")
    
    print(f"\nResults:")
    print(f"  Global RMSE: {comparison['global']['overall']['rmse']:.4f}")
    print(f"  Hybrid RMSE: {comparison['hybrid']['overall']['rmse']:.4f}")
    print(f"  Improvement: {comparison['improvement']['rmse_percent']:+.2f}%")
    
    return {
        'hybrid_predictor': hybrid_predictor,
        'global_predictor': global_predictor,
        'comparison': comparison,
        'metadata': {
            'optimal_k': optimal_k,
            'lambda_param': lambda_param,
            'feature_names': feature_names,
            'cluster_info': cluster_info
        },
        'test_data': {
            'X_test': X_test,
            'y_test': y_test
        }
    }


if __name__ == "__main__":
    """
    Run training pipeline
    """
    # Configuration
    config = {
        'data_path': 'energydata_complete.csv',
        'test_size': 0.2,
        'clustering_method': 'gmm',  # or 'kmeans'
        'k_range': range(2, 8),
        'lambda_param': None,  # Auto-select
        'save_models': True,
        'output_dir': 'models'
    }
    
    # Train
    results = train_hybrid_system(**config)
    
    print("\n" + "="*70)
    print("Ready for prediction! Use predict.py to make predictions.")
    print("="*70)
