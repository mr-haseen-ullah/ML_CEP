"""
Evaluation Module
Adaptive Micro-Grid Segmentation - CEP Solution

This script evaluates and compares models:
- Global model vs Hybrid model
- RMSE comparison
- Cluster size analysis
- Failure case identification
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import pickle
import os


def evaluate_models(model_dir: str = 'models', output_dir: str = 'results'):
    """
    Comprehensive model evaluation and comparison.
    
    Args:
        model_dir: Directory containing trained models
        output_dir: Directory to save evaluation results
    """
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Load models
    print("\nLoading models...")
    from hybrid_predictor import HybridPredictor
    
    hybrid_predictor = HybridPredictor.load(f'{model_dir}/hybrid_predictor.pkl')
    
    with open(f'{model_dir}/global_predictor.pkl', 'rb') as f:
        global_predictor = pickle.load(f)
    
    with open(f'{model_dir}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print("✓ Models loaded")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test data (reload from training)
    print("\nLoading test data...")
    from data_loader import DataLoader, create_sample_data
    from sklearn.preprocessing import StandardScaler
    
    try:
        loader = DataLoader('energydata_complete.csv')
        data = loader.get_full_pipeline(test_size=0.2)
        X_test = data['X_test']
        y_test = data['y_test']
    except:
        # Use synthetic data
        X, y = create_sample_data(n_samples=2000)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test = X[1600:]
        y_test = y[1600:]
    
    print(f"Test samples: {X_test.shape[0]}")
    
    # Evaluate both models
    from hybrid_predictor import compare_models
    comparison = compare_models(hybrid_predictor, global_predictor, X_test, y_test)
    
    # ==================== VISUALIZATIONS ====================
    
    # 1. RMSE Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    models = ['Global Model', 'Hybrid Model']
    rmse_values = [
        comparison['global']['overall']['rmse'],
        comparison['hybrid']['overall']['rmse']
    ]
    
    colors = ['#e74c3c', '#27ae60']
    bars = plt.bar(models, rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, value in zip(bars, rmse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    plt.ylabel('RMSE (Wh)', fontsize=12)
    plt.title('Model Comparison: Global vs Hybrid', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rmse_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir}/rmse_comparison.png")
    plt.close()
    
    # 2. Per-Cluster RMSE
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cluster_metrics = comparison['hybrid']['per_cluster']
    cluster_ids = [m['cluster_id'] for m in cluster_metrics]
    cluster_rmses = [m['rmse'] for m in cluster_metrics]
    cluster_sizes = [m['n_samples'] for m in cluster_metrics]
    
    bars = ax.bar(cluster_ids, cluster_rmses, color='#3498db', alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    
    # Add size labels
    for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'n={size}', ha='center', fontsize=10)
    
    # Add global RMSE line
    ax.axhline(comparison['global']['overall']['rmse'], color='red', 
               linestyle='--', linewidth=2, label='Global Model RMSE')
    
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('RMSE (Wh)', fontsize=12)
    ax.set_title('Per-Cluster RMSE Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_cluster_rmse.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/per_cluster_rmse.png")
    plt.close()
    
    # 3. Cluster Size Distribution
    plt.figure(figsize=(10, 6))
    
    plt.pie(cluster_sizes, labels=[f'Cluster {i}' for i in cluster_ids],
            autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3'))
    plt.title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    plt.savefig(f'{output_dir}/cluster_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/cluster_distribution.png")
    plt.close()
    
    # 4. Residual Analysis
    y_pred_hybrid = hybrid_predictor.predict(X_test)
    y_pred_global = global_predictor.predict(X_test)
    
    residuals_hybrid = y_test - y_pred_hybrid
    residuals_global = y_test - y_pred_global
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Global residuals
    axes[0].scatter(y_pred_global, residuals_global, alpha=0.5, s=20)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Energy (Wh)', fontsize=11)
    axes[0].set_ylabel('Residuals (Wh)', fontsize=11)
    axes[0].set_title('Global Model Residuals', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Hybrid residuals
    axes[1].scatter(y_pred_hybrid, residuals_hybrid, alpha=0.5, s=20, color='green')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Energy (Wh)', fontsize=11)
    axes[1].set_ylabel('Residuals (Wh)', fontsize=11)
    axes[1].set_title('Hybrid Model Residuals', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residual_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/residual_analysis.png")
    plt.close()
    
    # ==================== FAILURE CASE ANALYSIS ====================
    print("\n" + "="*70)
    print("FAILURE CASE ANALYSIS")
    print("="*70)
    
    # Identify small clusters
    small_cluster_threshold = 50
    small_clusters = [(m['cluster_id'], m['n_samples']) 
                      for m in cluster_metrics if m['n_samples'] < small_cluster_threshold]
    
    if small_clusters:
        print(f"\n⚠ Small clusters detected (n < {small_cluster_threshold}):")
        for cluster_id, size in small_clusters:
            rmse = cluster_metrics[cluster_id]['rmse']
            print(f"  Cluster {cluster_id}: {size} samples, RMSE={rmse:.2f}")
            print(f"    → Ridge regularization prevents singularity!")
    else:
        print(f"\n✓ No small clusters (all clusters have n ≥ {small_cluster_threshold})")
    
    # Identify worst-performing clusters
    worst_cluster = max(cluster_metrics, key=lambda x: x['rmse'])
    print(f"\nWorst-performing cluster:")
    print(f"  Cluster {worst_cluster['cluster_id']}")
    print(f"  RMSE: {worst_cluster['rmse']:.2f}")
    print(f"  Samples: {worst_cluster['n_samples']}")
    
    # Save evaluation summary
    summary = {
        'global_rmse': comparison['global']['overall']['rmse'],
        'hybrid_rmse': comparison['hybrid']['overall']['rmse'],
        'improvement_percent': comparison['improvement']['rmse_percent'],
        'n_clusters': metadata['optimal_k'],
        'lambda': metadata['lambda_param'],
        'small_clusters': small_clusters,
        'worst_cluster': worst_cluster
    }
    
    with open(f'{output_dir}/evaluation_summary.pkl', 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\n✓ Evaluation summary saved to {output_dir}/evaluation_summary.pkl")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    return summary


if __name__ == "__main__":
    """
    Run evaluation
    """
    summary = evaluate_models(model_dir='models', output_dir='results')
    
    print("\nKey Findings:")
    print(f"  Global RMSE: {summary['global_rmse']:.4f}")
    print(f"  Hybrid RMSE: {summary['hybrid_rmse']:.4f}")
    print(f"  Improvement: {summary['improvement_percent']:+.2f}%")
