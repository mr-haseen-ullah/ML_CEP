"""
Prediction Script
Adaptive Micro-Grid Segmentation - CEP Solution

This script loads trained models and makes predictions on new data.
"""

import numpy as np
import pickle
import argparse
import json
from typing import Dict, List
from hybrid_predictor import HybridPredictor


def load_models(model_dir: str = 'models') -> Dict:
    """
    Load trained models and metadata.
    
    Args:
        model_dir: Directory containing saved models
        
    Returns:
        Dictionary with models and metadata
    """
    print(f"Loading models from {model_dir}...")
    
    # Load hybrid predictor
    hybrid_predictor = HybridPredictor.load(f'{model_dir}/hybrid_predictor.pkl')
    
    # Load metadata
    with open(f'{model_dir}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"✓ Models loaded successfully!")
    print(f"  Clusters: {metadata['optimal_k']}")
    print(f"  Features: {len(metadata['feature_names'])}")
    
    return {
        'hybrid_predictor': hybrid_predictor,
        'metadata': metadata
    }


def predict_from_array(
    features: np.ndarray,
    model_dir: str = 'models',
    explain: bool = False
) -> Dict:
    """
    Make prediction from numpy array.
    
    Args:
        features: Feature vector or matrix
        model_dir: Directory containing models
        explain: Whether to provide explanation
        
    Returns:
        Prediction results
    """
    models = load_models(model_dir)
    hybrid_predictor = models['hybrid_predictor']
    
    # Handle single vector
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Make predictions
    predictions, clusters = hybrid_predictor.predict(features, return_clusters=True)
    
    results = {
        'predictions': predictions.tolist(),
        'cluster_assignments': clusters.tolist()
    }
    
    # Add explanation if requested
    if explain and features.shape[0] == 1:
        explanation = hybrid_predictor.explain_prediction(features[0])
        results['explanation'] = {
            'bias': float(explanation['bias']),
            'top_contributing_features': explanation['top_features'].tolist()
        }
    
    return results


def predict_from_json(
    json_path: str,
    model_dir: str = 'models'
) -> Dict:
    """
    Make prediction from JSON file.
    
    Expected JSON format:
    {
        "features": [value1, value2, ..., valueN]
    }
    or
    {
        "features": [[row1], [row2], ...]
    }
    
    Args:
        json_path: Path to JSON file
        model_dir: Directory containing models
        
    Returns:
        Prediction results
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    features = np.array(data['features'])
    
    return predict_from_array(features, model_dir, explain=True)


def interactive_prediction(model_dir: str = 'models'):
    """
    Interactive prediction mode.
    
    Args:
        model_dir: Directory containing models
    """
    print("\n" + "="*70)
    print("INTERACTIVE PREDICTION MODE")
    print("="*70)
    
    models = load_models(model_dir)
    hybrid_predictor = models['hybrid_predictor']
    metadata = models['metadata']
    feature_names = metadata['feature_names']
    
    print(f"\nPlease provide {len(feature_names)} feature values:")
    
    # Get user input
    features = []
    for i, name in enumerate(feature_names):
        while True:
            try:
                value = input(f"  {i+1}. {name}: ")
                features.append(float(value))
                break
            except ValueError:
                print("     Invalid input. Please enter a number.")
    
    features = np.array(features)
    
    # Make prediction
    prediction, cluster = hybrid_predictor.predict_single(features, return_cluster=True)
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"\nPredicted Energy Consumption: {prediction:.2f} Wh")
    print(f"Cluster Assignment: Cluster {cluster}")
    print(f"  (Cluster {cluster} has {metadata['cluster_info']['cluster_sizes'][cluster]} training samples)")
    
    # Get explanation
    explanation = hybrid_predictor.explain_prediction(features)
    
    print(f"\nModel Details:")
    print(f"  Bias term: {explanation['bias']:.4f}")
    print(f"\n  Top 5 Contributing Features:")
    for rank, idx in enumerate(explanation['top_features'][:5], 1):
        contribution = explanation['feature_contributions'][idx]
        print(f"    {rank}. {feature_names[idx]}: {contribution:+.4f}")


def batch_prediction(
    input_file: str,
    output_file: str,
    model_dir: str = 'models'
):
    """
    Batch prediction from CSV file.
    
    Args:
        input_file: CSV file with features (no header)
        output_file: Output CSV file
        model_dir: Directory containing models
    """
    print(f"Loading data from {input_file}...")
    
    # Load data
    data = np.loadtxt(input_file, delimiter=',')
    
    print(f"Loaded {data.shape[0]} samples with {data.shape[1]} features")
    
    # Make predictions
    results = predict_from_array(data, model_dir)
    
    # Save results
    output_data = np.column_stack([
        data,
        results['predictions'],
        results['cluster_assignments']
    ])
    
    np.savetxt(output_file, output_data, delimiter=',', 
               header='features...,prediction,cluster', comments='')
    
    print(f"✓ Predictions saved to {output_file}")


def main():
    """
    Main function with CLI interface.
    """
    parser = argparse.ArgumentParser(
        description='Adaptive Micro-Grid Segmentation - Prediction'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'json', 'batch'],
        default='interactive',
        help='Prediction mode'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input file path (for json or batch mode)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (for batch mode)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained models'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        interactive_prediction(args.model_dir)
    
    elif args.mode == 'json':
        if not args.input:
            print("Error: --input required for json mode")
            return
        
        results = predict_from_json(args.input, args.model_dir)
        print("\nPrediction Results:")
        print(json.dumps(results, indent=2))
    
    elif args.mode == 'batch':
        if not args.input or not args.output:
            print("Error: --input and --output required for batch mode")
            return
        
        batch_prediction(args.input, args.output, args.model_dir)


if __name__ == "__main__":
    # If run without arguments, show usage example
    import sys
    
    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print("PREDICTION SCRIPT - USAGE EXAMPLES")
        print("="*70)
        print("\n1. Interactive mode (default):")
        print("   python predict.py")
        print("\n2. Predict from JSON file:")
        print("   python predict.py --mode json --input data.json")
        print("\n3. Batch prediction from CSV:")
        print("   python predict.py --mode batch --input data.csv --output results.csv")
        print("\n" + "="*70)
        
        # Run interactive mode by default
        interactive_prediction()
    else:
        main()
