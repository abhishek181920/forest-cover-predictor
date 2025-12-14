#!/usr/bin/env python3
"""
Command Line Interface for Forest Cover Type Prediction System
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forest_cover_predictor import ForestCoverPredictor
from config import FILE_PATHS

def main():
    parser = argparse.ArgumentParser(description='Forest Cover Type Prediction System')
    parser.add_argument('--data', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Make predictions on new data')
    parser.add_argument('--explore', action='store_true', help='Explore the dataset')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--test-file', type=str, help='Path to test data for predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ForestCoverPredictor()
    
    # Determine data file path
    data_file = args.data if args.data else FILE_PATHS.get('data_file', 'forest_cover_data.csv')
    
    if args.explore:
        print("=== Data Exploration ===")
        data = predictor.load_data(data_file)
        predictor.explore_data(data)
        return
    
    if args.train:
        print("=== Model Training ===")
        data = predictor.load_data(data_file)
        X, y = predictor.preprocess_data(data)
        X_train, X_test, y_train, y_test, y_pred = predictor.train_model(X, y)
        
        if args.visualize:
            predictor.visualize_results(y_test, y_pred)
        
        print("Model training completed!")
        return
    
    if args.predict:
        print("=== Making Predictions ===")
        # Load and train model first
        data = predictor.load_data(data_file)
        X, y = predictor.preprocess_data(data)
        X_train, X_test, y_train, y_test, y_pred = predictor.train_model(X, y)
        
        if args.test_file:
            # Load test data
            test_data = pd.read_csv(args.test_file)
            predictions = predictor.predict_cover_type(test_data)
        else:
            # Use a sample from the test set
            sample = X_test.iloc[:5]  # First 5 samples
            predictions = predictor.predict_cover_type(sample)
        
        print("\nPredictions:")
        for i, pred in enumerate(predictions):
            print(f"Sample {i+1}: {pred['cover_type']} (Class {pred['predicted_class']}) "
                  f"- Confidence: {max(pred['probabilities'].values()):.4f}")
        return
    
    if args.visualize:
        print("=== Generating Visualizations ===")
        data = predictor.load_data(data_file)
        X, y = predictor.preprocess_data(data)
        X_train, X_test, y_train, y_test, y_pred = predictor.train_model(X, y)
        predictor.visualize_results(y_test, y_pred)
        return
    
    # If no specific action is provided, show help
    parser.print_help()

if __name__ == "__main__":
    main()