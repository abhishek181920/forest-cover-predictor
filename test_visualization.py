import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from forest_cover_predictor import ForestCoverPredictor

# Initialize predictor
predictor = ForestCoverPredictor()

# Load data
data = predictor.load_data('train.csv')

# Preprocess data
X, y = predictor.preprocess_data(data)

# Train model
X_train, X_test, y_train, y_test, y_pred = predictor.train_model(X, y)

# Check if model and feature names are available
print("Model available:", predictor.model is not None)
print("Feature names available:", predictor.feature_names is not None)

# Try to generate just the feature importance plot
if predictor.model and predictor.feature_names:
    try:
        print("Generating feature importance plot...")
        
        # Get feature importances
        importances = predictor.model.feature_importances_
        print(f"Number of importances: {len(importances)}")
        print(f"Number of feature names: {len(predictor.feature_names)}")
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': predictor.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(20)
        
        print("Feature importance DataFrame created:")
        print(feature_importance.head())
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save plot
        plt.savefig('test_feature_importance.png')
        plt.close()
        
        print("Feature importance plot saved as test_feature_importance.png")
        
        # Also try to save with the configured filename
        from config import FILE_PATHS
        filename = FILE_PATHS.get('feature_importance_plot', 'feature_importance.png')
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        print(f"Feature importance plot also saved as {filename}")
        
    except Exception as e:
        print(f"Error generating feature importance plot: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Cannot generate feature importance plot: model or feature names not available")