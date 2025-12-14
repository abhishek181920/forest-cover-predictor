from forest_cover_predictor import ForestCoverPredictor
import pandas as pd

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
if predictor.feature_names:
    print("Number of features:", len(predictor.feature_names))

# Try to generate feature importance manually
if predictor.model and predictor.feature_names:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from config import VIZ_PARAMS, FILE_PATHS
        
        top_n = VIZ_PARAMS.get('feature_importance_top_n', 20)
        print("Top N features to show:", top_n)
        
        feature_importance = pd.DataFrame({
            'feature': predictor.feature_names,
            'importance': predictor.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        print("Feature importance DataFrame created successfully")
        print(feature_importance.head())
        
        plt.figure(figsize=VIZ_PARAMS.get('figure_size', (10, 8)))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('debug_feature_importance.png')
        plt.close()
        print("Feature importance plot saved as debug_feature_importance.png")
    except Exception as e:
        print(f"Error generating feature importance plot: {e}")
else:
    print("Cannot generate feature importance: model or feature names not available")

print("Debug completed")