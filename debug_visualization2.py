import sys
import traceback
from forest_cover_predictor import ForestCoverPredictor
import pandas as pd

try:
    # Initialize predictor
    print("Initializing predictor...")
    predictor = ForestCoverPredictor()

    # Load data
    print("Loading data...")
    data = predictor.load_data('train.csv')

    # Preprocess data
    print("Preprocessing data...")
    X, y = predictor.preprocess_data(data)

    # Train model
    print("Training model...")
    X_train, X_test, y_train, y_test, y_pred = predictor.train_model(X, y)

    # Check if model and feature names are available
    print("Model available:", predictor.model is not None)
    print("Feature names available:", predictor.feature_names is not None)
    if predictor.feature_names:
        print("Number of features:", len(predictor.feature_names))
        print("First 5 feature names:", predictor.feature_names[:5])

    # Try to generate feature importance manually
    print("Attempting to generate feature importance...")
    if predictor.model and predictor.feature_names:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            from config import VIZ_PARAMS, FILE_PATHS
            
            print("Libraries imported successfully")
            
            top_n = VIZ_PARAMS.get('feature_importance_top_n', 20)
            print("Top N features to show:", top_n)
            
            print("Creating feature importance DataFrame...")
            feature_importance = pd.DataFrame({
                'feature': predictor.feature_names,
                'importance': predictor.model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            print("Feature importance DataFrame created successfully")
            print(feature_importance.head())
            
            print("Generating plot...")
            plt.figure(figsize=VIZ_PARAMS.get('figure_size', (10, 8)))
            sns.barplot(data=feature_importance, x='importance', y='feature')
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('debug_feature_importance2.png')
            plt.close()
            print("Feature importance plot saved as debug_feature_importance2.png")
        except Exception as e:
            print(f"Error generating feature importance plot: {e}")
            traceback.print_exc()
    else:
        print("Cannot generate feature importance: model or feature names not available")
        if not predictor.model:
            print("Model is None")
        if not predictor.feature_names:
            print("Feature names is None or empty")

    print("Debug completed successfully")
except Exception as e:
    print(f"Error in debug script: {e}")
    traceback.print_exc()
    sys.exit(1)