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
if predictor.feature_names:
    print("Number of features:", len(predictor.feature_names))
    print("First 5 feature names:", predictor.feature_names[:5])

# Try to access feature importances
if predictor.model and predictor.feature_names:
    try:
        importances = predictor.model.feature_importances_
        print("Feature importances available:", len(importances))
        print("First 5 importances:", importances[:5])
    except Exception as e:
        print(f"Error accessing feature importances: {e}")
else:
    print("Cannot access feature importances: model or feature names not available")