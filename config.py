# Configuration file for Forest Cover Type Prediction System

# Model parameters
MODEL_PARAMS = {
    'random_state': 42,
    'n_estimators': 100,
    'n_jobs': -1,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

# Data preprocessing parameters
PREPROCESSING_PARAMS = {
    'test_size': 0.2,
    'validation_size': 0.1
}

# Feature groups
FEATURE_GROUPS = {
    'continuous': [
        'Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points'
    ],
    'wilderness_areas': [f'Wilderness_Area{i}' for i in range(1, 5)],
    'soil_types': [f'Soil_Type{i}' for i in range(1, 41)]
}

# Target variable mapping
COVER_TYPE_MAPPING = {
    1: 'Spruce/Fir',
    2: 'Lodgepole Pine',
    3: 'Ponderosa Pine',
    4: 'Cottonwood/Willow',
    5: 'Aspen',
    6: 'Douglas-fir',
    7: 'Krummholz'
}

# Visualization settings
VIZ_PARAMS = {
    'figure_size': (10, 8),
    'confusion_matrix_cmap': 'Blues',
    'feature_importance_top_n': 20
}

# File paths
FILE_PATHS = {
    'data_file': 'train.csv',
    'confusion_matrix_plot': 'confusion_matrix.png',
    'feature_importance_plot': 'feature_importance.png'
}