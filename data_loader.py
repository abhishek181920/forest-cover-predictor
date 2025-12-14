import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_forest_cover_data(file_path):
    """
    Load the forest cover dataset from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file containing the dataset
    
    Returns:
    pandas.DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def validate_dataset_structure(data):
    """
    Validate that the dataset has the expected structure.
    
    Parameters:
    data (pandas.DataFrame): Dataset to validate
    
    Returns:
    bool: True if dataset structure is valid, False otherwise
    """
    if data is None:
        return False
    
    # Expected columns
    expected_columns = [
        'Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points'
    ]
    
    # Check for continuous variables
    missing_continuous = [col for col in expected_columns if col not in data.columns]
    if missing_continuous:
        print(f"Warning: Missing continuous variables: {missing_continuous}")
    
    # Check for wilderness area columns (4 binary columns)
    wilderness_cols = [col for col in data.columns if col.startswith('Wilderness_Area')]
    if len(wilderness_cols) != 4:
        print(f"Warning: Expected 4 Wilderness_Area columns, found {len(wilderness_cols)}")
    
    # Check for soil type columns (40 binary columns)
    soil_type_cols = [col for col in data.columns if col.startswith('Soil_Type')]
    if len(soil_type_cols) != 40:
        print(f"Warning: Expected 40 Soil_Type columns, found {len(soil_type_cols)}")
    
    # Check for target variable
    if 'Cover_Type' not in data.columns:
        print("Warning: 'Cover_Type' column not found in dataset")
        return False
    
    print("Dataset structure validation completed.")
    return True

def prepare_data_for_modeling(data, test_size=0.2, random_state=42):
    """
    Prepare the dataset for modeling by separating features and target.
    
    Parameters:
    data (pandas.DataFrame): Input dataset
    test_size (float): Proportion of dataset to include in the test split
    random_state (int): Random state for reproducibility
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    if data is None:
        return None, None, None, None
    
    # Separate features and target
    if 'Cover_Type' not in data.columns:
        print("Error: 'Cover_Type' column not found in dataset")
        return None, None, None, None
    
    X = data.drop('Cover_Type', axis=1)
    y = data['Cover_Type']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data prepared for modeling:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")
    
    return X_train, X_test, y_train, y_test

def get_feature_groups(data):
    """
    Group features by type for analysis.
    
    Parameters:
    data (pandas.DataFrame): Input dataset
    
    Returns:
    dict: Dictionary with feature groups
    """
    if data is None:
        return {}
    
    feature_groups = {
        'continuous': [
            'Elevation', 'Aspect', 'Slope',
            'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points'
        ],
        'wilderness_areas': [col for col in data.columns if col.startswith('Wilderness_Area')],
        'soil_types': [col for col in data.columns if col.startswith('Soil_Type')],
        'target': ['Cover_Type'] if 'Cover_Type' in data.columns else []
    }
    
    return feature_groups

# Example usage
if __name__ == "__main__":
    # This is just for demonstration
    print("Forest Cover Data Loader Utility")
    print("================================")
    print("This module provides utilities for loading and preparing the forest cover dataset.")
    print("\nFunctions available:")
    print("- load_forest_cover_data(file_path)")
    print("- validate_dataset_structure(data)")
    print("- prepare_data_for_modeling(data)")
    print("- get_feature_groups(data)")