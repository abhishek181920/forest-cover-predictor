import unittest
import pandas as pd
import numpy as np
from forest_cover_predictor import ForestCoverPredictor

class TestForestCoverPredictor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.predictor = ForestCoverPredictor()
        # Create a small sample dataset for testing
        self.sample_data = pd.DataFrame({
            'Elevation': [2500, 2600, 2700, 2800, 2900],
            'Aspect': [45, 90, 135, 180, 225],
            'Slope': [10, 15, 20, 25, 30],
            'Horizontal_Distance_To_Hydrology': [100, 200, 300, 400, 500],
            'Vertical_Distance_To_Hydrology': [50, 75, 100, 125, 150],
            'Horizontal_Distance_To_Roadways': [500, 1000, 1500, 2000, 2500],
            'Hillshade_9am': [200, 210, 220, 230, 240],
            'Hillshade_Noon': [220, 230, 240, 250, 260],
            'Hillshade_3pm': [150, 160, 170, 180, 190],
            'Horizontal_Distance_To_Fire_Points': [1000, 1500, 2000, 2500, 3000],
            'Wilderness_Area1': [1, 0, 1, 0, 1],
            'Wilderness_Area2': [0, 1, 0, 1, 0],
            'Wilderness_Area3': [0, 0, 1, 0, 1],
            'Wilderness_Area4': [1, 1, 0, 1, 0],
            'Soil_Type1': [1, 0, 1, 0, 1],
            'Soil_Type2': [0, 1, 0, 1, 0],
            'Soil_Type3': [1, 1, 0, 0, 1],
            'Cover_Type': [1, 2, 1, 3, 2]
        })
    
    def test_initialization(self):
        """Test that the predictor initializes correctly."""
        self.assertIsNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.scaler)
    
    def test_create_sample_dataset(self):
        """Test that sample dataset creation works."""
        sample_df = self.predictor.create_sample_dataset()
        self.assertIsInstance(sample_df, pd.DataFrame)
        self.assertGreater(len(sample_df), 0)
        self.assertIn('Cover_Type', sample_df.columns)
    
    def test_preprocess_data(self):
        """Test data preprocessing functionality."""
        X, y = self.predictor.preprocess_data(self.sample_data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsNotNone(y)
        self.assertEqual(len(X), len(y))
        self.assertNotIn('Cover_Type', X.columns)
    
    def test_predict_without_training(self):
        """Test that prediction raises error when model is not trained."""
        with self.assertRaises(ValueError):
            sample = self.sample_data.drop('Cover_Type', axis=1).iloc[:1]
            self.predictor.predict_cover_type(sample)
    
    def test_complete_pipeline(self):
        """Test the complete pipeline from preprocessing to prediction."""
        # Preprocess data
        X, y = self.predictor.preprocess_data(self.sample_data)
        
        # Train model (using a small subset for testing)
        X_train = X.iloc[:4]
        y_train = y.iloc[:4]
        X_test = X.iloc[4:]
        y_test = y.iloc[4:]
        
        # Manually train a simple model for testing
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        self.predictor.model = model
        self.predictor.feature_names = X.columns.tolist()
        
        # Test prediction
        predictions = self.predictor.predict_cover_type(X_test)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(X_test))
        self.assertIn('predicted_class', predictions[0])
        self.assertIn('cover_type', predictions[0])

if __name__ == '__main__':
    unittest.main()