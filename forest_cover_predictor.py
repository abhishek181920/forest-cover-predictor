import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our data loader utility
from data_loader import load_forest_cover_data, validate_dataset_structure, prepare_data_for_modeling

# Import configuration
from config import MODEL_PARAMS, PREPROCESSING_PARAMS, COVER_TYPE_MAPPING, VIZ_PARAMS, FILE_PATHS

class ForestCoverPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, file_path):
        """Load the forest cover dataset"""
        # Use our data loader utility
        data = load_forest_cover_data(file_path)
        if data is not None:
            # Validate dataset structure
            if validate_dataset_structure(data):
                print("Dataset structure is valid.")
            else:
                print("Warning: Dataset structure validation failed.")
            return data
        else:
            print(f"File {file_path} not found. Creating sample dataset for demonstration.")
            return self.create_sample_dataset()
    
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration purposes"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample features
        data = {
            'Elevation': np.random.randint(2000, 4000, n_samples),
            'Aspect': np.random.randint(0, 360, n_samples),
            'Slope': np.random.randint(0, 50, n_samples),
            'Horizontal_Distance_To_Hydrology': np.random.randint(0, 1000, n_samples),
            'Vertical_Distance_To_Hydrology': np.random.randint(-200, 500, n_samples),
            'Horizontal_Distance_To_Roadways': np.random.randint(0, 5000, n_samples),
            'Hillshade_9am': np.random.randint(0, 255, n_samples),
            'Hillshade_Noon': np.random.randint(0, 255, n_samples),
            'Hillshade_3pm': np.random.randint(0, 255, n_samples),
            'Horizontal_Distance_To_Fire_Points': np.random.randint(0, 5000, n_samples)
        }
        
        # Create wilderness area columns
        for i in range(1, 5):
            data[f'Wilderness_Area{i}'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        
        # Create soil type columns
        for i in range(1, 41):
            data[f'Soil_Type{i}'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Create target variable with realistic distribution
        cover_types = np.random.choice([1, 2, 3, 4, 5, 6, 7], n_samples, 
                                     p=[0.3, 0.3, 0.1, 0.05, 0.05, 0.1, 0.1])
        df['Cover_Type'] = cover_types
        
        print(f"Sample dataset created with shape: {df.shape}")
        return df
    
    def explore_data(self, data):
        """Perform exploratory data analysis"""
        print("\n=== Data Exploration ===")
        print(f"Dataset shape: {data.shape}")
        print(f"\nColumn names: {list(data.columns)}")
        print(f"\nFirst 5 rows:")
        print(data.head())
        print(f"\nData info:")
        print(data.info())
        print(f"\nBasic statistics:")
        print(data.describe())
        print(f"\nMissing values:")
        print(data.isnull().sum())
        
        # Distribution of target variable
        print(f"\nCover Type Distribution:")
        print(data['Cover_Type'].value_counts().sort_index())
        
        return data
    
    def preprocess_data(self, data):
        """Preprocess the data for modeling"""
        print("\n=== Data Preprocessing ===")
        
        # Store feature names
        self.feature_names = [col for col in data.columns if col != 'Cover_Type']
        
        # Separate features and target
        if 'Cover_Type' in data.columns:
            X = data.drop('Cover_Type', axis=1)
            y = data['Cover_Type']
        else:
            X = data
            y = None
            
        print(f"Features shape: {X.shape}")
        if y is not None:
            print(f"Target shape: {y.shape}")
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y
    
    def train_model(self, X, y):
        """Train the Random Forest classifier"""
        print("\n=== Model Training ===")
        
        # Split the data
        test_size = PREPROCESSING_PARAMS.get('test_size', 0.2)
        random_state = MODEL_PARAMS.get('random_state', 42)
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train Random Forest Classifier with config parameters
        self.model = RandomForestClassifier(
            n_estimators=MODEL_PARAMS.get('n_estimators', 100),
            random_state=MODEL_PARAMS.get('random_state', 42),
            n_jobs=MODEL_PARAMS.get('n_jobs', -1),
            max_depth=MODEL_PARAMS.get('max_depth', None),
            min_samples_split=MODEL_PARAMS.get('min_samples_split', 2),
            min_samples_leaf=MODEL_PARAMS.get('min_samples_leaf', 1)
        )
        
        self.model.fit(X_train, y_train)
        print("Model training completed!")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_train, X_test, y_train, y_test, y_pred
    
    def visualize_results(self, y_test, y_pred):
        """Visualize model performance"""
        print("\n=== Results Visualization ===")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=VIZ_PARAMS.get('figure_size', (10, 8)))
        sns.heatmap(cm, annot=True, fmt='d', cmap=VIZ_PARAMS.get('confusion_matrix_cmap', 'Blues'),
                    xticklabels=[COVER_TYPE_MAPPING[i] for i in range(1, 8)],
                    yticklabels=[COVER_TYPE_MAPPING[i] for i in range(1, 8)])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(FILE_PATHS.get('confusion_matrix_plot', 'confusion_matrix.png'))
        plt.show()
        
        # Feature Importance
        if self.model and self.feature_names:
            top_n = VIZ_PARAMS.get('feature_importance_top_n', 20)
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=VIZ_PARAMS.get('figure_size', (10, 8)))
            sns.barplot(data=feature_importance, x='importance', y='feature')
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(FILE_PATHS.get('feature_importance_plot', 'feature_importance.png'))
            plt.show()
    
    def predict_cover_type(self, sample_data):
        """Predict forest cover type for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
            
        # Scale the sample data
        sample_scaled = self.scaler.transform(sample_data)
        
        # Make prediction
        prediction = self.model.predict(sample_scaled)
        probabilities = self.model.predict_proba(sample_scaled)
        
        results = []
        for i, pred in enumerate(prediction):
            result = {
                'predicted_class': int(pred),
                'cover_type': COVER_TYPE_MAPPING.get(int(pred), f'Unknown Class {pred}'),
                'probabilities': dict(zip(range(1, 8), probabilities[i]))
            }
            results.append(result)
            
        return results

def main():
    """Main function to run the forest cover prediction system"""
    # Initialize predictor
    predictor = ForestCoverPredictor()
    
    # Load data
    data = predictor.load_data('train.csv')
    
    # Explore data
    data = predictor.explore_data(data)
    
    # Preprocess data
    X, y = predictor.preprocess_data(data)
    
    # Train model
    X_train, X_test, y_train, y_test, y_pred = predictor.train_model(X, y)
    
    # Visualize results
    predictor.visualize_results(y_test, y_pred)
    
    # Example prediction on a sample
    print("\n=== Sample Prediction ===")
    sample = X_test.iloc[:1]  # Take first test sample
    predictions = predictor.predict_cover_type(sample)
    
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}:")
        print(f"  Predicted Cover Type: {pred['cover_type']} (Class {pred['predicted_class']})")
        print(f"  Probabilities:")
        for cls, prob in pred['probabilities'].items():
            print(f"    {COVER_TYPE_MAPPING[cls]}: {prob:.4f}")

if __name__ == "__main__":
    main()