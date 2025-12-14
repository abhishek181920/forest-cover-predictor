import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Import our modules
from forest_cover_predictor import ForestCoverPredictor
from config import COVER_TYPE_MAPPING, FEATURE_GROUPS

# Set page configuration
st.set_page_config(
    page_title="Forest Cover Type Predictor",
    page_icon="🌲",
    layout="wide"
)

# Title and description
st.title("🌲 Forest Cover Type Prediction System")
st.markdown("""
This application predicts the type of forest cover based on cartographic data from the Roosevelt National Forest.
Enter the required information below to get a prediction.
""")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.predictor = None

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the mode",
    ["Home", "Data Explorer", "Model Training", "Prediction", "About"]
)

if app_mode == "Home":
    st.header("Welcome to the Forest Cover Type Prediction System")
    st.markdown("""
    This system uses machine learning to predict forest cover types based on:
    - Elevation and terrain features
    - Distance to hydrology, roads, and fire points
    - Hillshade indices
    - Wilderness area and soil type information
    
    Use the navigation menu to:
    - Explore sample data
    - Train the model
    - Make predictions on new data
    """)
    
    # Show sample data
    st.subheader("Sample Data Preview")
    predictor = ForestCoverPredictor()
    sample_data = predictor.create_sample_dataset()
    st.dataframe(sample_data.head(10))
    
elif app_mode == "Data Explorer":
    st.header("Data Exploration")
    
    # Load data
    predictor = ForestCoverPredictor()
    data = predictor.create_sample_dataset()  # In a real app, you'd load actual data
    
    st.subheader("Dataset Overview")
    st.write(f"Dataset shape: {data.shape}")
    st.write(f"Number of features: {data.shape[1] - 1}")  # Excluding target
    st.write(f"Number of samples: {data.shape[0]}")
    
    # Show basic statistics
    st.subheader("Basic Statistics")
    st.dataframe(data.describe())
    
    # Show target distribution
    st.subheader("Cover Type Distribution")
    cover_dist = data['Cover_Type'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cover_dist.index, cover_dist.values, color=plt.cm.Set3(np.linspace(0, 1, len(cover_dist))))
    ax.set_xlabel('Cover Type')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Forest Cover Types')
    ax.set_xticks(cover_dist.index)
    ax.set_xticklabels([COVER_TYPE_MAPPING[i] for i in cover_dist.index], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Show correlation heatmap for continuous variables
    st.subheader("Correlation Heatmap (Continuous Variables)")
    continuous_features = FEATURE_GROUPS['continuous']
    if all(feature in data.columns for feature in continuous_features):
        corr_data = data[continuous_features]
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix of Continuous Features')
        st.pyplot(fig)
    else:
        st.warning("Not all continuous features found in the dataset.")

elif app_mode == "Model Training":
    st.header("Model Training")
    
    # Training parameters
    st.subheader("Training Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        max_depth = st.slider("Max Depth", 1, 50, 10)
    
    with col2:
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)
    
    # Train button
    if st.button("Train Model"):
        with st.spinner("Training the model..."):
            # Create predictor
            predictor = ForestCoverPredictor()
            
            # Load data
            data = predictor.load_data('train.csv')
            
            # Preprocess data
            X, y = predictor.preprocess_data(data)
            
            # Train model with selected parameters
            st.session_state.predictor = ForestCoverPredictor()
            
            # Update model parameters
            from config import MODEL_PARAMS
            MODEL_PARAMS['n_estimators'] = n_estimators
            MODEL_PARAMS['max_depth'] = max_depth if max_depth > 0 else None
            MODEL_PARAMS['min_samples_split'] = min_samples_split
            MODEL_PARAMS['min_samples_leaf'] = min_samples_leaf
            
            # Train the model
            X_train, X_test, y_train, y_test, y_pred = st.session_state.predictor.train_model(X, y)
            
            # Store in session state
            st.session_state.model_trained = True
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            
            st.success("Model trained successfully!")
            
            # Show results
            st.subheader("Training Results")
            accuracy = np.mean(y_test == y_pred)
            st.metric("Accuracy", f"{accuracy:.4f}")
            
            # Show classification report
            st.subheader("Classification Report")
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Show confusion matrix
            st.subheader("Confusion Matrix")
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[COVER_TYPE_MAPPING[i] for i in range(1, 8)],
                        yticklabels=[COVER_TYPE_MAPPING[i] for i in range(1, 8)], ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

elif app_mode == "Prediction":
    st.header("Forest Cover Type Prediction")
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Model Training' section.")
        
        # Provide option to quickly train a model
        if st.button("Quick Train (Default Settings)"):
            with st.spinner("Training model with default settings..."):
                predictor = ForestCoverPredictor()
                data = predictor.load_data('train.csv')
                X, y = predictor.preprocess_data(data)
                X_train, X_test, y_train, y_test, y_pred = predictor.train_model(X, y)
                
                st.session_state.predictor = predictor
                st.session_state.model_trained = True
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                
                st.success("Model trained successfully! You can now make predictions.")
    else:
        st.success("Model is ready for predictions!")
        
        # Input form for prediction
        st.subheader("Enter Forest Data")
        
        # Create tabs for different feature groups
        tab1, tab2, tab3 = st.tabs(["Terrain Features", "Distance Features", "Categorical Features"])
        
        # Initialize input dictionary
        input_data = {}
        
        with tab1:
            st.markdown("#### Terrain Characteristics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                input_data['Elevation'] = st.number_input("Elevation (meters)", min_value=0, value=2500)
                input_data['Aspect'] = st.slider("Aspect (degrees)", 0, 360, 45)
                
            with col2:
                input_data['Slope'] = st.slider("Slope (degrees)", 0, 50, 10)
                input_data['Hillshade_9am'] = st.slider("Hillshade 9am", 0, 255, 200)
                
            with col3:
                input_data['Hillshade_Noon'] = st.slider("Hillshade Noon", 0, 255, 220)
                input_data['Hillshade_3pm'] = st.slider("Hillshade 3pm", 0, 255, 150)
        
        with tab2:
            st.markdown("#### Distances")
            col1, col2 = st.columns(2)
            
            with col1:
                input_data['Horizontal_Distance_To_Hydrology'] = st.number_input(
                    "Horizontal Distance to Hydrology", min_value=0, value=100)
                input_data['Vertical_Distance_To_Hydrology'] = st.number_input(
                    "Vertical Distance to Hydrology", value=50)
                    
            with col2:
                input_data['Horizontal_Distance_To_Roadways'] = st.number_input(
                    "Horizontal Distance to Roadways", min_value=0, value=500)
                input_data['Horizontal_Distance_To_Fire_Points'] = st.number_input(
                    "Horizontal Distance to Fire Points", min_value=0, value=1000)
        
        with tab3:
            st.markdown("#### Wilderness Areas")
            wilderness_cols = [f'Wilderness_Area{i}' for i in range(1, 5)]
            for i, col_name in enumerate(wilderness_cols):
                input_data[col_name] = st.checkbox(col_name, value=(i == 0))
            
            st.markdown("#### Soil Types")
            st.caption("Select one or more soil types (showing first 10 for brevity)")
            soil_cols = [f'Soil_Type{i}' for i in range(1, 11)]  # Showing first 10 for UI simplicity
            for i, col_name in enumerate(soil_cols):
                input_data[col_name] = st.checkbox(col_name, value=(i == 0))
            
            # Add remaining soil types as zeros for a complete dataset
            for i in range(11, 41):
                input_data[f'Soil_Type{i}'] = 0
        
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Add missing columns (in case of incomplete input)
        all_columns = FEATURE_GROUPS['continuous'] + \
                     FEATURE_GROUPS['wilderness_areas'] + \
                     [f'Soil_Type{i}' for i in range(1, 41)]
        
        for col in all_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[all_columns]
        
        # Prediction button
        if st.button("Predict Forest Cover Type"):
            with st.spinner("Making prediction..."):
                try:
                    # Make prediction
                    predictions = st.session_state.predictor.predict_cover_type(input_df)
                    
                    # Display results
                    st.subheader("Prediction Results")
                    pred = predictions[0]
                    
                    # Show predicted class with styling
                    st.markdown(f"### Predicted Cover Type: **{pred['cover_type']}**")
                    st.markdown(f"**Class:** {pred['predicted_class']}")
                    
                    # Show confidence
                    confidence = max(pred['probabilities'].values())
                    st.progress(float(confidence))
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    
                    # Show probability distribution
                    st.subheader("Probability Distribution")
                    prob_data = pd.DataFrame([
                        {'Cover Type': COVER_TYPE_MAPPING[cls], 
                         'Probability': prob} 
                        for cls, prob in pred['probabilities'].items()
                        if prob > 0.01  # Only show significant probabilities
                    ]).sort_values('Probability', ascending=False)
                    
                    if not prob_data.empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(prob_data['Cover Type'], prob_data['Probability'], 
                                    color=plt.cm.Set3(np.linspace(0, 1, len(prob_data))))
                        ax.set_ylabel('Probability')
                        ax.set_title('Prediction Probability Distribution')
                        ax.set_ylim(0, 1)
                        
                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.annotate(f'{height:.2%}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),
                                        textcoords="offset points",
                                        ha='center', va='bottom')
                        
                        plt.xticks(rotation=45, ha='right')
                        st.pyplot(fig)
                        
                        st.dataframe(prob_data.set_index('Cover Type'))
                    else:
                        st.info("All probabilities are very low. This might indicate an unusual input.")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

elif app_mode == "About":
    st.header("About This Application")
    st.markdown("""
    ### Forest Cover Type Prediction System
    
    This application uses machine learning to predict forest cover types based on 
    cartographic variables from the Roosevelt National Forest dataset.
    
    #### Dataset Information
    The dataset contains information about 30m x 30m patches of forest land with:
    - **Continuous Variables**: Elevation, Aspect, Slope, Distances, Hillshade indices
    - **Categorical Variables**: Wilderness areas (4) and Soil types (40)
    - **Target Variable**: 7 forest cover types
    
    #### Machine Learning Model
    - **Algorithm**: Random Forest Classifier
    - **Features**: 54 cartographic features
    - **Classes**: 7 forest cover types
    
    #### Forest Cover Types
    1. Spruce/Fir
    2. Lodgepole Pine
    3. Ponderosa Pine
    4. Cottonwood/Willow
    5. Aspen
    6. Douglas-fir
    7. Krummholz
    
    #### Technical Details
    - Built with Python, Scikit-learn, and Streamlit
    - Implements data preprocessing and feature scaling
    - Provides model evaluation metrics
    - Supports interactive predictions
    """)
    
    st.subheader("How to Use This Application")
    st.markdown("""
    1. **Data Explorer**: View sample data and statistics
    2. **Model Training**: Train the model with custom parameters
    3. **Prediction**: Enter forest data to predict cover type
    4. **About**: Learn more about the system and dataset
    
    For best results, ensure your input data is similar to the training data characteristics.
    """)

# Footer
st.markdown("---")
st.markdown("🌲 Forest Cover Type Prediction System | Built with Streamlit")