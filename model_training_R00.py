import os
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_save_model(data, column_id):
    """
    Trains a machine learning model using the provided data and saves the
    trained model and scaler to disk.
    """
    logging.info(f"--- Starting model training for {column_id} ---")
    
    # Define features and target
    # We are using all numerical columns from the SCADA data as features, and the GC value as the target.
    # The `EQ_Ratio` is what we want to predict.
    try:
        # The GC stream is the target
        target_column = 'gc_stream'
        
        # All other numerical columns (excluding the target and EQ_Ratio) are features
        features = data.drop(columns=[target_column, 'EQ_Ratio'], errors='ignore')
        
        # Check if features and target are available
        if target_column not in data.columns:
            logging.error(f"Target column '{target_column}' not found in data for {column_id}. Cannot train model.")
            return

        X = features
        y = data[target_column]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logging.info("Features successfully scaled.")

        # Train a Random Forest Regressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        logging.info("Model training complete.")

        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logging.info(f"Model performance for {column_id}:")
        logging.info(f"Mean Squared Error: {mse:.2f}")
        logging.info(f"R-squared: {r2:.2f}")

        # Save the model and scaler
        from config_R00 import MODEL_PATH
        model_dir = os.path.join(MODEL_PATH, column_id)
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logging.info(f"Model and scaler for {column_id} saved successfully to {model_dir}.")

    except Exception as e:
        logging.error(f"An error occurred during model training for {column_id}: {e}")
