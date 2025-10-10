import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import random
import time
from datetime import datetime

# Import the main data retrieval function and configuration
from data_retrieval_R00 import main_data_retrieval
from config_R00 import START_DATE, END_DATE, GC_FILE, SCADA_TAGS_C00, SCADA_TAGS_C01, SCADA_TAGS_C02, SCADA_TAGS_C03

# Define the dictionary of relevant SCADA tags for each column.
SCADA_TAGS_BY_COLUMN = {
    'C-00': SCADA_TAGS_C00,
    'C-01': SCADA_TAGS_C01,
    'C-02': SCADA_TAGS_C02,
    'C-03': SCADA_TAGS_C03
}

def train_models(scada_df, column_id):
    """
    Trains a predictive maintenance model and an anomaly detection model on historical data.
    """
    print(f"\n--- Training Models for {column_id} ---")
    if scada_df is None or scada_df.empty:
        print("No data available to train models.")
        return None, None, None

    # --- Predictive Maintenance Model (Random Forest) ---
    # Create a mock sensor degradation trend for demonstration
    df_pred_maint = scada_df.copy()
    df_pred_maint['Day'] = (df_pred_maint['DATEANDTIME'] - df_pred_maint['DATEANDTIME'].min()).dt.days + 1
    np.random.seed(42)
    base_pressure = 50.0
    pressure_increase_rate = 0.5
    noise = np.random.normal(0, 2, len(df_pred_maint))
    df_pred_maint['Pressure_Reading'] = base_pressure + df_pred_maint['Day']**1.5 * 0.05 + noise # Non-linear trend

    # Train a RandomForestRegressor
    model_pm = RandomForestRegressor(n_estimators=100, random_state=42)
    model_pm.fit(df_pred_maint[['Day']], df_pred_maint['Pressure_Reading'])

    # --- Anomaly Detection Model (Isolation Forest) ---
    # Select all available numerical tags for multivariate analysis
    numerical_cols = [col for col in scada_df.columns if col not in ['DATEANDTIME', 'DateAndTime', 'dateandtime']]
    if len(numerical_cols) < 2:
        print(f"Not enough numerical tags for anomaly detection for {column_id}. Skipping.")
        return model_pm, None, None

    features = scada_df.dropna(subset=numerical_cols)[numerical_cols].values
    
    # Scale the data for better performance with some models
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Train an IsolationForest model
    model_ad = IsolationForest(contamination='auto', random_state=42)
    model_ad.fit(scaled_features)

    return model_pm, model_ad, scaler, numerical_cols

def run_real_time_analysis(column_id, scada_df, pm_model, ad_model, scaler, ad_cols):
    """
    Simulates a real-time data stream and performs analysis.
    """
    print(f"\n--- Running Real-time Analysis for {column_id} ---")
    if pm_model is None:
        print("Skipping predictive maintenance analysis due to missing model.")
    if ad_model is None:
        print("Skipping anomaly detection analysis due to missing model.")
        
    critical_threshold = 100.0

    # Simulate a stream of data points one at a time
    for i in range(len(scada_df)):
        live_data_point = scada_df.iloc[i:i+1].copy()
        
        # --- Predictive Maintenance Inference ---
        if pm_model:
            # Predict the pressure for the next 5 days based on the current day
            current_day = (live_data_point['DATEANDTIME'].iloc[0] - scada_df['DATEANDTIME'].min()).days + 1
            future_days = np.arange(current_day, current_day + 5).reshape(-1, 1)
            predicted_pressure = pm_model.predict(future_days)
            
            # Check for a future failure alert
            if any(p >= critical_threshold for p in predicted_pressure):
                print(f"ALERT! PM: Failure for {column_id} predicted within the next 5 days!")

        # --- Anomaly Detection Inference ---
        if ad_model and ad_cols:
            live_features = live_data_point[ad_cols].values
            
            # The mock anomaly is still added here for demonstration
            if i == 50:
                live_features = np.array([[10, 55, 12, 11]]).reshape(1, -1)
                
            # Check for anomalies
            live_scaled = scaler.transform(live_features)
            is_anomaly = ad_model.predict(live_scaled)
            
            if is_anomaly[0] == -1: # -1 indicates an anomaly
                print(f"ALERT! AD: Anomaly detected for {column_id} at {live_data_point['DATEANDTIME'].iloc[0]}")
        
        # Optional: Simulate a delay to mimic real-time streaming
        # time.sleep(0.1) 
        
        # Limit output for brevity
        if i % 100 == 0:
            print(f"Processing data point {i}...")

    print(f"\n--- Real-time analysis for {column_id} complete. ---\n")

def main():
    """Main execution function to run advanced analysis on all columns."""
    for column_id, tags in SCADA_TAGS_BY_COLUMN.items():
        print("="*60)
        print(f"Beginning advanced analysis for {column_id}")
        
        # Get data for the current column using the data pipeline
        scada_data, _ = main_data_retrieval(column_id, tags, START_DATE, END_DATE, GC_FILE, [])
        
        if scada_data is None or scada_data.empty:
            print(f"No SCADA data available for {column_id}. Skipping.")
            continue
            
        # Step 1: Train the models on historical data
        pm_model, ad_model, scaler, ad_cols = train_models(scada_data, column_id)
        
        # Step 2: Run the real-time simulation
        run_real_time_analysis(column_id, scada_data, pm_model, ad_model, scaler, ad_cols)
        
if __name__ == "__main__":
    main()