import os
from datetime import datetime

# --- Database Credentials ---
DB_HOST = 'localhost'
DB_NAME = 'scada_data_analysis'
DB_USER = 'postgres'
DB_PASSWORD = 'ADMIN'

# --- File Paths and Names ---
# Define the base directory for model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_artifacts')
GC_FILE = os.path.join(BASE_DIR, 'gc_data.csv')

# Define the paths for the trained model and scaler
TRAINED_MODEL_PATH = os.path.join(MODEL_PATH, 'trained_model.joblib')
TRAINED_SCALER_PATH = os.path.join(MODEL_PATH, 'scaler.joblib')

# --- Date Range for Analysis ---
# Set the start and end dates for data retrieval
START_DATE = datetime(2025, 9, 3).strftime('%Y-%m-%d')
END_DATE = datetime(2025, 9, 16).strftime('%Y-%m-%d')

# --- Column Mappings ---
# List of columns to be analyzed
COLUMNS = ['C-00', 'C-01', 'C-02', 'C-03']

# SCADA tag mappings for each column
SCADA_MAPPINGS = {
    'column_to_scada_tags_map': {
        'C-00': ['FT-01', 'FT-61', 'FT-62', 'TI-202', 'HeatDuty_C-00_Reboiler'],
        'C-01': ['FT-62', 'FT-02', 'TI-04', 'TI-204', 'FT-05', 'FT-08', 'HeatDuty_C-01_Reboiler'],
        'C-02': ['FT-02', 'FT-03', 'FT-09', 'TI-14', 'FT-06', 'HeatDuty_C-02_Reboiler'],
        'C-03': ['FT-06', 'TI-32', 'FT-10', 'FT-07', 'FT-04', 'HeatDuty_C-03_Reboiler']
    },
    'feed_flow_map': {
        'C-00': 'FT-01',
        'C-01': 'FT-02',
        'C-02': 'FT-03',
        'C-03': 'FT-06'
    },
    'heat_duty_map': {
        'C-00': 'HeatDuty_C-00_Reboiler',
        'C-01': 'HeatDuty_C-01_Reboiler',
        'C-02': 'HeatDuty_C-02_Reboiler',
        'C-03': 'HeatDuty_C-03_Reboiler'
    }
}

# GC stream mappings for each column
GC_MAPPINGS = {
    'column_to_stream_map': {
        'C-00': 'P-01',
        'C-01': 'C-01-B',
        'C-02': 'C-02-T',
        'C-03': 'C-03-B'
    }
}

# Mapping for standardizing GC column names
GC_COLUMN_RENAME = {
    'Analysis Date': 'analysis_date',
    'Analysis Time': 'analysis_time',
    'Sample Detail': 'stream_name',
    'Naphth. % by GC': 'naphth_by_gc'
}
