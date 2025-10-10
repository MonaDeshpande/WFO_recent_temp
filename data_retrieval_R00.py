import pandas as pd
from sqlalchemy import create_engine
import logging
from datetime import datetime
import joblib
import os
from config_R00 import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, GC_FILE, START_DATE, END_DATE, GC_MAPPINGS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}')
        conn = engine.connect()
        logging.info("Database connection successful.")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        return None

def get_scada_data(conn, tags, start_date, end_date):
    """
    Retrieves SCADA data from the database.
    
    Args:
        conn (sqlalchemy.engine.Connection): The database connection object.
        tags (list): A list of SCADA tags to retrieve.
        start_date (str): The start date for the data query.
        end_date (str): The end date for the data query.
    
    Returns:
        pd.DataFrame: A DataFrame containing the retrieved SCADA data.
    """
    if conn is None:
        return pd.DataFrame()

    query_tags = ", ".join([f'"{tag}"' for tag in tags])
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    query = f"""
    SELECT "DateAndTime", {query_tags}
    FROM public."data_cleaning_with_report"
    WHERE "DateAndTime" >= '{start_dt.strftime('%Y-%m-%d %H:%M:%S')}' AND "DateAndTime" <= '{end_dt.strftime('%Y-%m-%d %H:%M:%S')}';
    """
    
    try:
        df = pd.read_sql(query, conn)
        logging.info(f"SCADA data for tags {tags} retrieved successfully.")
        return df
    except Exception as e:
        logging.error(f"Error retrieving SCADA data: {e}")
        return pd.DataFrame()

def get_gc_data(file_path, stream_ids):
    """
    Retrieves and processes GC data from a CSV file.
    
    Args:
        file_path (str): The path to the GC data CSV file.
        stream_ids (list): A list of GC stream IDs to retrieve.
        
    Returns:
        pd.DataFrame: A DataFrame with the processed GC data.
    """
    if not os.path.exists(file_path):
        logging.warning(f"GC data file not found at {file_path}. Returning empty DataFrame.")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(file_path)
        
        # Check for required columns before processing
        required_gc_cols = ['Analysis Date', 'Analysis  Time', 'Sample Detail', 'Naphth. % by GC']
        missing_cols = [col for col in required_gc_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing required columns in GC data file: {missing_cols}")
            return pd.DataFrame()

        # Combine 'Analysis Date' and 'Analysis  Time' into a single 'DateAndTime' column
        df['DateAndTime'] = pd.to_datetime(
            df['Analysis Date'] + ' ' + df['Analysis  Time'],
            format='%d.%m.%y %I.%M%p'
        )

        # Explicitly convert the 'Naphth. % by GC' column to a numeric type
        # 'errors=coerce' will turn any non-numeric values into NaN, preventing crashes
        df['Naphth. % by GC'] = pd.to_numeric(df['Naphth. % by GC'], errors='coerce')

        # Filter by stream IDs and use pivot_table to handle duplicate entries
        gc_data_pivot = df[df['Sample Detail'].isin(stream_ids)].pivot_table(
            index='DateAndTime',
            columns='Sample Detail',
            values='Naphth. % by GC',
            aggfunc='mean'  # Use the mean to aggregate duplicate timestamps
        ).reset_index()

        logging.info(f"GC data for streams {stream_ids} retrieved, pivoted, and combined successfully.")
        return gc_data_pivot

    except Exception as e:
        logging.error(f"Error processing GC data: {e}")
        return pd.DataFrame()

def resample_and_merge_data(scada_df, gc_df):
    """
    Resamples data to a common hourly frequency and merges.
    
    Args:
        scada_df (pd.DataFrame): The SCADA data DataFrame.
        gc_df (pd.DataFrame): The GC data DataFrame.
        
    Returns:
        pd.DataFrame: The merged and resampled DataFrame.
    """
    logging.info("Resampling data to a common hourly frequency...")
    
    # Set 'DateAndTime' as index for resampling
    scada_df = scada_df.set_index('DateAndTime')
    gc_df = gc_df.set_index('DateAndTime')
    
    # Resample to an hourly frequency, taking the mean of each hour's data
    # Use 'h' instead of 'H' to avoid a FutureWarning
    scada_resampled = scada_df.resample('h').mean()
    gc_resampled = gc_df.resample('h').mean()
    
    # Merge the resampled dataframes
    merged_data = pd.merge(scada_resampled, gc_resampled, on='DateAndTime', how='outer')
    
    # Fill missing values using forward-fill
    merged_data = merged_data.ffill()
    
    logging.info("Data successfully resampled and merged.")
    return merged_data.reset_index()


def main_data_retrieval(column_id, scada_tags, start_date, end_date, gc_file_path, gc_stream_ids, required_cols):
    """
    Main function to retrieve, merge, and preprocess all required data.
    
    Args:
        column_id (str): The ID of the distillation column being processed (e.g., 'C-00').
        scada_tags (list): List of SCADA tags to retrieve.
        start_date (str): The start date for data retrieval.
        end_date (str): The end date for data retrieval.
        gc_file_path (str): The path to the GC data file.
        gc_stream_ids (list): List of GC stream IDs.
        required_cols (list): List of all required columns for the final DataFrame.
    
    Returns:
        tuple: A tuple containing the merged DataFrame and the scaler (if applicable).
    """
    # 1. Retrieve SCADA data
    conn = get_db_connection()
    scada_df = get_scada_data(conn, scada_tags, start_date, end_date)
    
    # 2. Retrieve GC data
    gc_df = get_gc_data(gc_file_path, gc_stream_ids)
    
    # 3. Merge data sources
    if scada_df.empty or gc_df.empty:
        logging.warning("One or both data sources are empty. Cannot proceed with merge.")
        return pd.DataFrame(), None

    # Resample and merge the dataframes
    merged_data = resample_and_merge_data(scada_df, gc_df)

    # 4. Calculate EQ_Ratio
    # This must be done after merging, as it requires data from both sources.
    gc_stream_name = GC_MAPPINGS['column_to_stream_map'].get(column_id)
    heat_duty_col = f'HeatDuty_{column_id}_Reboiler'
    
    if gc_stream_name and heat_duty_col in merged_data.columns and gc_stream_name in merged_data.columns:
        # Avoid division by zero by replacing 0 with a small epsilon
        merged_data[gc_stream_name] = merged_data[gc_stream_name].replace(0, 1e-9)
        merged_data['EQ_Ratio'] = merged_data[heat_duty_col] / merged_data[gc_stream_name]
        logging.info(f"EQ_Ratio successfully calculated for column {column_id}.")
    else:
        logging.error(f"Cannot calculate EQ_Ratio: Missing required columns for calculation ({heat_duty_col} or {gc_stream_name}).")
        return pd.DataFrame(), None
    
    # Check if the final merged data contains all required columns and has data
    if merged_data.empty:
        logging.error("Merged data is empty after resampling. Cannot proceed.")
        return pd.DataFrame(), None
        
    missing_cols = [col for col in required_cols if col not in merged_data.columns]
    if missing_cols:
        logging.error(f"Merged data is missing the following required columns: {missing_cols}")
        return pd.DataFrame(), None

    return merged_data, None
