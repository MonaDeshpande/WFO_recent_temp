import psycopg2
import sys
import csv
import pandas as pd
from datetime import datetime
import xlsxwriter
import numpy as np
from io import StringIO
from scipy.stats import zscore

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PG_HOST = "localhost"
PG_PORT = "5432"
PG_USER = "postgres"
PG_PASSWORD = "ADMIN"  # <-- IMPORTANT: Add your PostgreSQL password here
PG_DB_NAME = "scada_data_analysis"
PG_RAW_TABLE = "wide_scada_data"
PG_MAPPING_TABLE = "tag_mapping"
PG_CLEANED_TABLE = "data_cleaning_with_report"
TAGS_CSV_FILE = "TAG_INDEX_FINAL.csv"

# --- USER INPUT ---
START_DATE = "2025-09-22 15:00:00"
END_DATE = "2025-10-08 11:00:00"

# Faulty value constant
FAULTY_VALUE = 32767
# Thresholds for anomaly detection
FLOW_ANOMALY_THRESHOLD = 0.5
TI_SPIKE_THRESHOLD = 20  # 20°C
SMOOTHING_WINDOW_SIZE = 3 # in minutes
ZSCORE_THRESHOLD = 3  # New threshold for statistical anomalies

# New constants for heat duty calculations
CP_TF66 = 2.10  # kJ/(kg·K)
CP_WATER = 4.186  # kJ/(kg·K)
RHO_WATER = 1000  # kg/m³
RHO_TF66 = 750  # kg/m³

# Distillation column tag sequences
DISTILLATION_COLUMNS = {
    "Column_1": {
        "temperature": ["TI61", "TI62", "TI63", "TI64", "TI65"],
        "pressure": {"PTT": "PI61", "PTB": "PI62"}
    },
    "Column_2": {
        "temperature": ["TI03", "TI04", "TI05", "TI06"],
        "pressure": {"PTT": "PI03", "PTB": "PI04"}
    },
    "Column_3": {
        "temperature": ["TI13", "TI14", "TI15", "TI16", "TI17", "TI18", "TI19", "TI20", "TI21", "TI22", "TI23", "TI24"],
        "pressure": {"PTT": "PI13", "PTB": "PI24"}
    },
    "Column_4": {
        "temperature": ["TI31", "TI32", "TI33", "TI34", "TI35", "TI36", "TI37", "TI38", "TI39"],
        "pressure": {"PTT": "PI31", "PTB": "PI39"}
    }
}

# Tag types for multi-sensor logic
TAG_TYPES = {
    "TI": ["TI61", "TI62", "TI63", "TI64", "TI65", "TI03", "TI04", "TI05", "TI06", "TI13", "TI14", "TI15", "TI16", "TI17", "TI18", "TI19", "TI20", "TI21", "TI22", "TI23", "TI24", "TI31", "TI32", "TI33", "TI34", "TI35", "TI36", "TI37", "TI38", "TI39", "TI215"],
    "PI": ["PI61", "PI62", "PI03", "PI04", "PI13", "PI24", "PI31", "PI39"],
    "LI": ["LI61", "LI62", "LI03", "LI04"],
    "FI": ["FI08", "FI09", "FI10", "FT08", "FT09", "FT10", "FT-01"]
}

# Example operating ranges (adjust these based on your process data)
OPERATING_RANGES = {
    "TI61": (100, 150),
    "TI215": (325, 340),
}

# --- Distillation Column Tags for calculations (as per user input)
COLUMN_MAPPING = {
    "C-00": {"PTB": "PTB-04", "PTT": "PTT-04", "Reflux_Flow": "FT-Reflux_C00", "Top_Product_Flow": "FT-Top_Prod_C00"},
    "C-01": {"PTB": "PTB-01", "PTT": "PTT-01", "Reflux_Flow": "FT-08", "Top_Product_Flow": "FT-02"},
    "C-02": {"PTB": "PTB-02", "PTT": "PTT-02", "Reflux_Flow": "FT-09", "Top_Product_Flow": "FT-03"},
    "C-03": {"PTB": "PTB-03", "PTT": "PTT-03", "Reflux_Flow": "FT-10", "Top_Product_Flow": "FT-04"}
}

# --- Heat Exchanger Tags for calculations (as per user input)
HEAT_EXCHANGERS = {
    "C-00_Condenser": {"flow": "FI-101", "supply_temp": "TI-110", "return_temp": "TI-112", "fluid": "water"},
    "C-01_Condenser": {"flow": "FI-102", "supply_temp": "TI-110", "return_temp": "TI-101", "fluid": "water"},
    "C-02_Condenser": {"flow": "FI-103", "supply_temp": "TI-110", "return_temp": "TI-104", "fluid": "water"},
    "C-03_Condenser": {"flow": "FI-104", "supply_temp": "TI-110", "return_temp": "TI-107", "fluid": "water"},
    "C-00_Reboiler": {"flow": "FT-204", "supply_temp": "TI-221", "return_temp": "TI-222", "fluid": "tf66"},
    "C-01_Reboiler": {"flow": "FT-201", "supply_temp": "TI-203", "return_temp": "TI-204", "fluid": "tf66"},
    "C-02_Reboiler": {"flow": "FT-202", "supply_temp": "TI-207", "return_temp": "TI-208", "fluid": "tf66"},
    "C-03_Reboiler": {"flow": "FT-203", "supply_temp": "TI-211", "return_temp": "TI-212", "fluid": "tf66"}
}

# --------------------------------------------------------------------------------------------------

def create_mapping_table(pg_cursor, pg_conn, tag_data):
    """Creates the tag mapping table and populates it."""
    print(f"\n--- Creating mapping table '{PG_MAPPING_TABLE}' if it doesn't exist ---")
    create_mapping_table_query = f"""
    CREATE TABLE IF NOT EXISTS "{PG_MAPPING_TABLE}" (
        "TagIndex" INTEGER PRIMARY KEY,
        "TagName" VARCHAR(255) UNIQUE
    );
    """
    pg_cursor.execute(create_mapping_table_query)

    insert_mapping_query = f"""
    INSERT INTO "{PG_MAPPING_TABLE}" ("TagIndex", "TagName")
    VALUES (%s, %s)
    ON CONFLICT ("TagIndex") DO UPDATE SET "TagName" = EXCLUDED."TagName";
    """
    pg_cursor.executemany(insert_mapping_query, tag_data)
    pg_conn.commit()
    print(f"✅ Successfully inserted/updated {pg_cursor.rowcount} tags.")

def get_existing_columns(pg_cursor, table_name):
    """Fetches the names of all existing columns in a given table."""
    query = f"""
    SELECT "column_name"
    FROM information_schema.columns
    WHERE table_name = '{table_name}'
    ORDER BY ordinal_position;
    """
    try:
        pg_cursor.execute(query)
        return [row[0] for row in pg_cursor.fetchall()]
    except psycopg2.ProgrammingError:
        # Table does not exist
        return []

def get_required_columns(tag_data):
    """Generates a list of all required columns for the cleaned data table."""
    # Start with the tags from the CSV file
    required_columns = [tag for _, tag in tag_data]
    
    # Add the anomaly and metadata columns
    required_columns.extend([
        "is_faulty_sensor", "is_temp_anomaly", "is_pressure_anomaly", "is_process_excursion",
        "is_flow_level_anomaly", "is_plant_shutdown", "is_boiler_anomaly",
        "imputed_with", "notes"
    ])
    
    # Add calculated columns dynamically
    for col in COLUMN_MAPPING.keys():
        required_columns.append(f"DP_{col}")
        required_columns.append(f"Reflux_Ratio_{col}")
    
    for exchanger in HEAT_EXCHANGERS.keys():
        required_columns.append(f"HeatDuty_{exchanger}")
    
    # Add smoothed columns
    smoothing_tags = TAG_TYPES["TI"] + TAG_TYPES["PI"]
    for tag in smoothing_tags:
        required_columns.append(f"{tag}_smoothed")
        
    return required_columns

def update_cleaned_table_schema(pg_cursor, pg_conn, tag_data):
    """
    Creates the cleaned data table if it doesn't exist.
    If it exists, it checks for and adds any new columns from the CSV.
    """
    print(f"\n--- Checking/Updating table '{PG_CLEANED_TABLE}' schema ---")
    
    required_columns = get_required_columns(tag_data)
    existing_columns = get_existing_columns(pg_cursor, PG_CLEANED_TABLE)

    if not existing_columns:
        print(f"✅ Table '{PG_CLEANED_TABLE}' not found. Creating a new one...")
        
        # Define column definitions based on required columns
        columns_definitions = ['"DateAndTime" TIMESTAMP PRIMARY KEY']
        for col in required_columns:
            if col.startswith("is_"):
                columns_definitions.append(f'"{col}" BOOLEAN DEFAULT FALSE')
            elif col in ["imputed_with", "notes"]:
                columns_definitions.append(f'"{col}" TEXT')
            else:
                columns_definitions.append(f'"{col}" DOUBLE PRECISION')
                
        columns_str = ",\n".join(columns_definitions)
        
        create_table_query = f"""
        CREATE TABLE "{PG_CLEANED_TABLE}" (
            {columns_str}
        );
        """
        pg_cursor.execute(create_table_query)
        pg_conn.commit()
        print(f"✅ Table '{PG_CLEANED_TABLE}' successfully created.")
    else:
        print(f"✅ Table '{PG_CLEANED_TABLE}' already exists. Checking for new columns...")
        
        new_columns = [col for col in required_columns if col not in existing_columns]
        
        if new_columns:
            print(f"⚠️ Found {len(new_columns)} new columns to add.")
            for col in new_columns:
                if col.startswith("is_"):
                    alter_query = f"""ALTER TABLE "{PG_CLEANED_TABLE}" ADD COLUMN "{col}" BOOLEAN DEFAULT FALSE;"""
                elif col in ["imputed_with", "notes"]:
                    alter_query = f"""ALTER TABLE "{PG_CLEANED_TABLE}" ADD COLUMN "{col}" TEXT;"""
                else:
                    alter_query = f"""ALTER TABLE "{PG_CLEANED_TABLE}" ADD COLUMN "{col}" DOUBLE PRECISION;"""
                
                pg_cursor.execute(alter_query)
                print(f"  ➡️ Added column '{col}'.")
            pg_conn.commit()
            print("✅ Schema updated successfully.")
        else:
            print("➡️ No new columns to add. Schema is up to date.")

def generate_excel_report(summary_data, detail_log, start_dt, end_dt):
    """Generates an Excel report with summary and detailed logs."""
    report_filename = f"SCADA_Report_{start_dt.strftime('%Y%m%d')}_to_{end_dt.strftime('%Y%m%d')}.xlsx"
    print(f"\n--- Generating Excel Report: {report_filename} ---")
    
    writer = pd.ExcelWriter(report_filename, engine='xlsxwriter')
    workbook = writer.book

    summary_sheet = workbook.add_worksheet('Summary')
    summary_sheet.write('A1', 'SCADA Data Analysis Report')
    summary_sheet.write('A2', f"Period: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    summary_sheet.write('A4', 'Anomaly Summary')
    anomaly_summary = pd.DataFrame(summary_data['anomaly_counts'], index=['Total Count']).T
    anomaly_summary.to_excel(writer, sheet_name='Summary', startrow=5, startcol=0)
    
    summary_sheet.write('A10', 'Top 20 Most Faulty Instruments by Percentage')
    faulty_instruments_df = pd.DataFrame(summary_data['faulty_instruments'])
    # Ensure sorting works by converting the string percentage to a float
    if not faulty_instruments_df.empty:
        faulty_instruments_df['Faulty Readings (%)'] = faulty_instruments_df['Faulty Readings (%)'].str.rstrip('%').astype(float)
        faulty_instruments_df = faulty_instruments_df.sort_values(by='Faulty Readings (%)', ascending=False).head(20)
        faulty_instruments_df.to_excel(writer, sheet_name='Summary', startrow=11, startcol=0, index=False)
    else:
        summary_sheet.write('A11', 'No faulty instruments found.')

    detail_df = pd.DataFrame(detail_log)
    if not detail_df.empty:
        detail_df.to_excel(writer, sheet_name='Detailed Log', index=False)

    writer.close()
    print(f"✅ Report saved to {report_filename}")

def add_calculated_columns(df):
    """
    Performs all DP, reflux ratio, and heat duty calculations and adds them to the DataFrame.
    Returns the modified DataFrame and a list of the new column names.
    """
    print("--- Performing DP and Heat Duty calculations ---")

    calculated_cols = []

    # Differential Pressure (DP) and Reflux Ratio calculations
    for col_name, tags in COLUMN_MAPPING.items():
        dp_col = f"DP_{col_name}"
        reflux_ratio_col = f"Reflux_Ratio_{col_name}"

        # Check if both required tags exist before performing the calculation
        required_dp_tags = [tags.get("PTB"), tags.get("PTT")]
        if all(tag in df.columns for tag in required_dp_tags):
            df[dp_col] = df[tags["PTB"]] - df[tags["PTT"]]
            calculated_cols.append(dp_col)

        required_reflux_tags = [tags.get("Reflux_Flow"), tags.get("Top_Product_Flow")]
        if all(tag in df.columns for tag in required_reflux_tags):
            df[reflux_ratio_col] = np.where(df[tags["Top_Product_Flow"]] != 0,
                                            df[tags["Reflux_Flow"]] / df[tags["Top_Product_Flow"]],
                                            0)
            calculated_cols.append(reflux_ratio_col)

    # --- UPDATED: Heat Duty calculations ---
    for exchanger_name, tags in HEAT_EXCHANGERS.items():
        heat_duty_col = f"HeatDuty_{exchanger_name}"
        flow_tag = tags["flow"]
        supply_temp_tag = tags["supply_temp"]
        return_temp_tag = tags["return_temp"]

        # Check if all required tags exist before performing the calculation
        required_heat_tags = [flow_tag, supply_temp_tag, return_temp_tag]
        if all(tag in df.columns for tag in required_heat_tags):
            # Calculate temperature difference
            delta_T = df[return_temp_tag] - df[supply_temp_tag]

            # Calculate heat duty in kJ/h
            if tags["fluid"] == "water":
                # Removed RHO_WATER
                heat_duty_kJ_per_h = df[flow_tag] * CP_WATER * delta_T
            elif tags["fluid"] == "tf66":
                # Removed RHO_TF66
                heat_duty_kJ_per_h = df[flow_tag] * CP_TF66 * delta_T

            # Convert kJ/h to kW (1 kW = 1 kJ/s; 1 hour = 3600 seconds)
            df[heat_duty_col] = heat_duty_kJ_per_h / 3600

            calculated_cols.append(heat_duty_col)

    return df, calculated_cols

def process_scada_data_in_range(start_timestamp, end_timestamp):
    """Connects to PostgreSQL, processes raw SCADA data, and generates a report."""
    pg_conn = None
    try:
        print(f"\n--- Connecting to PostgreSQL and processing data from {start_timestamp} to {end_timestamp} ---")
        pg_conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB_NAME)
        pg_cursor = pg_conn.cursor()
        print("✅ Successfully connected to PostgreSQL.")

        tag_data = []
        try:
            with open(TAGS_CSV_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    tag_data.append((int(row[0]), row[1]))
            print(f"📁 Found {len(tag_data)} tags in {TAGS_CSV_FILE}.")
        except FileNotFoundError:
            print(f"❌ Error: {TAGS_CSV_FILE} not found. Please ensure the file exists.")
            return

        create_mapping_table(pg_cursor, pg_conn, tag_data)
        
        all_tags_from_csv = [t[1] for t in tag_data]
        
        # Update schema to include all tags, calculated columns, and smoothed columns
        update_cleaned_table_schema(pg_cursor, pg_conn, tag_data)
        
        columns_to_select = ', '.join([f'"{tag}"' for tag in all_tags_from_csv])
        
        print("--- Fetching wide-format data from the database ---")
        fetch_wide_data_query = f"""
        SELECT
            "DateAndTime",
            {columns_to_select}
        FROM
            "{PG_RAW_TABLE}"
        WHERE
            "DateAndTime" BETWEEN %s AND %s
        ORDER BY
            "DateAndTime" ASC;
        """
        raw_data_df = pd.read_sql_query(fetch_wide_data_query, pg_conn, params=(start_timestamp, end_timestamp))
        
        if raw_data_df.empty:
            print("No data found in the specified range.")
            return
            
        processed_df = raw_data_df.set_index('DateAndTime').resample('1T').mean().reset_index()
        processed_df.index = processed_df['DateAndTime']
        
        print(f"Processing {len(processed_df)} minutes of new data...")

        # --- Rule 1: Faulty Sensor Value Check and Process Excursion ---
        print("--- Applying Rule 1: Faulty Sensor Value Check ---")
        flag_cols = [
            'is_faulty_sensor', 'is_temp_anomaly', 'is_pressure_anomaly',
            'is_process_excursion', 'is_flow_level_anomaly',
            'is_plant_shutdown', 'is_boiler_anomaly'
        ]
        
        # Initialize new flag columns
        for col in flag_cols:
            processed_df[col] = False
        processed_df['imputed_with'] = None
        processed_df['notes'] = ''
        
        # Create a boolean mask to track which values need imputation
        imputation_mask = processed_df.isnull().copy()

        detailed_log = []
        anomaly_counts = {col: 0 for col in flag_cols}

        ti_tags = TAG_TYPES["TI"]
        
        # HYBRID APPROACH: Rule-Based Cleaning (First Pass) - Definitive Faults
        print("--- Applying Rule-Based Cleaning (First Pass) ---")
        for tag in all_tags_from_csv:
            if tag in processed_df.columns:
                
                # Rule 1: Direct faulty value (32767)
                faulty_mask = (processed_df[tag] == FAULTY_VALUE)
                if faulty_mask.any():
                    imputation_mask.loc[faulty_mask, tag] = True
                    processed_df.loc[faulty_mask, 'is_faulty_sensor'] = True
                    processed_df.loc[faulty_mask, 'notes'] += f'Definitive faulty reading ({FAULTY_VALUE}). '
                    anomaly_counts['is_faulty_sensor'] += faulty_mask.sum()
                    processed_df.loc[faulty_mask, tag] = np.nan # Replace with NaN for imputation
                    for ts in processed_df[faulty_mask].index:
                        detailed_log.append({
                            'DateAndTime': ts,
                            'Instrument': tag,
                            'Anomaly_Type': 'Definitive Fault',
                            'Value': processed_df.loc[ts, tag],
                            'Notes': f'Detected faulty value {FAULTY_VALUE}.'
                        })
                
                # Rule 2: Sudden temperature spike
                if tag in ti_tags:
                    diff = processed_df[tag].diff().abs()
                    spike_mask = (diff > TI_SPIKE_THRESHOLD) & (processed_df[tag].shift(-1).abs() < processed_df[tag].abs())
                    if spike_mask.any():
                        imputation_mask.loc[spike_mask, tag] = True
                        processed_df.loc[spike_mask, 'is_faulty_sensor'] = True
                        processed_df.loc[spike_mask, 'notes'] += 'Sudden temp spike detected. '
                        anomaly_counts['is_faulty_sensor'] += spike_mask.sum()
                        processed_df.loc[spike_mask, tag] = np.nan # Replace with NaN
                        for ts in processed_df[spike_mask].index:
                            detailed_log.append({
                                'DateAndTime': ts,
                                'Instrument': tag,
                                'Anomaly_Type': 'Definitive Fault',
                                'Value': processed_df.loc[ts, tag],
                                'Notes': 'Sudden temperature spike detected.'
                            })
                
                # Rule 3: Handle all negative values as faults, regardless of tag type
                negative_value_mask = processed_df[tag] < 0
                if negative_value_mask.any():
                    imputation_mask.loc[negative_value_mask, tag] = True
                    processed_df.loc[negative_value_mask, 'is_faulty_sensor'] = True
                    processed_df.loc[negative_value_mask, 'notes'] += 'Negative value detected. '
                    anomaly_counts['is_faulty_sensor'] += negative_value_mask.sum()
                    processed_df.loc[negative_value_mask, tag] = np.nan # Replace with NaN
                    for ts in processed_df[negative_value_mask].index:
                        detailed_log.append({
                            'DateAndTime': ts,
                            'Instrument': tag,
                            'Anomaly_Type': 'Definitive Fault',
                            'Value': processed_df.loc[ts, tag],
                            'Notes': 'Negative value detected.'
                        })

        # HYBRID APPROACH: Statistical Anomaly Detection (Second Pass) - Flag but Retain
        print("--- Applying Statistical Anomaly Detection (Second Pass) ---")
        for tag in all_tags_from_csv:
            if tag in processed_df.columns and processed_df[tag].dtype in [np.float64, np.int64]:
                # Only analyze columns that are not completely empty and don't contain NaNs from the first pass
                numeric_data = processed_df[tag].dropna()
                if len(numeric_data) > 2:
                    try:
                        z_scores = np.abs(zscore(numeric_data))
                        is_anomaly_series = pd.Series(z_scores > ZSCORE_THRESHOLD, index=numeric_data.index)
                        is_anomaly_mask = processed_df.index.isin(is_anomaly_series[is_anomaly_series].index)

                        if is_anomaly_mask.any():
                            # Flag the anomaly but DO NOT replace the value with NaN
                            processed_df.loc[is_anomaly_mask, 'is_faulty_sensor'] = True
                            processed_df.loc[is_anomaly_mask, 'notes'] += f'Statistical outlier (Z-score > {ZSCORE_THRESHOLD}) detected. '
                            anomaly_counts['is_faulty_sensor'] += is_anomaly_mask.sum()
                            
                            for ts in processed_df[is_anomaly_mask].index:
                                detailed_log.append({
                                    'DateAndTime': ts,
                                    'Instrument': tag,
                                    'Anomaly_Type': 'Statistical Outlier',
                                    'Value': processed_df.loc[ts, tag],
                                    'Notes': f"Value is a statistical outlier (Z-score > {ZSCORE_THRESHOLD})."
                                })
                    except Exception as e:
                        print(f"❌ Error during z-score calculation for {tag}: {e}")
                        continue

        # --- Rule 2: Process-Based Logic (Temperature and Pressure) ---
        print("--- Applying Rule 2: Process-Based Logic ---")
        for col_name, col_data in DISTILLATION_COLUMNS.items():
            # Temperature Profile Check
            temp_tags = col_data.get("temperature", [])
            for i in range(len(temp_tags) - 1):
                current_tag = temp_tags[i]
                next_tag = temp_tags[i+1]
                if current_tag in processed_df.columns and next_tag in processed_df.columns:
                    anomaly_mask = (processed_df[current_tag] < processed_df[next_tag]) & (processed_df[current_tag].notna()) & (processed_df[next_tag].notna())
                    if anomaly_mask.any():
                        processed_df.loc[anomaly_mask, 'is_temp_anomaly'] = True
                        processed_df.loc[anomaly_mask, 'notes'] += f'Temp profile anomaly between {current_tag} and {next_tag} in {col_name}. '
                        anomaly_counts['is_temp_anomaly'] += anomaly_mask.sum()
                        for ts in processed_df[anomaly_mask].index:
                            detailed_log.append({
                                'DateAndTime': ts,
                                'Instrument': current_tag,
                                'Anomaly_Type': 'Process Anomaly',
                                'Value': processed_df.loc[ts, current_tag],
                                'Notes': f'Temperature {current_tag} is less than {next_tag}.'
                            })

            # Pressure Profile Check
            pressure_tags = col_data.get("pressure", {})
            if "PTT" in pressure_tags and "PTB" in pressure_tags:
                ptt = pressure_tags["PTT"]
                ptb = pressure_tags["PTB"]
                if ptt in processed_df.columns and ptb in processed_df.columns:
                    anomaly_mask = (processed_df[ptt] > processed_df[ptb]) & (processed_df[ptt].notna()) & (processed_df[ptb].notna())
                    if anomaly_mask.any():
                        processed_df.loc[anomaly_mask, 'is_pressure_anomaly'] = True
                        processed_df.loc[anomaly_mask, 'notes'] += f'Pressure profile anomaly in {col_name}. '
                        anomaly_counts['is_pressure_anomaly'] += anomaly_mask.sum()
                        for ts in processed_df[anomaly_mask].index:
                            detailed_log.append({
                                'DateAndTime': ts,
                                'Instrument': ptt,
                                'Anomaly_Type': 'Process Anomaly',
                                'Value': processed_df.loc[ts, ptt],
                                'Notes': f'Pressure at top ({ptt}) is greater than pressure at bottom ({ptb}).'
                            })
                        
        # --- Rule 3: Multi-Sensor Correlation Logic ---
        print("--- Applying Rule 3: Multi-Sensor Correlation ---")
        li_tags = TAG_TYPES["LI"]
        
        # Flow/Level Discrepancy (Line Choking)
        for li_tag in li_tags:
            fi_tag = li_tag.replace('LI', 'FT')
            if li_tag in processed_df.columns and fi_tag in processed_df.columns:
                level_increasing = processed_df[li_tag].diff() > 0.1
                flow_low = processed_df[fi_tag] <= FLOW_ANOMALY_THRESHOLD
                anomaly_mask = level_increasing & flow_low
                if anomaly_mask.any():
                    processed_df.loc[anomaly_mask, 'is_flow_level_anomaly'] = True
                    processed_df.loc[anomaly_mask, 'notes'] += f'Flow/Level anomaly (choking) at {li_tag}. '
                    anomaly_counts['is_flow_level_anomaly'] += anomaly_mask.sum()
                    for ts in processed_df[anomaly_mask].index:
                        detailed_log.append({
                            'DateAndTime': ts,
                            'Instrument': li_tag,
                            'Anomaly_Type': 'Process Anomaly',
                            'Value': processed_df.loc[ts, li_tag],
                            'Notes': 'Tank level increasing while flow is low.'
                        })

        # --- Rule 4: Event-Triggered Checks ---
        print("--- Applying Rule 4: Event-Triggered Checks ---")
        # Plant Shutdown Event
        ft_shutdown_tags = ["FT08", "FT09", "FT10"]
        if all(tag in processed_df.columns for tag in ft_shutdown_tags):
            all_ft_zero = (processed_df[ft_shutdown_tags] <= FLOW_ANOMALY_THRESHOLD).all(axis=1)
            if all_ft_zero.any():
                processed_df.loc[all_ft_zero, 'is_plant_shutdown'] = True
                processed_df.loc[all_ft_zero, 'notes'] += 'Plant Shutdown Event. '
                anomaly_counts['is_plant_shutdown'] += all_ft_zero.sum()
                for ts in processed_df[all_ft_zero].index:
                    detailed_log.append({
                        'DateAndTime': ts,
                        'Instrument': 'Multiple',
                        'Anomaly_Type': 'Plant Event',
                        'Value': None,
                        'Notes': 'All major flow meters are at or near zero.'
                    })

        # Boiler Issue
        if 'TI215' in processed_df.columns:
            boiler_mask = (processed_df['TI215'] < 325) | (processed_df['TI215'] > 340)
            if boiler_mask.any():
                processed_df.loc[boiler_mask, 'is_boiler_anomaly'] = True
                processed_df.loc[boiler_mask, 'notes'] += 'Boiler-Related Anomaly. '
                anomaly_counts['is_boiler_anomaly'] += boiler_mask.sum()
                for ts in processed_df[boiler_mask].index:
                    detailed_log.append({
                        'DateAndTime': ts,
                        'Instrument': 'TI215',
                        'Anomaly_Type': 'Process Anomaly',
                        'Value': processed_df.loc[ts, 'TI215'],
                        'Notes': 'Boiler temperature is outside its normal operating range.'
                    })

        # --- Rule 5: Handling and Imputing Faulty Values ---
        # NOTE: This section now only imputes values previously set to NaN
        print("--- Applying Rule 5: Handling and Imputing ---")
        
        # Linear interpolation for NaNs, then Last Good Known (LKG) for remaining NaNs
        for tag in all_tags_from_csv:
            if tag in processed_df.columns:
                original_nan_mask = processed_df[tag].isnull()
                
                # Interpolate only the NaNs we created from definitive faults
                processed_df[tag] = processed_df[tag].interpolate(method='linear', limit_direction='both')
                
                interpolated_mask = processed_df[tag].notna() & original_nan_mask
                if interpolated_mask.any():
                    processed_df.loc[interpolated_mask, 'imputed_with'] = 'Linear Interpolation'
                
                # Apply LKG for any remaining NaNs (e.g., at the very start of the data)
                lkg_mask = processed_df[tag].isnull()
                processed_df[tag].ffill(inplace=True)
                
                # Only update the 'imputed_with' column for newly filled LKG values
                processed_df.loc[lkg_mask & processed_df[tag].notna(), 'imputed_with'] = 'LKG'
                
        # --- Perform Calculations (AFTER Imputation) ---
        processed_df, calculated_cols = add_calculated_columns(processed_df)
        
        # --- Basic Data Smoothing (AFTER Imputation) ---
        print("--- Applying basic data smoothing ---")
        smoothing_tags = TAG_TYPES["TI"] + TAG_TYPES["PI"]
        smoothed_cols = []
        for tag in smoothing_tags:
            if tag in processed_df.columns:
                smoothed_column_name = f"{tag}_smoothed"
                processed_df[smoothed_column_name] = processed_df[tag].rolling(
                    window=f'{SMOOTHING_WINDOW_SIZE}T',
                    min_periods=1,
                    center=True
                ).mean()
                smoothed_cols.append(smoothed_column_name)
                
        # --- Prepare for insertion into the database ---
        final_df_columns = ['DateAndTime'] + all_tags_from_csv + calculated_cols + flag_cols + ['imputed_with', 'notes'] + smoothed_cols
        processed_df_final = processed_df.reset_index(drop=True).reindex(columns=final_df_columns)
        processed_df_final = processed_df_final.replace({np.nan: None})
        
        # Log to the database
        print("--- Inserting cleaned data into the database ---")
        sql_cols = ', '.join([f'"{c}"' for c in final_df_columns])
        sql_params = ', '.join(['%s'] * len(final_df_columns))
        
        update_cols = [f'"{c}" = EXCLUDED."{c}"' for c in final_df_columns if c != "DateAndTime"]
        update_clause = ', '.join(update_cols)
        
        insert_query = f"""
            INSERT INTO "{PG_CLEANED_TABLE}" ({sql_cols})
            VALUES ({sql_params})
            ON CONFLICT ("DateAndTime") DO UPDATE SET
                {update_clause}
            """
        
        pg_cursor.executemany(insert_query, [tuple(row) for row in processed_df_final.values])

        pg_conn.commit()
        print(f"✅ Successfully inserted/updated {len(processed_df_final)} rows into {PG_CLEANED_TABLE}.")
        
        # --- Report Generation ---
        print("\n--- Generating Report Data ---")
        faulty_instruments_list = []
        for tag in all_tags_from_csv:
            if tag in processed_df.columns:
                total_readings = len(processed_df)
                
                # Count definitive faulty conditions (now includes Z-score flagged points)
                faulty_count = processed_df.loc[processed_df['is_faulty_sensor'], tag].notna().sum()
                
                if total_readings > 0:
                    percentage = (faulty_count / total_readings) * 100
                    faulty_instruments_list.append({
                        "Instrument Name": tag,
                        "Total Faulty Readings": int(faulty_count),
                        "Total Readings": total_readings,
                        "Faulty Readings (%)": f"{percentage:.2f}%"
                    })
        
        faulty_instruments_list = sorted(faulty_instruments_list, key=lambda x: float(x['Faulty Readings (%)'].strip('%')), reverse=True)

        summary_report = {
            "anomaly_counts": anomaly_counts,
            "faulty_instruments": faulty_instruments_list
        }
        
        generate_excel_report(summary_report, detailed_log, start_timestamp, end_timestamp)

    except psycopg2.Error as e:
        print(f"❌ PostgreSQL connection or query failed. Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        if pg_conn:
            pg_conn.close()
            print("\nDatabase connection closed.")


# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    start_dt = datetime.strptime(START_DATE, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(END_DATE, '%Y-%m-%d %H:%M:%S')
    process_scada_data_in_range(start_dt, end_dt)