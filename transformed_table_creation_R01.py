import psycopg2
import sys
import csv
import time
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SQL_SERVER_NAME = r"WFOPC3"
SQL_DB_NAME = "Report1database"
SQL_TABLE_NAME = "dbo.FloatTable"
PG_HOST = "localhost"
PG_PORT = "5432"
PG_USER = "postgres"
PG_PASSWORD = "ADMIN"  # <-- IMPORTANT: Add your PostgreSQL password here
PG_DB_NAME = "scada_data_analysis"
PG_RAW_TABLE = "raw_data"
PG_MAPPING_TABLE = "tag_mapping"
PG_TRANSFORMED_TABLE = "wide_scada_data"
TAGS_CSV_FILE = "TAG_INDEX_FINAL.csv"

# --- USER INPUT ---
# Placeholder: Change this to your desired start date for the initial run.
# The format must be 'YYYY-MM-DD HH:MM:SS'. After the first run,
# the script will automatically continue from the last processed time.
START_DATE = "2025-09-21 00:00:00"

# --- ENGINEERING CONSTANTS ---
# Specific Heat Capacity (Cp) values for heat duty calculation.
# Assumed units: kJ/kg·K (or kJ/kg·°C).
CP_COOLING_WATER = 4.184  # Standard value for water
CP_THERMIC_FLUID_TF66 = 1.47  # Typical value for Therminol 66

# ==============================================================================
# Main Logic
# ==============================================================================
def get_tag_data(pg_cursor):
    """Reads tag data from the CSV and handles database mapping."""
    tag_data = []
    try:
        with open(TAGS_CSV_FILE, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                tag_data.append((int(row[0]), row[1]))
        print(f"📁 Found {len(tag_data)} tags in {TAGS_CSV_FILE}.")
    except FileNotFoundError:
        print(f"❌ Error: {TAGS_CSV_FILE} not found. Please ensure the file exists.")
        return None
        
    return tag_data

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

def get_existing_columns(pg_cursor):
    """Fetches the names of all existing columns in the transformed table."""
    query = f"""
    SELECT "column_name" 
    FROM information_schema.columns 
    WHERE table_name = '{PG_TRANSFORMED_TABLE}' 
    ORDER BY ordinal_position;
    """
    pg_cursor.execute(query)
    return [row[0] for row in pg_cursor.fetchall()]

def update_transformed_table_schema(pg_cursor, pg_conn, tag_data):
    """
    Creates the transformed table if it doesn't exist.
    If it exists, it checks for and adds any new columns from the CSV.
    """
    print(f"\n--- Checking/Updating table '{PG_TRANSFORMED_TABLE}' schema ---")

    # List of all required columns based on CSV tags and calculated values
    required_columns = [tag for _, tag in tag_data]
    required_columns.extend([
        "DP_C-00", "DP_C-01", "DP_C-02", "DP_C-03",
        "Heat_Duty_Condenser_C-00", "Heat_Duty_Condenser_C-01",
        "Heat_Duty_Condenser_C-02", "Heat_Duty_Condenser_C-03",
        "Heat_Duty_Reboiler_C-00", "Heat_Duty_Reboiler_C-01",
        "Heat_Duty_Reboiler_C-02", "Heat_Duty_Reboiler_C-03",
        "Unmapped_TagIndex", "Unmapped_Value"
    ])

    try:
        existing_columns = get_existing_columns(pg_cursor)
    except psycopg2.ProgrammingError:
        # The table does not exist, so we create it.
        existing_columns = []

    if not existing_columns:
        print(f"✅ Table '{PG_TRANSFORMED_TABLE}' not found. Creating a new one...")
        # Define dynamic columns from CSV tags
        columns_definitions = [f'"{col}" DOUBLE PRECISION' for col in required_columns]
        
        columns_str = ",\n".join(columns_definitions)
        
        create_table_query = f"""
        CREATE TABLE "{PG_TRANSFORMED_TABLE}" (
            "DateAndTime" TIMESTAMP PRIMARY KEY,
            {columns_str}
        );
        """
        pg_cursor.execute(create_table_query)
        pg_conn.commit()
        print(f"✅ Table '{PG_TRANSFORMED_TABLE}' successfully created with the initial schema.")
    else:
        print(f"✅ Table '{PG_TRANSFORMED_TABLE}' already exists. Checking for new columns...")
        
        # Check for and add any new columns
        new_columns = [col for col in required_columns if col not in existing_columns]
        
        if new_columns:
            print(f"⚠️ Found {len(new_columns)} new columns to add: {', '.join(new_columns)}")
            for col in new_columns:
                alter_table_query = f"""
                ALTER TABLE "{PG_TRANSFORMED_TABLE}" ADD COLUMN "{col}" DOUBLE PRECISION;
                """
                pg_cursor.execute(alter_table_query)
                print(f"  ➡️ Added column '{col}'.")
            pg_conn.commit()
            print("✅ Schema updated successfully.")
        else:
            print("➡️ No new columns to add. Schema is up to date.")


def process_scada_data():
    """
    Main function to connect to PostgreSQL and process data.
    """
    pg_conn = None
    try:
        print("Connecting to PostgreSQL...")
        pg_conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB_NAME)
        pg_cursor = pg_conn.cursor()
        print("✅ Successfully connected to PostgreSQL.")

        # Read tags from CSV and create/update the mapping table
        tag_data = get_tag_data(pg_cursor)
        if tag_data is None:
            return
            
        create_mapping_table(pg_cursor, pg_conn, tag_data)
        
        # Check and update schema before processing data
        update_transformed_table_schema(pg_cursor, pg_conn, tag_data)

        # Find the last processed timestamp
        get_last_timestamp_query = f"""
        SELECT MAX("DateAndTime") FROM "{PG_TRANSFORMED_TABLE}";
        """
        pg_cursor.execute(get_last_timestamp_query)
        last_processed_timestamp = pg_cursor.fetchone()[0]
        start_timestamp_for_this_run = last_processed_timestamp or START_DATE
        print(f"➡️ Starting data processing from: {start_timestamp_for_this_run}")

        # Dynamically generate and execute the pivot query
        print(f"\n--- Inserting dynamic pivoted data into '{PG_TRANSFORMED_TABLE}' ---")
        
        # Base pivot cases for all tags from CSV
        pivot_cases = [f"""MAX(CASE WHEN "TagName" = '{tag}' THEN "Val"::DOUBLE PRECISION END) AS "{tag}" """ for _, tag in tag_data]

        # Add calculated columns to the select statement
        calculated_cases = [
            f"""MAX(CASE WHEN "TagName" = 'PTB-04' THEN "Val"::DOUBLE PRECISION END) - MAX(CASE WHEN "TagName" = 'PTT-04' THEN "Val"::DOUBLE PRECISION END) AS "DP_C-00" """,
            f"""MAX(CASE WHEN "TagName" = 'PTB-01' THEN "Val"::DOUBLE PRECISION END) - MAX(CASE WHEN "TagName" = 'PTT-01' THEN "Val"::DOUBLE PRECISION END) AS "DP_C-01" """,
            f"""MAX(CASE WHEN "TagName" = 'PTB-02' THEN "Val"::DOUBLE PRECISION END) - MAX(CASE WHEN "TagName" = 'PTT-02' THEN "Val"::DOUBLE PRECISION END) AS "DP_C-02" """,
            f"""MAX(CASE WHEN "TagName" = 'PTB-03' THEN "Val"::DOUBLE PRECISION END) - MAX(CASE WHEN "TagName" = 'PTT-03' THEN "Val"::DOUBLE PRECISION END) AS "DP_C-03" """,
            f"""MAX(CASE WHEN "TagName" = 'FI-101' THEN "Val"::DOUBLE PRECISION END) * {CP_COOLING_WATER} * (MAX(CASE WHEN "TagName" = 'TI-112' THEN "Val"::DOUBLE PRECISION END) - MAX(CASE WHEN "TagName" = 'TI-110' THEN "Val"::DOUBLE PRECISION END)) AS "Heat_Duty_Condenser_C-00" """,
            f"""MAX(CASE WHEN "TagName" = 'FI-102' THEN "Val"::DOUBLE PRECISION END) * {CP_COOLING_WATER} * (MAX(CASE WHEN "TagName" = 'TI-101' THEN "Val"::DOUBLE PRECISION END) - MAX(CASE WHEN "TagName" = 'TI-110' THEN "Val"::DOUBLE PRECISION END)) AS "Heat_Duty_Condenser_C-01" """,
            f"""MAX(CASE WHEN "TagName" = 'FI-103' THEN "Val"::DOUBLE PRECISION END) * {CP_COOLING_WATER} * (MAX(CASE WHEN "TagName" = 'TI-104' THEN "Val"::DOUBLE PRECISION END) - MAX(CASE WHEN "TagName" = 'TI-110' THEN "Val"::DOUBLE PRECISION END)) AS "Heat_Duty_Condenser_C-02" """,
            f"""MAX(CASE WHEN "TagName" = 'FI-104' THEN "Val"::DOUBLE PRECISION END) * {CP_COOLING_WATER} * (MAX(CASE WHEN "TagName" = 'TI-107' THEN "Val"::DOUBLE PRECISION END) - MAX(CASE WHEN "TagName" = 'TI-110' THEN "Val"::DOUBLE PRECISION END)) AS "Heat_Duty_Condenser_C-03" """,
            f"""MAX(CASE WHEN "TagName" = 'FT-204' THEN "Val"::DOUBLE PRECISION END) * {CP_THERMIC_FLUID_TF66} * (MAX(CASE WHEN "TagName" = 'TI-222' THEN "Val"::DOUBLE PRECISION END) - MAX(CASE WHEN "TagName" = 'TI-221' THEN "Val"::DOUBLE PRECISION END)) AS "Heat_Duty_Reboiler_C-00" """,
            f"""MAX(CASE WHEN "TagName" = 'FT-201' THEN "Val"::DOUBLE PRECISION END) * {CP_THERMIC_FLUID_TF66} * (MAX(CASE WHEN "TagName" = 'TI-204' THEN "Val"::DOUBLE PRECISION END) - MAX(CASE WHEN "TagName" = 'TI-203' THEN "Val"::DOUBLE PRECISION END)) AS "Heat_Duty_Reboiler_C-01" """,
            f"""MAX(CASE WHEN "TagName" = 'FT-202' THEN "Val"::DOUBLE PRECISION END) * {CP_THERMIC_FLUID_TF66} * (MAX(CASE WHEN "TagName" = 'TI-208' THEN "Val"::DOUBLE PRECISION END) - MAX(CASE WHEN "TagName" = 'TI-207' THEN "Val"::DOUBLE PRECISION END)) AS "Heat_Duty_Reboiler_C-02" """,
            f"""MAX(CASE WHEN "TagName" = 'FT-203' THEN "Val"::DOUBLE PRECISION END) * {CP_THERMIC_FLUID_TF66} * (MAX(CASE WHEN "TagName" = 'TI-212' THEN "Val"::DOUBLE PRECISION END) - MAX(CASE WHEN "TagName" = 'TI-211' THEN "Val"::DOUBLE PRECISION END)) AS "Heat_Duty_Reboiler_C-03" """
        ]
        
        all_select_cases = pivot_cases + calculated_cases
        select_cases_str = ",\n".join(all_select_cases)

        insert_data_query = f"""
        INSERT INTO "{PG_TRANSFORMED_TABLE}"
        WITH MappedData AS (
            SELECT
                date_trunc('minute', r."DateAndTime") AS "DateAndTime",
                COALESCE(m."TagName", 'Unmapped') AS "TagName",
                r."Val",
                r."TagIndex"
            FROM
                "{PG_RAW_TABLE}" AS r
            LEFT JOIN
                "{PG_MAPPING_TABLE}" AS m ON r."TagIndex"::INTEGER = m."TagIndex"
            WHERE
                r."DateAndTime" > %s
        )
        SELECT
            "DateAndTime",
            {select_cases_str},
            MAX(CASE WHEN "TagName" = 'Unmapped' THEN "TagIndex"::INTEGER END) AS "Unmapped_TagIndex",
            MAX(CASE WHEN "TagName" = 'Unmapped' THEN "Val"::DOUBLE PRECISION END) AS "Unmapped_Value"
        FROM
            MappedData
        GROUP BY
            "DateAndTime"
        ORDER BY
            "DateAndTime" ASC
        ON CONFLICT ("DateAndTime") DO NOTHING;
        """
        
        pg_cursor.execute(insert_data_query, (start_timestamp_for_this_run,))
        rows_inserted = pg_cursor.rowcount
        pg_conn.commit()
        print(f"✅ Successfully inserted {rows_inserted} new rows into '{PG_TRANSFORMED_TABLE}'.")

    except psycopg2.Error as e:
        print(f"❌ PostgreSQL connection or query failed. Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        if pg_conn:
            pg_conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    while True:
        process_scada_data()
        print("Waiting for 60 seconds before the next run...")
        time.sleep(60)