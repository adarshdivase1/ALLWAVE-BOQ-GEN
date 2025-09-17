import sqlite3
import pandas as pd # Import the pandas library
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATABASE_FILE = 'products.db'
CSV_FILE = 'master_product_catalog.csv'

def ingest_data_from_csv():
    """
    Reads data from a CSV file, removes duplicates, and ingests it into the SQLite database.
    """
    try:
        # --- NEW: Read the CSV using pandas ---
        logging.info(f"Reading data from {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)

        # --- NEW: Automatically remove duplicate rows based on the 'id' column ---
        # This is the key step to fix the repeating products issue.
        initial_rows = len(df)
        df.drop_duplicates(subset=['id'], keep='first', inplace=True)
        final_rows = len(df)
        
        if initial_rows > final_rows:
            logging.warning(f"Removed {initial_rows - final_rows} duplicate products based on the 'id' column.")
        else:
            logging.info("No duplicate IDs found. Data is clean.")

        # --- Connect to the database and ingest the cleaned data ---
        with sqlite3.connect(DATABASE_FILE) as conn:
            logging.info(f"Connecting to database '{DATABASE_FILE}'...")
            
            # Use the DataFrame's to_sql method for easy and safe ingestion
            # 'if_exists='replace'' will drop the old table and create a new one with the clean data.
            df.to_sql('products', conn, if_exists='replace', index=False, dtype={
                'id': 'TEXT PRIMARY KEY',
                'name': 'TEXT NOT NULL',
                'price': 'REAL NOT NULL',
                'brand': 'TEXT',
                'category': 'TEXT',
                'features': 'TEXT',
                'specifications': 'TEXT',
                'warranty_years': 'INTEGER',
                'model_number': 'TEXT',
                'availability': 'TEXT'
            })
            
            logging.info(f"Successfully ingested {final_rows} unique products into the 'products' table.")

    except FileNotFoundError:
        logging.error(f"Error: The file '{CSV_FILE}' was not found.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    ingest_data_from_csv()
