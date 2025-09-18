# ingest_data.py (v8.0 - Enhanced with debugging and better column detection)

import os
import pandas as pd
import sqlite3
import uuid
import re
import logging
from typing import Dict, List, Any

# --- CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingest_data.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATA_FOLDER = 'data'
DATABASE_FILE = 'products.db'

# Enhanced column matching - more flexible patterns
POSSIBLE_DESCRIPTION_COLUMNS = [
    'description', 'product_description', 'item_description', 'name', 
    'product_name', 'item_name', 'desc', 'product', 'title'
]
POSSIBLE_MODEL_COLUMNS = [
    'model', 'model_no', 'model_number', 'model_#', 'part_no', 'part_#', 
    'sku', 'part_number', 'item_no', 'item_#', 'code', 'product_code'
]

# --- DATABASE FUNCTIONS ---
def initialize_database():
    """Initializes a fresh, empty database with the correct schema."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS products')
    logger.info("Cleared old products table.")
    cursor.execute('''
    CREATE TABLE products (
        id TEXT PRIMARY KEY, 
        category TEXT, 
        brand TEXT, 
        name TEXT NOT NULL,
        model_number TEXT,
        price REAL DEFAULT 0, 
        features TEXT, 
        tier TEXT,
        use_case_tags TEXT, 
        compatibility_tags TEXT
    )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

def insert_products_to_db(products: List[Dict[str, Any]]) -> int:
    """Inserts a list of product dictionaries into the database using a fast bulk method."""
    if not products: 
        logger.warning("No products to insert!")
        return 0
    
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    product_tuples = [
        (
            p['id'], p.get('category', 'Other'), p.get('brand', 'Unknown'), p['name'],
            p.get('model_number', ''), 0.0, p.get('features', ''), 'standard', '', ''
        ) for p in products
    ]
    
    try:
        cursor.executemany(
            'INSERT INTO products (id, category, brand, name, model_number, price, features, tier, use_case_tags, compatibility_tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', 
            product_tuples
        )
        conn.commit()
        logger.info(f"Successfully inserted {len(product_tuples)} products.")
    except Exception as e:
        logger.error(f"Bulk insert failed: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()
    return len(product_tuples)

# --- DATA PROCESSING FUNCTIONS ---
def clean_text(text: Any) -> str:
    """Cleans text by removing extra whitespace."""
    if not text or pd.isna(text): 
        return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def categorize_product(description: str) -> str:
    """Assigns a category based on keywords in the product description."""
    if not description: 
        return "Other"
    text_lower = description.lower()
    if any(k in text_lower for k in ['camera', 'conferencing', 'video bar', 'webcam']): 
        return "UC & Collaboration Devices"
    if any(k in text_lower for k in ['display', 'projector', 'screen', 'monitor', 'tv']): 
        return "Displays & Projectors"
    if any(k in text_lower for k in ['speaker', 'microphone', 'audio', 'headset']): 
        return "Audio Systems"
    if any(k in text_lower for k in ['mount', 'rack', 'bracket', 'stand']): 
        return "Mounts, Racks & Enclosures"
    if any(k in text_lower for k in ['cable', 'hdmi', 'usb', 'connector', 'adapter']): 
        return "Cables & Connectors"
    return "Other"

def find_best_column_match(columns: List[str], possible_columns: List[str]) -> str:
    """Find the best matching column from a list of possibilities."""
    # First, try exact matches
    for possible in possible_columns:
        if possible in columns:
            return possible
    
    # Then try partial matches
    for possible in possible_columns:
        for col in columns:
            if possible in col or col in possible:
                return col
    
    return None

def debug_dataframe_info(df: pd.DataFrame, sheet_name: str):
    """Print detailed debugging information about the DataFrame."""
    logger.info(f"=== DEBUG INFO for sheet '{sheet_name}' ===")
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Column names (original): {list(df.columns)}")
    
    # Show first few rows
    logger.info("First 3 rows of data:")
    for i, row in df.head(3).iterrows():
        logger.info(f"Row {i}: {dict(row)}")
    
    # Check for empty DataFrame
    if df.empty:
        logger.warning(f"Sheet '{sheet_name}' is empty!")
        return
    
    # Check for columns with data
    non_empty_cols = []
    for col in df.columns:
        non_empty_count = df[col].notna().sum()
        if non_empty_count > 0:
            non_empty_cols.append(f"{col}({non_empty_count})")
    
    logger.info(f"Columns with data: {non_empty_cols}")

def process_dataframe(df: pd.DataFrame, brand_name: str) -> List[Dict[str, Any]]:
    """Takes a DataFrame and a brand name, returns a list of cleaned product dicts."""
    if df.empty:
        logger.warning(f"Skipping empty sheet for '{brand_name}'")
        return []
    
    # Debug the dataframe first
    debug_dataframe_info(df, brand_name)
    
    # Clean column names
    original_columns = list(df.columns)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('#', '_no').str.replace('.', '')
    cleaned_columns = list(df.columns)
    
    logger.info(f"Column mapping: {dict(zip(original_columns, cleaned_columns))}")

    # Find the best matching columns
    desc_col = find_best_column_match(cleaned_columns, POSSIBLE_DESCRIPTION_COLUMNS)
    model_col = find_best_column_match(cleaned_columns, POSSIBLE_MODEL_COLUMNS)

    logger.info(f"Found description column: '{desc_col}'")
    logger.info(f"Found model column: '{model_col}'")

    if not desc_col:
        logger.error(f"Could not find description column for '{brand_name}'")
        logger.error(f"Available columns: {cleaned_columns}")
        logger.error(f"Looking for: {POSSIBLE_DESCRIPTION_COLUMNS}")
        return []
    
    products = []
    valid_rows = 0
    
    for index, row in df.iterrows():
        description = clean_text(row.get(desc_col))
        model_no = clean_text(row.get(model_col)) if model_col else ""

        # Skip empty descriptions
        if not description:
            continue
        
        valid_rows += 1

        # Combine model number and description for the final product name
        final_name = description
        if model_no and model_no.lower() not in description.lower():
            final_name = f"{model_no} - {description}"

        product = {
            'id': str(uuid.uuid4()),
            'category': categorize_product(description),
            'brand': brand_name,
            'name': final_name,
            'model_number': model_no,
            'features': description,
        }
        
        products.append(product)
    
    logger.info(f"Processed {valid_rows} valid rows from '{brand_name}', created {len(products)} products")
    return products

def main():
    """Main function to orchestrate the data ingestion process."""
    logger.info("Starting Excel data ingestion process (Enhanced Debug Mode)")
    initialize_database()
    
    # Check if data folder exists
    if not os.path.exists(DATA_FOLDER):
        logger.error(f"Data folder '{DATA_FOLDER}' does not exist!")
        return
    
    # Find Excel files
    all_files = os.listdir(DATA_FOLDER)
    excel_files = [f for f in all_files if f.lower().endswith(('.xlsx', '.xls'))]
    
    logger.info(f"Files in {DATA_FOLDER}: {all_files}")
    logger.info(f"Excel files found: {excel_files}")
    
    if not excel_files:
        logger.error(f"FATAL: No Excel files found in the '{DATA_FOLDER}' directory. Aborting.")
        return

    master_list_file = excel_files[0]
    file_path = os.path.join(DATA_FOLDER, master_list_file)
    logger.info(f"Processing file: {master_list_file}")

    try:
        # Try different engines for reading Excel
        xls = None
        try:
            xls = pd.ExcelFile(file_path, engine='openpyxl')
        except Exception as e1:
            logger.warning(f"Failed with openpyxl: {e1}")
            try:
                xls = pd.ExcelFile(file_path, engine='xlrd')
            except Exception as e2:
                logger.error(f"Failed with xlrd: {e2}")
                raise e1
        
        logger.info(f"Sheet names found: {xls.sheet_names}")
        
        all_products = []
        for sheet_name in xls.sheet_names:
            logger.info(f"\n--- Processing sheet: '{sheet_name}' ---")
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                products_from_sheet = process_dataframe(df, sheet_name.strip())
                all_products.extend(products_from_sheet)
                logger.info(f"Added {len(products_from_sheet)} products from '{sheet_name}'")
            except Exception as e:
                logger.error(f"Error processing sheet '{sheet_name}': {e}")
                continue
        
        logger.info(f"\nTotal products collected: {len(all_products)}")
        
        if all_products:
            inserted_count = insert_products_to_db(all_products)
            logger.info(f"Successfully processed and inserted {inserted_count} products")
        else:
            logger.warning("No valid products were extracted from the Excel file.")

    except Exception as e:
        logger.critical(f"Failed to process Excel file {master_list_file}: {e}", exc_info=True)

    logger.info("Data ingestion process completed.")

if __name__ == "__main__":
    main()
