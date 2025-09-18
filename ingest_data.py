# ingest_data.py (v7.0 - Reads Model No. and Description)

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

# NEW: Specific columns to search for, based on your data structure
POSSIBLE_DESCRIPTION_COLUMNS = ['description', 'product_description', 'item description', 'name']
POSSIBLE_MODEL_COLUMNS = ['model', 'model_no', 'model #', 'part_no', 'part #', 'sku', 'part_number']

# --- DATABASE FUNCTIONS ---
def initialize_database():
    """Initializes a fresh, empty database with the correct schema."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS products')
    logger.info("Cleared old products table.")
    cursor.execute('''
    CREATE TABLE products (
        id TEXT PRIMARY KEY, category TEXT, brand TEXT, name TEXT NOT NULL,
        price REAL DEFAULT 0, features TEXT, tier TEXT,
        use_case_tags TEXT, compatibility_tags TEXT
    )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

def insert_products_to_db(products: List[Dict[str, Any]]) -> int:
    """Inserts a list of product dictionaries into the database using a fast bulk method."""
    if not products: return 0
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    product_tuples = [
        (
            p['id'], p.get('category', 'Other'), p.get('brand', 'Unknown'), p['name'],
            0.0, p.get('features', ''), 'standard', '', ''
        ) for p in products
    ]
    
    try:
        cursor.executemany('INSERT INTO products (id, category, brand, name, price, features, tier, use_case_tags, compatibility_tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', product_tuples)
        conn.commit()
        logger.info(f"Successfully inserted {len(product_tuples)} products.")
    except Exception as e:
        logger.error(f"Bulk insert failed: {e}")
        conn.rollback()
    finally:
        conn.close()
    return len(product_tuples)

# --- DATA PROCESSING FUNCTIONS ---
def clean_text(text: Any) -> str:
    """Cleans text by removing extra whitespace."""
    if not text or pd.isna(text): return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def categorize_product(description: str) -> str:
    """Assigns a category based on keywords in the product description."""
    if not description: return "Other"
    text_lower = description.lower()
    if any(k in text_lower for k in ['camera', 'conferencing', 'video bar']): return "UC & Collaboration Devices"
    if any(k in text_lower for k in ['display', 'projector', 'screen', 'monitor']): return "Displays & Projectors"
    if any(k in text_lower for k in ['speaker', 'microphone', 'audio']): return "Audio Systems"
    if any(k in text_lower for k in ['mount', 'rack']): return "Mounts, Racks & Enclosures"
    if any(k in text_lower for k in ['cable', 'hdmi', 'usb', 'connector']): return "Cables & Connectors"
    return "Other"

def process_dataframe(df: pd.DataFrame, brand_name: str) -> List[Dict[str, Any]]:
    """Takes a DataFrame and a brand name, returns a list of cleaned product dicts."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').replace('#', '_no')

    # NEW: Smartly find the description and model number columns
    desc_col = next((col for col in df.columns if col in POSSIBLE_DESCRIPTION_COLUMNS), None)
    model_col = next((col for col in df.columns if col in POSSIBLE_MODEL_COLUMNS), None)

    if not desc_col:
        logger.warning(f"Skipping sheet for '{brand_name}': Could not find a required 'Description' column.")
        return []
    
    logger.info(f"Processing sheet for '{brand_name}' -> Using Description: '{desc_col}', Model: '{model_col}'")
    
    products = []
    for index, row in df.iterrows():
        description = clean_text(row.get(desc_col))
        model_no = clean_text(row.get(model_col)) if model_col else ""

        if not description:
            continue

        # NEW: Combine model number and description for the final product name
        final_name = description
        if model_no and model_no.lower() not in description.lower():
            final_name = f"{model_no} - {description}"

        products.append({
            'id': str(uuid.uuid4()),
            'category': categorize_product(description),
            'brand': brand_name,
            'name': final_name,
            'features': description, # Store the original description in the 'features' column
        })
            
    return products

def main():
    """Main function to orchestrate the data ingestion process."""
    logger.info("Starting Excel data ingestion process (Model No. & Description Mode)")
    initialize_database()
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    excel_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.xlsx')]
    if not excel_files:
        logger.error(f"FATAL: No Excel (.xlsx) files found in the '{DATA_FOLDER}' directory. Aborting.")
        return

    master_list_file = excel_files[0]
    file_path = os.path.join(DATA_FOLDER, master_list_file)
    logger.info(f"Found Master List: {master_list_file}")

    try:
        xls = pd.ExcelFile(file_path, engine='openpyxl')
        all_products = []
        for sheet_name in xls.sheet_names:
            logger.info(f"--- Reading sheet: {sheet_name} ---")
            df = pd.read_excel(xls, sheet_name=sheet_name)
            # Use the sheet name as the definitive brand name
            products_from_sheet = process_dataframe(df, sheet_name.strip())
            all_products.extend(products_from_sheet)
        
        if all_products:
            insert_products_to_db(all_products)
        else:
            logger.warning("No valid products were extracted from the Excel file.")

    except Exception as e:
        logger.critical(f"Failed to process Excel file {master_list_file}: {e}", exc_info=True)

    logger.info("Data ingestion from Excel file completed.")

if __name__ == "__main__":
    main()
