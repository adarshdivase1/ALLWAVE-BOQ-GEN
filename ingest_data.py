# ingest_data.py (v6.0 - No Price Required)

import os
import pandas as pd
import sqlite3
import uuid
import re
import logging
from typing import Dict, List, Any
from datetime import datetime

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
POSSIBLE_NAME_COLUMNS = ['name', 'product_name', 'item', 'description', 'product_description', 'model']

# --- DATABASE FUNCTIONS ---
def initialize_database():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS products')
    logger.info("Cleared old products table.")
    # Create table WITHOUT price being required
    cursor.execute('''
    CREATE TABLE products (
        id TEXT PRIMARY KEY, category TEXT, brand TEXT, name TEXT NOT NULL,
        price REAL DEFAULT 0, features TEXT, tier TEXT,
        use_case_tags TEXT, compatibility_tags TEXT
    )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully for price-less data.")

def insert_products_to_db(products: List[Dict[str, Any]]) -> int:
    if not products: return 0
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    product_tuples = [
        (
            p['id'], p.get('category', 'Other'), p.get('brand', 'Unknown'), p['name'],
            0.0, p.get('features', ''), p.get('tier', 'standard'), '', ''
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
    if not text or pd.isna(text): return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def categorize_product(product_name: str) -> str:
    if not product_name: return "Other"
    text_lower = product_name.lower()
    if any(k in text_lower for k in ['camera', 'conferencing', 'video bar']): return "UC & Collaboration Devices"
    if any(k in text_lower for k in ['display', 'projector', 'screen', 'monitor']): return "Displays & Projectors"
    if any(k in text_lower for k in ['speaker', 'microphone', 'audio']): return "Audio Systems"
    if any(k in text_lower for k in ['mount', 'rack']): return "Mounts, Racks & Enclosures"
    if any(k in text_lower for k in ['cable', 'hdmi', 'usb']): return "Cables & Connectors"
    return "Other"

def process_dataframe(df: pd.DataFrame, brand_hint: str) -> List[Dict[str, Any]]:
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    name_col = next((col for col in df.columns if col in POSSIBLE_NAME_COLUMNS), None)

    if not name_col:
        logger.warning(f"Skipping sheet for '{brand_hint}': Could not find a product name column.")
        return []

    logger.info(f"Processing sheet for '{brand_hint}' using column '{name_col}' for product name.")
    
    products = []
    for index, row in df.iterrows():
        name = clean_text(row.get(name_col))
        
        # *** THE KEY CHANGE IS HERE: We only require a name to proceed ***
        if not name:
            continue

        products.append({
            'id': str(uuid.uuid4()),
            'category': categorize_product(name),
            'brand': brand_hint,
            'name': name,
            'features': clean_text(row.get('features')),
            'tier': 'standard',
        })
            
    return products

def main():
    logger.info("Starting Excel data ingestion process (No Price Mode)")
    initialize_database()
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    excel_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.xlsx')]
    if not excel_files:
        logger.error(f"No Excel (.xlsx) files found in the '{DATA_FOLDER}' directory. Aborting.")
        return

    file_path = os.path.join(DATA_FOLDER, excel_files[0])
    logger.info(f"Found Master List: {excel_files[0]}")

    try:
        xls = pd.ExcelFile(file_path, engine='openpyxl')
        all_products = []
        for sheet_name in xls.sheet_names:
            logger.info(f"--- Reading sheet: {sheet_name} ---")
            df = pd.read_excel(xls, sheet_name=sheet_name)
            products_from_sheet = process_dataframe(df, sheet_name.strip())
            all_products.extend(products_from_sheet)
        
        if all_products:
            insert_products_to_db(all_products)
        else:
            logger.warning("No products were extracted from the Excel file.")
    except Exception as e:
        logger.critical(f"Failed to process Excel file: {e}", exc_info=True)

    logger.info("Data ingestion from Excel file completed.")

if __name__ == "__main__":
    main()
