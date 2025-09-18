# ingest_data.py (v5.0 - Direct Excel Multi-Sheet Processing)

import os
import pandas as pd
import sqlite3
import uuid
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

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
BACKUP_FOLDER = 'backups'

# --- DATA MAPPING AND PATTERNS ---
POSSIBLE_NAME_COLUMNS = ['name', 'product_name', 'item', 'description', 'product_description', 'model']
POSSIBLE_PRICE_COLUMNS = ['price', 'msrp', 'list_price', 'dealer_price', 'cost', 'list', 'dealer']

TIER_MAPPING = {
    'standard': (0, 500), 'business': (500, 2000),
    'premium': (2000, 10000), 'enterprise': (10000, float('inf'))
}

# --- DATABASE FUNCTIONS ---
def create_backup():
    if not os.path.exists(DATABASE_FILE): return
    os.makedirs(BACKUP_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_FOLDER, f"products_backup_{timestamp}.db")
    try:
        import shutil
        shutil.copy2(DATABASE_FILE, backup_file)
        logger.info(f"Database backup created: {backup_file}")
    except Exception as e:
        logger.warning(f"Failed to create backup: {str(e)}")

def initialize_database():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS products') # Start fresh each time
    cursor.execute('''
    CREATE TABLE products (
        id TEXT PRIMARY KEY, category TEXT, brand TEXT, name TEXT NOT NULL,
        price REAL NOT NULL, features TEXT, tier TEXT,
        use_case_tags TEXT, compatibility_tags TEXT
    )
    ''')
    indexes = [
        'CREATE INDEX idx_brand ON products(brand)',
        'CREATE INDEX idx_category ON products(category)',
        'CREATE INDEX idx_price ON products(price)',
    ]
    for index_sql in indexes:
        cursor.execute(index_sql)
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully (old data cleared).")

def insert_products_to_db(products: List[Dict[str, Any]]) -> int:
    if not products: return 0
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Using executemany for much faster bulk inserts
    product_tuples = [
        (
            p['id'], p.get('category', 'Other'), p.get('brand', 'Unknown'), p['name'],
            p['price'], p.get('features', ''), p.get('tier', 'standard'),
            p.get('use_case_tags', ''), p.get('compatibility_tags', '')
        )
        for p in products
    ]
    
    try:
        cursor.executemany('''
            INSERT INTO products (id, category, brand, name, price, features, tier, use_case_tags, compatibility_tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', product_tuples)
        conn.commit()
        logger.info(f"Successfully inserted {len(product_tuples)} products into the database.")
    except Exception as e:
        logger.error(f"Bulk insert failed: {e}")
        conn.rollback()
    finally:
        conn.close()
        
    return len(product_tuples)

# --- DATA CLEANING AND EXTRACTION FUNCTIONS ---
def clean_text(text: Any) -> str:
    if not text or pd.isna(text): return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def categorize_product(product_name: str) -> str:
    # This can be expanded with your CATEGORY_MAPPING if needed
    if not product_name: return "Other"
    text_lower = product_name.lower()
    if any(k in text_lower for k in ['camera', 'conferencing', 'video bar']): return "UC & Collaboration Devices"
    if any(k in text_lower for k in ['display', 'projector', 'screen', 'monitor']): return "Displays & Projectors"
    if any(k in text_lower for k in ['speaker', 'microphone', 'audio']): return "Audio Systems"
    if any(k in text_lower for k in ['mount', 'rack']): return "Mounts, Racks & Enclosures"
    if any(k in text_lower for k in ['cable', 'hdmi', 'usb']): return "Cables & Connectors"
    return "Other"

def determine_tier(price: float) -> str:
    for tier_name, (min_price, max_price) in TIER_MAPPING.items():
        if min_price <= price < max_price: return tier_name
    return "standard"

def validate_price(price_str: Any) -> float:
    if not price_str or pd.isna(price_str): return 0.0
    price_clean = re.sub(r'[^\d\.]', '', str(price_str))
    try:
        return float(price_clean) if float(price_clean) > 0 else 0.0
    except (ValueError, TypeError):
        return 0.0

# --- CORE PROCESSING FUNCTION ---
def process_dataframe(df: pd.DataFrame, brand_hint: str) -> List[Dict[str, Any]]:
    """Takes a DataFrame and a brand hint, returns a list of cleaned product dicts."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    name_col = next((col for col in df.columns if col in POSSIBLE_NAME_COLUMNS), None)
    price_col = next((col for col in df.columns if col in POSSIBLE_PRICE_COLUMNS), None)

    if not name_col or not price_col:
        logger.warning(f"Skipping sheet for '{brand_hint}': Could not find required name/price columns.")
        return []

    logger.info(f"Processing sheet for '{brand_hint}' -> Name: '{name_col}', Price: '{price_col}'")
    
    products = []
    for index, row in df.iterrows():
        name = clean_text(row.get(name_col))
        price = validate_price(row.get(price_col))

        if not name or price == 0.0:
            continue

        products.append({
            'id': str(uuid.uuid4()),
            'category': categorize_product(name),
            'brand': brand_hint,
            'name': name,
            'price': price,
            'features': clean_text(row.get('features')),
            'tier': determine_tier(price),
            'use_case_tags': '', 'compatibility_tags': ''
        })
            
    return products

def main():
    logger.info("Starting Excel data ingestion process")
    create_backup()
    initialize_database()
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    excel_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.xlsx')]

    if not excel_files:
        logger.error(f"No Excel (.xlsx) files found in the '{DATA_FOLDER}' directory. Aborting.")
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
            brand_name = sheet_name.strip() # Use the sheet name as the brand
            products_from_sheet = process_dataframe(df, brand_name)
            all_products.extend(products_from_sheet)
        
        if all_products:
            insert_products_to_db(all_products)
        else:
            logger.warning("No products were extracted from the Excel file.")

    except Exception as e:
        logger.critical(f"Failed to process Excel file {master_list_file}: {e}", exc_info=True)

    logger.info("Data ingestion from Excel file completed.")

if __name__ == "__main__":
    main()
