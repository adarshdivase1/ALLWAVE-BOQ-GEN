# ingest_data.py (Production-Ready v4.0 - With Smart Column Detection)

import os
import pandas as pd
import sqlite3
import uuid
import re
import io
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingest_data.log', mode='w'), # Overwrite log each run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATA_FOLDER = 'data'
DATABASE_FILE = 'products.db'
BACKUP_FOLDER = 'backups'

# --- DATA MAPPING AND PATTERNS ---

# NEW: Flexible column name mapping to find the right data in different CSVs
POSSIBLE_NAME_COLUMNS = ['name', 'product_name', 'item', 'description', 'product_description', 'model']
POSSIBLE_PRICE_COLUMNS = ['price', 'msrp', 'list_price', 'dealer_price', 'cost', 'list']

# Enhanced product categorization mapping
CATEGORY_MAPPING = {
    'video conferencing': 'UC & Collaboration Devices', 'vc': 'UC & Collaboration Devices',
    'collaboration': 'UC & Collaboration Devices', 'camera': 'UC & Collaboration Devices',
    'microphone': 'UC & Collaboration Devices', 'speaker': 'Audio Systems',
    'display': 'Displays & Projectors', 'monitor': 'Displays & Projectors', 'projector': 'Displays & Projectors',
    'screen': 'Displays & Projectors', 'mount': 'Mounts, Racks & Enclosures', 'bracket': 'Mounts, Racks & Enclosures',
    'rack': 'Mounts, Racks & Enclosures', 'cable': 'Cables & Connectors', 'hdmi': 'Cables & Connectors',
    'usb': 'Cables & Connectors', 'network': 'Networking Equipment', 'switch': 'Networking Equipment',
    'router': 'Networking Equipment', 'power': 'Power & Connectivity', 'ups': 'Power & Connectivity',
    'pdu': 'Power & Connectivity', 'lighting': 'Lighting & Control', 'control': 'Lighting & Control',
    'automation': 'Lighting & Control', 'audio': 'Audio Systems', 'amplifier': 'Audio Systems', 'mixer': 'Audio Systems'
}

# Brand recognition patterns
BRAND_PATTERNS = {
    'logitech': ['logitech', 'logi'], 'poly': ['poly', 'polycom'], 'cisco': ['cisco', 'webex'],
    'microsoft': ['microsoft', 'teams', 'surface'], 'samsung': ['samsung'], 'lg': ['lg'], 'sony': ['sony'],
    'epson': ['epson'], 'barco': ['barco'], 'crestron': ['crestron'], 'extron': ['extron'],
    'amx': ['amx', 'harman'], 'shure': ['shure'], 'sennheiser': ['sennheiser'], 'bose': ['bose'],
    'qsc': ['qsc'], 'kramer': ['kramer'], 'jbl': ['jbl'], 'biamp': ['biamp'], 'apple': ['apple'],
    'yealink': ['yealink'], 'chief': ['chief'], 'dten': ['dten'], 'aten': ['aten'], 'belden': ['belden'],
    'yamaha': ['yamaha'], 'neat': ['neat'], 'jabra': ['jabra'], 'huddly': ['huddly']
}

TIER_MAPPING = {
    'standard': (0, 500), 'business': (500, 2000),
    'premium': (2000, 10000), 'enterprise': (10000, float('inf'))
}

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
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id TEXT PRIMARY KEY, category TEXT, brand TEXT, name TEXT NOT NULL,
        price REAL NOT NULL, features TEXT, tier TEXT,
        use_case_tags TEXT, compatibility_tags TEXT
    )
    ''')
    indexes = [
        'CREATE INDEX IF NOT EXISTS idx_brand ON products(brand)',
        'CREATE INDEX IF NOT EXISTS idx_category ON products(category)',
        'CREATE INDEX IF NOT EXISTS idx_price ON products(price)',
    ]
    for index_sql in indexes:
        cursor.execute(index_sql)
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def clean_text(text: Any) -> str:
    if not text or pd.isna(text): return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_brand_from_filename(filename: str) -> Optional[str]:
    """NEW: Extracts a potential brand name from the filename."""
    stem = os.path.splitext(filename)[0].lower()
    for brand, patterns in BRAND_PATTERNS.items():
        if any(pattern in stem for pattern in patterns):
            return brand.title()
    return None

def extract_brand(product_name: str, brand_hint: Optional[str] = None) -> str:
    if brand_hint: return brand_hint
    text_lower = product_name.lower()
    for brand, patterns in BRAND_PATTERNS.items():
        if any(pattern in text_lower for pattern in patterns):
            return brand.title()
    return "Unknown"

def categorize_product(product_name: str) -> str:
    text_lower = product_name.lower()
    for keyword, category in CATEGORY_MAPPING.items():
        if keyword in text_lower:
            return category
    return "Other"

def determine_tier(price: float) -> str:
    for tier_name, (min_price, max_price) in TIER_MAPPING.items():
        if min_price <= price < max_price:
            return tier_name
    return "standard"

def validate_price(price_str: Any) -> float:
    if not price_str or pd.isna(price_str): return 0.0
    price_clean = re.sub(r'[^\d\.]', '', str(price_str))
    try:
        price = float(price_clean)
        return price if price > 0 else 0.0
    except (ValueError, TypeError):
        return 0.0

def process_csv_file(file_path: str) -> List[Dict[str, Any]]:
    logger.info(f"Processing CSV file: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', engine='python')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip', engine='python')
    except Exception as e:
        logger.error(f"Could not read CSV file {os.path.basename(file_path)}: {e}")
        return []

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # --- Smart Column Detection ---
    name_col = next((col for col in df.columns if col in POSSIBLE_NAME_COLUMNS), None)
    price_col = next((col for col in df.columns if col in POSSIBLE_PRICE_COLUMNS), None)

    if not name_col or not price_col:
        logger.warning(f"Skipping {os.path.basename(file_path)}: Could not find required 'name' (found: {name_col}) or 'price' (found: {price_col}) columns.")
        return []
    
    logger.info(f"Identified columns in {os.path.basename(file_path)} -> Name: '{name_col}', Price: '{price_col}'")
    
    products = []
    brand_from_file = extract_brand_from_filename(os.path.basename(file_path))

    for index, row in df.iterrows():
        try:
            name = clean_text(row.get(name_col))
            price = validate_price(row.get(price_col))

            if not name or price == 0.0:
                continue

            brand = clean_text(row.get('brand')) or extract_brand(name, brand_from_file)
            category = clean_text(row.get('category')) or categorize_product(name)
            features = clean_text(row.get('features'))
            tier = clean_text(row.get('tier')) or determine_tier(price)

            products.append({
                'id': str(uuid.uuid4()), 'category': category, 'brand': brand, 'name': name,
                'price': price, 'features': features, 'tier': tier,
                'use_case_tags': '', 'compatibility_tags': '' # Can be enhanced later
            })
        except Exception as e:
            logger.error(f"Error processing row {index + 1} in {os.path.basename(file_path)}: {e}")
            continue
            
    logger.info(f"Successfully processed {len(products)} products from {os.path.basename(file_path)}")
    return products

def insert_products_to_db(products: List[Dict[str, Any]]) -> Tuple[int, int]:
    if not products: return 0, 0
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    inserted_count, updated_count = 0, 0
    
    for product in products:
        try:
            cursor.execute('SELECT id FROM products WHERE name = ? AND brand = ?', (product['name'], product['brand']))
            existing = cursor.fetchone()
            if existing:
                cursor.execute('''
                UPDATE products SET category = ?, price = ?, features = ?, tier = ?
                WHERE id = ?
                ''', (product['category'], product['price'], product['features'], product['tier'], existing[0]))
                updated_count += 1
            else:
                cursor.execute('''
                INSERT INTO products (id, category, brand, name, price, features, tier, use_case_tags, compatibility_tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    product['id'], product['category'], product['brand'], product['name'],
                    product['price'], product['features'], product['tier'],
                    product['use_case_tags'], product['compatibility_tags']
                ))
                inserted_count += 1
        except Exception as e:
            logger.error(f"Error inserting/updating product {product.get('name', 'Unknown')}: {e}")
            continue
    conn.commit()
    conn.close()
    return inserted_count, updated_count

def main():
    logger.info("Starting AV product data ingestion process")
    create_backup()
    initialize_database()
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    total_inserted, total_updated = 0, 0
    
    files_to_process = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.csv')]
    logger.info(f"Found {len(files_to_process)} CSV files to process.")

    for filename in files_to_process:
        file_path = os.path.join(DATA_FOLDER, filename)
        products = process_csv_file(file_path)
        if products:
            inserted, updated = insert_products_to_db(products)
            total_inserted += inserted
            total_updated += updated
    
    logger.info(f"Data ingestion completed. Total Inserted: {total_inserted}, Total Updated: {total_updated}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
