# ingest_data.py (v8.1 - Fixed for Company Sheets with Description and Model columns)

import os
import pandas as pd
import sqlite3
import uuid
import re
import logging
from typing import Dict, List, Any, Optional

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
    'product_name', 'item_name', 'desc', 'product', 'title', 'item'
]
POSSIBLE_MODEL_COLUMNS = [
    'model', 'model_no', 'model_number', 'model_#', 'part_no', 'part_#', 
    'sku', 'part_number', 'item_no', 'item_#', 'code', 'product_code', 'part'
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
        description TEXT,
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
    """Inserts a list of product dictionaries into the database."""
    if not products: 
        logger.warning("No products to insert!")
        return 0
    
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    product_tuples = []
    for p in products:
        product_tuple = (
            p['id'], 
            p.get('category', 'Other'), 
            p.get('brand', 'Unknown'), 
            p['name'],
            p.get('model_number', ''),
            p.get('description', ''),
            0.0, 
            p.get('features', ''), 
            'standard', 
            '', 
            ''
        )
        product_tuples.append(product_tuple)
    
    try:
        cursor.executemany(
            'INSERT INTO products (id, category, brand, name, model_number, description, price, features, tier, use_case_tags, compatibility_tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', 
            product_tuples
        )
        conn.commit()
        logger.info(f"Successfully inserted {len(product_tuples)} products into database.")
    except Exception as e:
        logger.error(f"Bulk insert failed: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()
    return len(product_tuples)

# --- DATA PROCESSING FUNCTIONS ---
def clean_text(text: Any) -> str:
    """Cleans text by removing extra whitespace and handling None/NaN values."""
    if not text or pd.isna(text): 
        return ""
    cleaned = re.sub(r'\s+', ' ', str(text)).strip()
    return cleaned

def categorize_product(description: str) -> str:
    """Assigns a category based on keywords in the product description."""
    if not description: 
        return "Other"
    text_lower = description.lower()
    
    # UC & Collaboration
    if any(k in text_lower for k in ['camera', 'conferencing', 'video bar', 'webcam', 'conference', 'zoom', 'teams']): 
        return "UC & Collaboration Devices"
    
    # Displays & Projectors
    if any(k in text_lower for k in ['display', 'projector', 'screen', 'monitor', 'tv', 'lcd', 'led']): 
        return "Displays & Projectors"
    
    # Audio Systems
    if any(k in text_lower for k in ['speaker', 'microphone', 'audio', 'headset', 'mic', 'sound']): 
        return "Audio Systems"
    
    # Mounts & Hardware
    if any(k in text_lower for k in ['mount', 'rack', 'bracket', 'stand', 'enclosure', 'cabinet']): 
        return "Mounts, Racks & Enclosures"
    
    # Cables & Connectivity
    if any(k in text_lower for k in ['cable', 'hdmi', 'usb', 'connector', 'adapter', 'wire', 'cord']): 
        return "Cables & Connectors"
    
    return "Other"

def find_column_match(df_columns: List[str], possible_columns: List[str], column_type: str) -> Optional[str]:
    """Find the best matching column from a list of possibilities."""
    df_columns_lower = [col.lower() for col in df_columns]
    
    logger.info(f"Looking for {column_type} column in: {df_columns}")
    
    # Try exact matches first
    for possible in possible_columns:
        if possible in df_columns_lower:
            matched_col = df_columns[df_columns_lower.index(possible)]
            logger.info(f"Found exact match for {column_type}: '{matched_col}'")
            return matched_col
    
    # Try partial matches
    for possible in possible_columns:
        for original_col, lower_col in zip(df_columns, df_columns_lower):
            if possible in lower_col or any(part in lower_col for part in possible.split('_')):
                logger.info(f"Found partial match for {column_type}: '{original_col}' (matches '{possible}')")
                return original_col
    
    logger.warning(f"No {column_type} column found. Checked: {possible_columns}")
    return None

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by removing empty rows and columns."""
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Remove completely empty columns
    df = df.dropna(axis=1, how='all')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

def process_sheet(df: pd.DataFrame, company_name: str) -> List[Dict[str, Any]]:
    """Process a single sheet and extract product information."""
    logger.info(f"\n=== Processing sheet: '{company_name}' ===")
    
    if df.empty:
        logger.warning(f"Sheet '{company_name}' is empty. Skipping.")
        return []
    
    # Clean the dataframe
    df = clean_dataframe(df)
    
    logger.info(f"DataFrame shape after cleaning: {df.shape}")
    logger.info(f"Original columns: {list(df.columns)}")
    
    if df.empty:
        logger.warning(f"Sheet '{company_name}' has no data after cleaning. Skipping.")
        return []
    
    # Show sample data
    logger.info("Sample data (first 2 rows):")
    for i in range(min(2, len(df))):
        sample_row = {col: df.iloc[i][col] for col in df.columns[:3]}  # Show first 3 columns
        logger.info(f"  Row {i}: {sample_row}")
    
    # Find the description and model columns
    desc_col = find_column_match(list(df.columns), POSSIBLE_DESCRIPTION_COLUMNS, "Description")
    model_col = find_column_match(list(df.columns), POSSIBLE_MODEL_COLUMNS, "Model")
    
    if not desc_col:
        logger.error(f"❌ CRITICAL: Could not find description column for '{company_name}'")
        logger.error(f"Available columns: {list(df.columns)}")
        return []
    
    if not model_col:
        logger.warning(f"⚠️  WARNING: Could not find model column for '{company_name}' (will continue without model numbers)")
    
    # Process each row
    products = []
    processed_rows = 0
    valid_products = 0
    
    for index, row in df.iterrows():
        processed_rows += 1
        
        # Extract and clean description
        description = clean_text(row[desc_col])
        if not description:
            logger.debug(f"Row {index}: Empty description, skipping")
            continue
        
        # Extract and clean model number
        model_number = ""
        if model_col:
            model_number = clean_text(row[model_col])
        
        # Create product name (combine model + description)
        if model_number and model_number.lower() not in description.lower():
            product_name = f"{model_number} - {description}"
        else:
            product_name = description
        
        # Create the product record
        product = {
            'id': str(uuid.uuid4()),
            'brand': company_name.strip(),
            'name': product_name,
            'description': description,
            'model_number': model_number,
            'category': categorize_product(description),
            'features': description,  # Store original description as features too
        }
        
        products.append(product)
        valid_products += 1
        
        # Log first few products for verification
        if valid_products <= 3:
            logger.info(f"  ✅ Product {valid_products}: '{product_name}' | Model: '{model_number}' | Category: '{product['category']}'")
    
    logger.info(f"✅ Sheet '{company_name}' processed: {processed_rows} rows examined, {valid_products} valid products created")
    return products

def main():
    """Main function to orchestrate the data ingestion process."""
    logger.info("🚀 Starting Excel data ingestion process")
    logger.info("=" * 60)
    
    # Initialize database
    initialize_database()
    
    # Check if data folder exists
    if not os.path.exists(DATA_FOLDER):
        logger.error(f"❌ Data folder '{DATA_FOLDER}' does not exist!")
        return
    
    # Find Excel files
    all_files = os.listdir(DATA_FOLDER)
    excel_files = [f for f in all_files if f.lower().endswith(('.xlsx', '.xls'))]
    
    logger.info(f"📁 Files in {DATA_FOLDER}: {all_files}")
    logger.info(f"📊 Excel files found: {excel_files}")
    
    if not excel_files:
        logger.error(f"❌ FATAL: No Excel files (.xlsx/.xls) found in '{DATA_FOLDER}' directory!")
        return

    # Process the first Excel file found
    excel_file = excel_files[0]
    file_path = os.path.join(DATA_FOLDER, excel_file)
    logger.info(f"📖 Processing Excel file: {excel_file}")

    try:
        # Try to read the Excel file
        excel_data = None
        try:
            excel_data = pd.ExcelFile(file_path, engine='openpyxl')
            logger.info("✅ Successfully opened Excel file with openpyxl")
        except Exception as e1:
            logger.warning(f"⚠️  Failed with openpyxl engine: {e1}")
            try:
                excel_data = pd.ExcelFile(file_path, engine='xlrd')
                logger.info("✅ Successfully opened Excel file with xlrd")
            except Exception as e2:
                logger.error(f"❌ Failed with xlrd engine: {e2}")
                raise Exception(f"Could not read Excel file with any engine. openpyxl error: {e1}, xlrd error: {e2}")
        
        sheet_names = excel_data.sheet_names
        logger.info(f"📋 Found {len(sheet_names)} sheets: {sheet_names}")
        
        # Process each sheet (each sheet = one company)
        all_products = []
        total_sheets_processed = 0
        
        for sheet_name in sheet_names:
            try:
                # Read the sheet data
                df = pd.read_excel(excel_data, sheet_name=sheet_name)
                
                # Use sheet name as company name
                company_name = sheet_name.strip()
                
                # Process this company's products
                company_products = process_sheet(df, company_name)
                all_products.extend(company_products)
                total_sheets_processed += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing sheet '{sheet_name}': {e}")
                continue
        
        # Final summary and database insertion
        logger.info("\n" + "=" * 60)
        logger.info("📊 FINAL SUMMARY:")
        logger.info(f"  • Sheets processed: {total_sheets_processed}/{len(sheet_names)}")
        logger.info(f"  • Total products collected: {len(all_products)}")
        
        if all_products:
            # Group by brand for summary
            brand_counts = {}
            for product in all_products:
                brand = product['brand']
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
            
            logger.info("  • Products per brand:")
            for brand, count in brand_counts.items():
                logger.info(f"    - {brand}: {count} products")
            
            # Insert into database
            inserted_count = insert_products_to_db(all_products)
            logger.info(f"✅ SUCCESS: {inserted_count} products inserted into database!")
        else:
            logger.error("❌ FAILURE: No products were extracted from any sheet!")
            logger.error("Check your Excel file structure and column names.")

    except Exception as e:
        logger.critical(f"❌ CRITICAL ERROR processing {excel_file}: {e}", exc_info=True)

    logger.info("🏁 Data ingestion process completed!")
    logger.info("Check 'ingest_data.log' for detailed logs.")

if __name__ == "__main__":
    main()
