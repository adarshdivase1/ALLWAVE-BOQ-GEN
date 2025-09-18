# ingest_data.py
import os
import pandas as pd
import sqlite3
import uuid
import re
import io
import pdfplumber
import spacy
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

# (Rest of the script is largely the same as it was well-structured)
# ...
# Key changes are in initialize_database() and process_csv_file()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# (BRAND_PATTERNS, TIER_MAPPING, etc. remain the same)
# ...

DATABASE_FILE = 'products.db'

def initialize_database():
    """Initialize database matching your existing schema, ensuring 'model' column exists."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(products)")
    existing_columns = [column[1] for column in cursor.fetchall()]

    if 'model' not in existing_columns:
        try:
            # Create table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY, category TEXT, brand TEXT, name TEXT NOT NULL,
                price REAL NOT NULL, features TEXT, tier TEXT, use_case_tags TEXT,
                compatibility_tags TEXT
            )''')
            # Add the model column
            cursor.execute('ALTER TABLE products ADD COLUMN model TEXT')
            logger.info("Added 'model' column to the products table.")
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not add column 'model': {e}")
    else:
        logger.info("'model' column already exists.")

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def extract_model(product_name: str) -> Optional[str]:
    """Extract a potential model number from the product name."""
    # Look for patterns like 'X50', 'PTZ-123', 'Rally Bar'
    match = re.search(r'\b([A-Z0-9]{2,}[-][A-Z0-9]{2,}|[A-Z]{2,}[0-9]{2,}|[A-Za-z]+[ -]?[Bb]ar|[A-Za-z]+[ -]?[Kk]it)\b', product_name)
    return match.group(0) if match else None

# In process_csv_file() and parse_pdf_catalog(), add the model extraction
def process_csv_file(file_path: str) -> List[Dict[str, Any]]:
    # ... inside the for loop after processing other fields ...
    # name = clean_text(row.get('name', ''))
    # ...
    
    # Example addition inside the loop:
    # product_data = { ... }
    # product_data['model'] = extract_model(name)
    # products.append(product_data)
    # ...
    # For brevity, the full function is omitted but this logic should be inserted.
    # The provided `ingest_data.py` was robust, so only this addition is critically needed.
    pass # Placeholder to indicate the rest of the file is unchanged.

# The rest of `ingest_data.py` can remain as it is. It's a solid script.
