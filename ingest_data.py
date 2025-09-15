# ingest_data.py

import os
import pandas as pd
import sqlite3
import uuid
import re

# --- IMPORTANT: Place all your data files (CSVs, Excels) in a folder named 'data' ---
DATA_FOLDER = 'data'
DATABASE_FILE = 'products.db'

def normalize_product_data(data: dict) -> dict or None:
    """This is a simplified version of the function from your main.py"""
    product = {}
    found_name = False
    found_price = False
    
    item = {str(k).lower(): v for k, v in data.items()}

    name_keys = ['name', 'product', 'item', 'description', 'title']
    price_keys = ['price', 'cost', 'amount', 'rate', 'value']
    brand_keys = ['brand', 'manufacturer', 'make']
    category_keys = ['category', 'type', 'group', 'class']

    for key in name_keys:
        if key in item and item[key]:
            product['name'] = str(item[key]).strip()
            found_name = True
            break
            
    for key in price_keys:
        if key in item and item[key]:
            try:
                price_str = re.sub(r'[^\d\.]', '', str(item[key]))
                price_val = float(price_str)
                if price_val > 0:
                    product['price'] = price_val
                    found_price = True
                    break
            except (ValueError, TypeError):
                continue
    
    if not (found_name and found_price):
        return None

    # Set defaults
    product['brand'] = 'Generic'
    product['category'] = 'Miscellaneous'

    for key in brand_keys:
        if key in item and item[key]:
            product['brand'] = str(item[key]).strip()
            break
    
    for key in category_keys:
        if key in item and item[key]:
            product['category'] = str(item[key]).strip()
            break

    product['id'] = str(uuid.uuid4())
    return product

def main():
    # Connect to SQLite database (this will create the file if it doesn't exist)
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Create the products table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        price REAL NOT NULL,
        brand TEXT,
        category TEXT
    )
    ''')
    conn.commit()

    print(f"Database '{DATABASE_FILE}' initialized.")
    
    products_to_add = []
    
    # Read files from the data folder
    for filename in os.listdir(DATA_FOLDER):
        filepath = os.path.join(DATA_FOLDER, filename)
        print(f"Processing {filename}...")
        
        df = None
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        
        if df is not None:
            for record in df.to_dict('records'):
                normalized = normalize_product_data(record)
                if normalized:
                    products_to_add.append(
                        (normalized['id'], normalized['name'], normalized['price'], normalized['brand'], normalized['category'])
                    )

    # Insert all products into the database
    if products_to_add:
        cursor.executemany('''
        INSERT INTO products (id, name, price, brand, category) VALUES (?, ?, ?, ?, ?)
        ''', products_to_add)
        conn.commit()
        print(f"Successfully added {len(products_to_add)} products to the database.")
    else:
        print("No valid products found to add.")
        
    conn.close()

if __name__ == '__main__':
    main()
