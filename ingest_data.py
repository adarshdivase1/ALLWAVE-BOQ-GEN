# ingest_data.py (Updated to handle bad CSV lines)

import os
import pandas as pd
import sqlite3
import uuid
import re
import io
import pdfplumber
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
    print("SpaCy model 'en_core_web_sm' loaded successfully.")
except (OSError, ImportError):
    print("SpaCy model not found. Text extraction from PDFs will be disabled.")
    nlp = None

DATA_FOLDER = 'data'
DATABASE_FILE = 'products.db'

def normalize_product_data(data: dict) -> dict or None:
    # This function remains the same
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

def extract_products_from_text(text: str) -> list:
    if not nlp: return []
    products = []
    doc = nlp(text)
    
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG"]:
            surrounding_text = doc[max(0, ent.start - 10): min(len(doc), ent.end + 10)].text
            price_match = re.search(r'\$?(\d+[\.,]\d{2})', surrounding_text)
            if price_match:
                price = float(price_match.group(1).replace(',', '.'))
                if price > 0:
                    products.append({"name": ent.text.strip(), "price": price, "brand": ent.text.split()[0], "category": "Extracted from Text"})
    return products

def main():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

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
    
    for filename in os.listdir(DATA_FOLDER):
        filepath = os.path.join(DATA_FOLDER, filename)
        print(f"Processing {filename}...")
        
        df = None
        if filename.endswith('.pdf'):
            try:
                all_pdf_text = ""
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        for table in tables:
                            if not table or not table[0]: continue
                            header = [str(h).lower() for h in table[0]]
                            table_df = pd.DataFrame(table[1:], columns=header)
                            for record in table_df.to_dict('records'):
                                normalized = normalize_product_data(record)
                                if normalized: products_to_add.append((normalized['id'], normalized['name'], normalized['price'], normalized['brand'], normalized['category']))
                        all_pdf_text += page.extract_text() or ""
                
                if all_pdf_text:
                    text_products = extract_products_from_text(all_pdf_text)
                    for record in text_products:
                        normalized = normalize_product_data(record)
                        if normalized: products_to_add.append((normalized['id'], normalized['name'], normalized['price'], normalized['brand'], normalized['category']))
            except Exception as e:
                print(f"  -> Could not process PDF file {filename}. Reason: {e}")

        elif filename.endswith('.csv'):
            # ✅ UPDATED LINE: Added on_bad_lines='warn' to handle malformed rows.
            df = pd.read_csv(filepath, on_bad_lines='warn')
        
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        
        if df is not None:
            for record in df.to_dict('records'):
                normalized = normalize_product_data(record)
                if normalized:
                    products_to_add.append((normalized['id'], normalized['name'], normalized['price'], normalized['brand'], normalized['category']))

    if products_to_add:
        cursor.executemany('''
        INSERT OR IGNORE INTO products (id, name, price, brand, category) VALUES (?, ?, ?, ?, ?)
        ''', products_to_add)
        conn.commit()
        print(f"Successfully processed {cursor.rowcount} new products into the database.")
    else:
        print("No valid new products found to add.")
        
    conn.close()

if __name__ == '__main__':
    main()
