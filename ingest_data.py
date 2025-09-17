# ingest_data.py (Production-Ready v3.0 - Matching Your Schema)

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
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingest_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("SpaCy model 'en_core_web_sm' loaded successfully.")
except (OSError, ImportError):
    logger.warning("SpaCy model not found. Text extraction from PDFs will be disabled.")
    nlp = None

DATA_FOLDER = 'data'
DATABASE_FILE = 'products.db'
BACKUP_FOLDER = 'backups'

# Enhanced product categorization mapping
CATEGORY_MAPPING = {
    'video conferencing': 'UC & Collaboration Devices',
    'vc': 'UC & Collaboration Devices',
    'collaboration': 'UC & Collaboration Devices',
    'camera': 'UC & Collaboration Devices',
    'microphone': 'UC & Collaboration Devices',
    'speaker': 'UC & Collaboration Devices',
    'display': 'Displays & Projectors',
    'monitor': 'Displays & Projectors',
    'projector': 'Displays & Projectors',
    'screen': 'Displays & Projectors',
    'mount': 'Mounts, Racks & Enclosures',
    'bracket': 'Mounts, Racks & Enclosures',
    'rack': 'Mounts, Racks & Enclosures',
    'cable': 'Cables & Connectors',
    'hdmi': 'Cables & Connectors',
    'usb': 'Cables & Connectors',
    'network': 'Networking Equipment',
    'switch': 'Networking Equipment',
    'router': 'Networking Equipment',
    'power': 'Power & Connectivity',
    'ups': 'Power & Connectivity',
    'pdu': 'Power & Connectivity',
    'lighting': 'Lighting & Control',
    'control': 'Lighting & Control',
    'automation': 'Lighting & Control',
    'audio': 'Audio Systems',
    'amplifier': 'Audio Systems',
    'mixer': 'Audio Systems'
}

# Brand recognition patterns
BRAND_PATTERNS = {
    'logitech': ['logitech', 'logi'],
    'poly': ['poly', 'polycom', 'plantronics'],
    'cisco': ['cisco', 'webex', 'tandberg'],
    'microsoft': ['microsoft', 'teams', 'surface'],
    'zoom': ['zoom', 'zoom rooms'],
    'samsung': ['samsung'],
    'lg': ['lg electronics', 'lg'],
    'sony': ['sony'],
    'epson': ['epson'],
    'barco': ['barco'],
    'crestron': ['crestron'],
    'extron': ['extron'],
    'amx': ['amx', 'harman'],
    'shure': ['shure'],
    'sennheiser': ['sennheiser'],
    'bose': ['bose'],
    'qsc': ['qsc'],
    'kramer': ['kramer']
}

# Tier mapping based on price ranges
TIER_MAPPING = {
    'budget': (0, 500),
    'mid-range': (500, 2000),
    'premium': (2000, 10000),
    'enterprise': (10000, float('inf'))
}

def create_backup():
    """Create a backup of the existing database"""
    if not os.path.exists(DATABASE_FILE):
        return
    
    os.makedirs(BACKUP_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_FOLDER, f"products_backup_{timestamp}.db")
    
    try:
        import shutil
        shutil.copy2(DATABASE_FILE, backup_file)
        logger.info(f"Database backup created: {backup_file}")
    except Exception as e:
        logger.warning(f"Failed to create backup: {str(e)}")

def get_existing_columns(cursor):
    """Get list of existing columns in the products table"""
    cursor.execute("PRAGMA table_info(products)")
    columns = cursor.fetchall()
    return [column[1] for column in columns]  # column[1] is the column name

def initialize_database():
    """Initialize database matching your existing schema"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Create table matching your schema: category, brand, name, price, features, tier, use_case_tags, compatibility_tags
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id TEXT PRIMARY KEY,
        category TEXT,
        brand TEXT,
        name TEXT NOT NULL,
        price REAL NOT NULL,
        features TEXT,
        tier TEXT,
        use_case_tags TEXT,
        compatibility_tags TEXT
    )
    ''')
    
    # Get existing columns
    existing_columns = get_existing_columns(cursor)
    
    # Define required columns that might be missing
    required_columns = {
        'id': 'TEXT PRIMARY KEY',
        'category': 'TEXT',
        'brand': 'TEXT',
        'name': 'TEXT NOT NULL',
        'price': 'REAL NOT NULL',
        'features': 'TEXT',
        'tier': 'TEXT',
        'use_case_tags': 'TEXT',
        'compatibility_tags': 'TEXT'
    }
    
    # Add missing columns (except PRIMARY KEY columns which can't be added after creation)
    for column_name, column_definition in required_columns.items():
        if column_name not in existing_columns and column_name != 'id':
            try:
                cursor.execute(f'ALTER TABLE products ADD COLUMN {column_name} {column_definition}')
                logger.info(f"Added {column_name} column to existing products table")
            except sqlite3.OperationalError as e:
                logger.warning(f"Could not add column {column_name}: {e}")
    
    # Create indexes for better performance
    indexes = [
        'CREATE INDEX IF NOT EXISTS idx_brand ON products(brand)',
        'CREATE INDEX IF NOT EXISTS idx_category ON products(category)',
        'CREATE INDEX IF NOT EXISTS idx_price ON products(price)',
        'CREATE INDEX IF NOT EXISTS idx_name ON products(name)',
        'CREATE INDEX IF NOT EXISTS idx_tier ON products(tier)'
    ]
    
    for index_sql in indexes:
        try:
            cursor.execute(index_sql)
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not create index: {e}")
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def clean_text(text: str) -> str:
    """Enhanced text cleaning with better normalization"""
    if not text or pd.isna(text):
        return ""
    
    # Convert to string and normalize
    text = str(text).strip()
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep essential punctuation
    text = re.sub(r'[^\w\s\-\.\,\(\)\&\+\:]', '', text)
    
    # Normalize common abbreviations
    text = re.sub(r'\bw\/\b', 'with', text, flags=re.IGNORECASE)
    text = re.sub(r'\b&\b', 'and', text, flags=re.IGNORECASE)
    
    return text.strip()

def extract_brand(product_name: str, features: str = "") -> str:
    """Enhanced brand extraction using pattern matching and NLP"""
    if not product_name:
        return "Unknown"
    
    combined_text = f"{product_name} {features}".lower()
    
    # Check against brand patterns
    for brand, patterns in BRAND_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in combined_text:
                return brand.title()
    
    # Use NLP for entity recognition if available
    if nlp:
        try:
            doc = nlp(product_name)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    brand_candidate = ent.text.lower()
                    for brand, patterns in BRAND_PATTERNS.items():
                        if brand_candidate in [p.lower() for p in patterns]:
                            return brand.title()
        except Exception as e:
            logger.debug(f"NLP brand extraction failed: {str(e)}")
    
    # Fallback: extract first word if it looks like a brand
    first_word = product_name.split()[0] if product_name.split() else ""
    if len(first_word) > 2 and first_word.isalpha():
        return first_word.title()
    
    return "Unknown"

def categorize_product(product_name: str, features: str = "") -> str:
    """Enhanced product categorization using keywords and patterns"""
    if not product_name:
        return "Other"
    
    combined_text = f"{product_name} {features}".lower()
    
    # Score-based categorization for better accuracy
    category_scores = {}
    
    for keyword, category in CATEGORY_MAPPING.items():
        if keyword in combined_text:
            if category not in category_scores:
                category_scores[category] = 0
            category_scores[category] += len(keyword)  # Longer matches get higher scores
    
    if category_scores:
        return max(category_scores, key=category_scores.get)
    
    # Advanced pattern matching
    if re.search(r'\b(4k|uhd|hd|led|lcd|oled)\b', combined_text):
        return "Displays & Projectors"
    
    if re.search(r'\b(wireless|bluetooth|wi-fi|wifi)\b', combined_text):
        return "Networking Equipment"
    
    if re.search(r'\b(conference|meeting|room|huddle)\b', combined_text):
        return "UC & Collaboration Devices"
    
    return "Other"

def determine_tier(price: float) -> str:
    """Determine product tier based on price"""
    for tier_name, (min_price, max_price) in TIER_MAPPING.items():
        if min_price <= price < max_price:
            return tier_name
    return "budget"

def extract_use_case_tags(product_name: str, features: str = "") -> str:
    """Extract use case tags based on product information"""
    combined_text = f"{product_name} {features}".lower()
    use_cases = []
    
    use_case_keywords = {
        'conference-room': ['conference', 'meeting', 'boardroom', 'huddle'],
        'classroom': ['education', 'classroom', 'school', 'university'],
        'auditorium': ['auditorium', 'theater', 'large venue', 'stadium'],
        'home-office': ['home', 'personal', 'small office'],
        'corporate': ['corporate', 'enterprise', 'business'],
        'broadcast': ['broadcast', 'streaming', 'production'],
        'retail': ['retail', 'store', 'kiosk', 'digital signage']
    }
    
    for use_case, keywords in use_case_keywords.items():
        for keyword in keywords:
            if keyword in combined_text:
                use_cases.append(use_case)
                break
    
    return ', '.join(use_cases) if use_cases else 'general-purpose'

def extract_compatibility_tags(product_name: str, features: str = "") -> str:
    """Extract compatibility tags based on product information"""
    combined_text = f"{product_name} {features}".lower()
    compatibility = []
    
    compatibility_keywords = {
        'teams': ['teams', 'microsoft teams'],
        'zoom': ['zoom', 'zoom rooms'],
        'webex': ['webex', 'cisco webex'],
        'skype': ['skype', 'skype for business'],
        'google-meet': ['google meet', 'hangouts'],
        'usb': ['usb', 'plug and play'],
        'hdmi': ['hdmi'],
        'wireless': ['wireless', 'wifi', 'bluetooth'],
        'poe': ['poe', 'power over ethernet'],
        'sip': ['sip', 'voip']
    }
    
    for compat, keywords in compatibility_keywords.items():
        for keyword in keywords:
            if keyword in combined_text:
                compatibility.append(compat)
                break
    
    return ', '.join(compatibility) if compatibility else 'standard'

def validate_price(price_str: str) -> Tuple[float, List[str]]:
    """Enhanced price validation with error tracking"""
    issues = []
    
    if not price_str or pd.isna(price_str):
        issues.append("Missing price")
        return 0.0, issues
    
    # Clean price string
    price_clean = str(price_str).strip()
    price_clean = re.sub(r'[^\d\.\-\+]', '', price_clean)
    
    try:
        price = float(price_clean)
        if price < 0:
            issues.append("Negative price")
            return abs(price), issues
        elif price == 0:
            issues.append("Zero price")
        elif price > 100000:
            issues.append("Unusually high price")
        return price, issues
    except ValueError:
        issues.append("Invalid price format")
        return 0.0, issues

def extract_text_from_pdf(file_path: str) -> str:
    """Enhanced PDF text extraction with better error handling"""
    try:
        text_content = ""
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1} of {file_path}: {str(e)}")
                    continue
        
        return clean_text(text_content)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {file_path}: {str(e)}")
        return ""

def parse_pdf_catalog(file_path: str) -> List[Dict[str, Any]]:
    """Enhanced PDF parsing with structured data extraction"""
    logger.info(f"Parsing PDF catalog: {file_path}")
    
    text_content = extract_text_from_pdf(file_path)
    if not text_content:
        logger.warning(f"No text extracted from PDF: {file_path}")
        return []
    
    products = []
    
    # Enhanced regex patterns for product extraction
    product_patterns = [
        r'(?:Product|Model):\s*([^\n]+)\s*(?:Price|Cost):\s*\$?([0-9,]+\.?[0-9]*)',
        r'([A-Z][A-Za-z0-9\s\-]+)\s+\$([0-9,]+\.?[0-9]*)',
        r'([A-Za-z0-9\s\-]+)(?:\s+[-|–]\s+)?\$([0-9,]+\.?[0-9]*)'
    ]
    
    for pattern in product_patterns:
        matches = re.findall(pattern, text_content, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            name, price_str = match
            name = clean_text(name)
            
            if len(name) > 3:  # Filter out very short matches
                price, price_issues = validate_price(price_str)
                
                brand = extract_brand(name)
                category = categorize_product(name)
                tier = determine_tier(price)
                use_case_tags = extract_use_case_tags(name)
                compatibility_tags = extract_compatibility_tags(name)
                
                product_data = {
                    'id': str(uuid.uuid4()),
                    'category': category,
                    'brand': brand,
                    'name': name,
                    'price': price,
                    'features': "",  # Could be enhanced with more sophisticated extraction
                    'tier': tier,
                    'use_case_tags': use_case_tags,
                    'compatibility_tags': compatibility_tags
                }
                
                products.append(product_data)
    
    logger.info(f"Extracted {len(products)} products from PDF")
    return products

def process_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """Enhanced CSV processing matching your database schema"""
    logger.info(f"Processing CSV file: {file_path}")
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                # Add error handling for malformed CSV lines
                df = pd.read_csv(
                    file_path, 
                    encoding=encoding,
                    on_bad_lines='skip',  # Skip bad lines instead of failing
                    engine='python'  # More robust parser
                )
                logger.info(f"Successfully read CSV with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Failed to read CSV with {encoding}: {str(e)}")
                continue
        
        if df is None:
            raise Exception("Could not read CSV with any supported encoding")
        
        # Log if any rows were skipped
        logger.info(f"CSV loaded with {len(df)} rows")
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        products = []
        
        for index, row in df.iterrows():
            try:
                # Extract basic information - map to your schema
                name = clean_text(row.get('name', ''))
                if not name:
                    logger.warning(f"Row {index + 1}: Missing product name, skipping")
                    continue
                
                price, price_issues = validate_price(row.get('price', ''))
                
                # Extract or infer other fields matching your schema
                brand = clean_text(row.get('brand', '')) or extract_brand(name)
                category = clean_text(row.get('category', '')) or categorize_product(name)
                features = clean_text(row.get('features', ''))
                tier = clean_text(row.get('tier', '')) or determine_tier(price)
                use_case_tags = clean_text(row.get('use_case_tags', '')) or extract_use_case_tags(name, features)
                compatibility_tags = clean_text(row.get('compatibility_tags', '')) or extract_compatibility_tags(name, features)
                
                product_data = {
                    'id': str(uuid.uuid4()),
                    'category': category,
                    'brand': brand,
                    'name': name,
                    'price': price,
                    'features': features,
                    'tier': tier,
                    'use_case_tags': use_case_tags,
                    'compatibility_tags': compatibility_tags
                }
                
                products.append(product_data)
                
            except Exception as e:
                logger.error(f"Error processing row {index + 1}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(products)} products from CSV")
        return products
        
    except Exception as e:
        logger.error(f"Error processing CSV file {file_path}: {str(e)}")
        return []

def insert_products_to_db(products: List[Dict[str, Any]]) -> int:
    """Insert products matching your database schema"""
    if not products:
        return 0
    
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    inserted_count = 0
    updated_count = 0
    
    # Verify all required columns exist
    existing_columns = get_existing_columns(cursor)
    required_columns = ['category', 'brand', 'name', 'price', 'features', 'tier', 'use_case_tags', 'compatibility_tags']
    
    missing_columns = [col for col in required_columns if col not in existing_columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.info("Re-initializing database to add missing columns...")
        conn.close()
        initialize_database()
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
    
    for product in products:
        try:
            # Ensure all required fields have default values
            product.setdefault('category', 'Other')
            product.setdefault('brand', 'Unknown')
            product.setdefault('features', '')
            product.setdefault('tier', 'budget')
            product.setdefault('use_case_tags', 'general-purpose')
            product.setdefault('compatibility_tags', 'standard')
            
            # Check for existing product with same name and brand
            cursor.execute('''
            SELECT id FROM products 
            WHERE name = ? AND brand = ?
            ''', (product['name'], product['brand']))
            
            existing = cursor.fetchone()
            
            if existing:
                existing_id = existing[0]
                # Update existing product
                cursor.execute('''
                UPDATE products SET 
                    category = ?, price = ?, features = ?, tier = ?,
                    use_case_tags = ?, compatibility_tags = ?
                WHERE id = ?
                ''', (
                    product['category'], product['price'], product['features'],
                    product['tier'], product['use_case_tags'], 
                    product['compatibility_tags'], existing_id
                ))
                updated_count += 1
                logger.debug(f"Updated existing product: {product['name']}")
            else:
                # Insert new product
                cursor.execute('''
                INSERT INTO products (
                    id, category, brand, name, price, features, tier, use_case_tags, compatibility_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    product['id'], product['category'], product['brand'], product['name'],
                    product['price'], product['features'], product['tier'],
                    product['use_case_tags'], product['compatibility_tags']
                ))
                inserted_count += 1
                
        except Exception as e:
            logger.error(f"Error inserting product {product.get('name', 'Unknown')}: {str(e)}")
            continue
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database operation completed: {inserted_count} inserted, {updated_count} updated")
    return inserted_count + updated_count

def generate_data_quality_report() -> Dict[str, Any]:
    """Generate comprehensive data quality report"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Basic statistics
    cursor.execute('SELECT COUNT(*) FROM products')
    total_products = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(price) FROM products WHERE price > 0')
    avg_price = cursor.fetchone()[0] or 0
    
    # Category distribution
    cursor.execute('SELECT category, COUNT(*) FROM products GROUP BY category ORDER BY COUNT(*) DESC')
    category_distribution = dict(cursor.fetchall())
    
    # Brand distribution
    cursor.execute('SELECT brand, COUNT(*) FROM products GROUP BY brand ORDER BY COUNT(*) DESC LIMIT 10')
    top_brands = dict(cursor.fetchall())
    
    # Tier distribution
    cursor.execute('SELECT tier, COUNT(*) FROM products GROUP BY tier ORDER BY COUNT(*) DESC')
    tier_distribution = dict(cursor.fetchall())
    
    # Price range statistics
    cursor.execute('SELECT MIN(price), MAX(price) FROM products WHERE price > 0')
    price_range = cursor.fetchone()
    
    conn.close()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_products': total_products,
        'average_price': round(avg_price, 2) if avg_price else 0,
        'price_range': {'min': price_range[0], 'max': price_range[1]} if price_range and price_range[0] is not None else {'min': 0, 'max': 0},
        'category_distribution': category_distribution,
        'top_brands': top_brands,
        'tier_distribution': tier_distribution
    }
    
    return report

def main():
    """Enhanced main function with comprehensive processing"""
    logger.info("Starting AV product data ingestion process")
    
    # Create backup
    create_backup()
    
    # Initialize database
    initialize_database()
    
    # Ensure data folder exists
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    total_processed = 0
    
    # Process all files in data folder
    for filename in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, filename)
        
        if not os.path.isfile(file_path):
            continue
            
        logger.info(f"Processing file: {filename}")
        
        products = []
        
        if filename.lower().endswith('.csv'):
            products = process_csv_file(file_path)
        elif filename.lower().endswith('.pdf'):
            products = parse_pdf_catalog(file_path)
        else:
            logger.warning(f"Unsupported file format: {filename}")
            continue
        
        if products:
            processed_count = insert_products_to_db(products)
            total_processed += processed_count
            logger.info(f"Successfully processed {processed_count} products from {filename}")
        else:
            logger.warning(f"No products extracted from {filename}")
    
    # Generate and save data quality report
    report = generate_data_quality_report()
    report_path = f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Data quality report saved to: {report_path}")
    logger.info(f"Data ingestion completed. Total products processed: {total_processed}")
    
    # Print summary
    print(f"\n=== DATA INGESTION SUMMARY ===")
    print(f"Total products processed: {total_processed}")
    print(f"Average price: ${report['average_price']}")
    print(f"Category distribution: {report['category_distribution']}")
    print(f"Tier distribution: {report['tier_distribution']}")
    print(f"Data quality report: {report_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise
