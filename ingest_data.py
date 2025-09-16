# ingest_data.py (Production-Ready v3.0)

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

def initialize_database():
    """Initialize database with enhanced schema"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Enhanced products table with additional fields
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        price REAL NOT NULL,
        brand TEXT,
        category TEXT,
        features TEXT,
        specifications TEXT,
        model_number TEXT,
        warranty_years INTEGER DEFAULT 1,
        availability TEXT DEFAULT 'In Stock',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        source_file TEXT,
        data_quality_score REAL DEFAULT 1.0
    )
    ''')
    
    # Add data_quality_score column if it doesn't exist (for existing databases)
    try:
        cursor.execute('ALTER TABLE products ADD COLUMN data_quality_score REAL DEFAULT 1.0')
        logger.info("Added data_quality_score column to existing products table")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_brand ON products(brand)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON products(category)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_price ON products(price)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_name ON products(name)')
    
    # Data quality tracking table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS data_quality_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT,
        issue_type TEXT,
        issue_description TEXT,
        severity TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    ''')
    
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

def calculate_data_quality_score(product_data: Dict) -> float:
    """Calculate data quality score based on completeness and validity"""
    score = 1.0
    
    # Required fields penalty
    if not product_data.get('name') or product_data['name'].strip() == "":
        score -= 0.3
    if not product_data.get('price') or product_data['price'] <= 0:
        score -= 0.2
    
    # Brand and category quality
    if product_data.get('brand') == "Unknown":
        score -= 0.1
    if product_data.get('category') == "Other":
        score -= 0.1
    
    # Feature completeness
    features = product_data.get('features', "")
    if not features or len(features.strip()) < 10:
        score -= 0.1
    
    # Model number presence
    if not product_data.get('model_number'):
        score -= 0.05
    
    return max(0.0, min(1.0, score))

def log_data_quality_issue(product_id: str, issue_type: str, description: str, severity: str = "medium"):
    """Log data quality issues for monitoring"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO data_quality_log (product_id, issue_type, issue_description, severity)
    VALUES (?, ?, ?, ?)
    ''', (product_id, issue_type, description, severity))
    
    conn.commit()
    conn.close()

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
                
                product_data = {
                    'id': str(uuid.uuid4()),
                    'name': name,
                    'price': price,
                    'brand': brand,
                    'category': category,
                    'features': "",  # Could be enhanced with more sophisticated extraction
                    'specifications': "",
                    'model_number': "",
                    'source_file': os.path.basename(file_path)
                }
                
                product_data['data_quality_score'] = calculate_data_quality_score(product_data)
                products.append(product_data)
                
                # Log quality issues
                if price_issues:
                    for issue in price_issues:
                        log_data_quality_issue(product_data['id'], "price", issue)
    
    logger.info(f"Extracted {len(products)} products from PDF")
    return products

def process_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """Enhanced CSV processing with better error handling and validation"""
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
        
        # Map common column variations
        column_mapping = {
            'product_name': ['name', 'product', 'title', 'item_name'],
            'price': ['cost', 'amount', 'value', 'msrp'],
            'brand': ['manufacturer', 'vendor', 'make'],
            'category': ['type', 'class', 'group'],
            'features': ['description', 'details', 'specs'],
            'model': ['model_number', 'sku', 'part_number']
        }
        
        # Rename columns based on mapping
        for target_col, variations in column_mapping.items():
            for variation in variations:
                if variation in df.columns and target_col not in df.columns:
                    df.rename(columns={variation: target_col}, inplace=True)
                    break
        
        products = []
        
        for index, row in df.iterrows():
            try:
                # Extract basic information
                name = clean_text(row.get('product_name', row.get('name', '')))
                if not name:
                    logger.warning(f"Row {index + 1}: Missing product name, skipping")
                    continue
                
                price, price_issues = validate_price(row.get('price', ''))
                
                # Extract or infer other fields
                brand = row.get('brand', '') or extract_brand(name)
                category = row.get('category', '') or categorize_product(name)
                features = clean_text(row.get('features', ''))
                specifications = clean_text(row.get('specifications', ''))
                model_number = clean_text(row.get('model', ''))
                
                # Handle warranty_years safely
                try:
                    warranty_years = int(row.get('warranty_years', 1))
                except (ValueError, TypeError):
                    warranty_years = 1
                
                product_data = {
                    'id': str(uuid.uuid4()),
                    'name': name,
                    'price': price,
                    'brand': brand,
                    'category': category,
                    'features': features,
                    'specifications': specifications,
                    'model_number': model_number,
                    'warranty_years': warranty_years,
                    'availability': row.get('availability', 'In Stock'),
                    'source_file': os.path.basename(file_path)
                }
                
                product_data['data_quality_score'] = calculate_data_quality_score(product_data)
                products.append(product_data)
                
                # Log quality issues
                if price_issues:
                    for issue in price_issues:
                        log_data_quality_issue(product_data['id'], "price", issue)
                
            except Exception as e:
                logger.error(f"Error processing row {index + 1}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(products)} products from CSV")
        return products
        
    except Exception as e:
        logger.error(f"Error processing CSV file {file_path}: {str(e)}")
        return []

def insert_products_to_db(products: List[Dict[str, Any]]) -> int:
    """Enhanced database insertion with conflict resolution"""
    if not products:
        return 0
    
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    inserted_count = 0
    updated_count = 0
    
    for product in products:
        try:
            # Check for existing product with same name and brand
            cursor.execute('''
            SELECT id, data_quality_score FROM products 
            WHERE name = ? AND brand = ?
            ''', (product['name'], product['brand']))
            
            existing = cursor.fetchone()
            
            if existing:
                existing_id, existing_score = existing
                # Update if new data has better quality score
                if product['data_quality_score'] > existing_score:
                    cursor.execute('''
                    UPDATE products SET 
                        price = ?, category = ?, features = ?, specifications = ?,
                        model_number = ?, warranty_years = ?, availability = ?,
                        updated_at = CURRENT_TIMESTAMP, source_file = ?,
                        data_quality_score = ?
                    WHERE id = ?
                    ''', (
                        product['price'], product['category'], product['features'],
                        product['specifications'], product['model_number'],
                        product['warranty_years'], product['availability'],
                        product['source_file'], product['data_quality_score'], existing_id
                    ))
                    updated_count += 1
                    logger.debug(f"Updated existing product: {product['name']}")
            else:
                # Insert new product
                cursor.execute('''
                INSERT INTO products (
                    id, name, price, brand, category, features, specifications,
                    model_number, warranty_years, availability, source_file, data_quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    product['id'], product['name'], product['price'], product['brand'],
                    product['category'], product['features'], product['specifications'],
                    product['model_number'], product['warranty_years'], product['availability'],
                    product['source_file'], product['data_quality_score']
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
    
    cursor.execute('SELECT AVG(data_quality_score) FROM products')
    avg_quality_score = cursor.fetchone()[0] or 0
    
    # Quality distribution
    cursor.execute('''
    SELECT 
        CASE 
            WHEN data_quality_score >= 0.8 THEN 'High'
            WHEN data_quality_score >= 0.6 THEN 'Medium'
            ELSE 'Low'
        END as quality_level,
        COUNT(*) as count
    FROM products
    GROUP BY quality_level
    ''')
    quality_distribution = dict(cursor.fetchall())
    
    # Category distribution
    cursor.execute('SELECT category, COUNT(*) FROM products GROUP BY category ORDER BY COUNT(*) DESC')
    category_distribution = dict(cursor.fetchall())
    
    # Brand distribution
    cursor.execute('SELECT brand, COUNT(*) FROM products GROUP BY brand ORDER BY COUNT(*) DESC LIMIT 10')
    top_brands = dict(cursor.fetchall())
    
    # Recent issues
    cursor.execute('''
    SELECT issue_type, COUNT(*) FROM data_quality_log 
    WHERE created_at >= datetime('now', '-7 days')
    GROUP BY issue_type
    ''')
    recent_issues = dict(cursor.fetchall())
    
    conn.close()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_products': total_products,
        'average_quality_score': round(avg_quality_score, 3),
        'quality_distribution': quality_distribution,
        'category_distribution': category_distribution,
        'top_brands': top_brands,
        'recent_issues': recent_issues
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
    print(f"Average data quality score: {report['average_quality_score']}")
    print(f"Quality distribution: {report['quality_distribution']}")
    print(f"Data quality report: {report_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise
