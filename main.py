from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import io
import json
import re
import pandas as pd
import pdfplumber
import spacy

# Load NLP model once at startup
try:
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    print("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    nlp = None

app = FastAPI(
    title="Professional BOQ Generator API",
    description="A backend service for intelligent BOQ generation from various document types."
)

# In-memory database
product_database = []

# Pydantic Models for API validation and structure
class Product(BaseModel):
    id: str
    name: str
    price: float
    brand: Optional[str] = "Generic"
    category: Optional[str] = "Miscellaneous"
    imported: Optional[bool] = False
    
class QuickAddProduct(BaseModel):
    name: str
    price: float
    brand: Optional[str] = ""
    category: Optional[str] = ""

class BoqConfig(BaseModel):
    projectName: str
    clientName: str
    requirements: Optional[str] = ""
    budgetRange: str
    priority: str
    contingency: float

class BoqItem(BaseModel):
    id: str
    productId: str
    name: str
    brand: str
    category: str
    quantity: int
    unit: str
    unitPrice: float
    totalPrice: float
    description: str

class BoqSummary(BaseModel):
    totalCost: float
    totalQuantity: int
    itemCount: int
    categoryCount: int
    categoryBreakdown: Dict[str, Any]

class Boq(BaseModel):
    id: str
    projectName: str
    clientName: str
    generatedAt: str
    items: List[BoqItem]
    summary: BoqSummary
    metadata: Dict[str, Any]

# --- Core Logic Functions ---
def normalize_product_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_products = []
    mappings = {
        'name': ['name', 'product', 'item', 'description', 'title'],
        'price': ['price', 'cost', 'amount', 'rate', 'value'],
        'brand': ['brand', 'manufacturer', 'make'],
        'category': ['category', 'type', 'group', 'class']
    }

    for item in data:
        product = {}
        found_name = False
        found_price = False

        for key, alternatives in mappings.items():
            for alt in alternatives:
                if alt in item and item[alt] is not None and str(item[alt]).strip() != '':
                    value = str(item[alt]).strip()
                    if key == 'price':
                        # Clean up price strings
                        value = re.sub(r'[$,]', '', value)
                        try:
                            product[key] = float(value)
                            found_price = True
                        except ValueError:
                            pass
                    else:
                        product[key] = value
                        if key == 'name':
                            found_name = True
                    break
        
        if found_name and found_price:
            product['id'] = str(uuid.uuid4())
            product['imported'] = True
            normalized_products.append(product)

    return normalized_products

def extract_products_from_text(text: str) -> List[Dict[str, Any]]:
    if not nlp:
        return []
    
    products = []
    doc = nlp(text)
    
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG", "WORK_OF_ART"]:
            surrounding_text = doc[max(0, ent.start - 10): min(len(doc), ent.end + 10)].text
            price_match = re.search(r'\$?(\d+[\.,]\d{2})', surrounding_text)
            if price_match:
                price = float(price_match.group(1).replace(',', '.'))
                if price > 0:
                    products.append({
                        "name": ent.text.strip(),
                        "price": price,
                        "brand": ent.text.split()[0], # Simple heuristic for brand
                        "category": ent.label_
                    })
    return products

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "BOQ Generator Backend is running. Access /docs for API documentation."}

@app.post("/api/upload_file")
async def upload_file_endpoint(file: UploadFile = File(...)):
    """
    Processes an uploaded file, extracts product data, and adds it to the database.
    """
    try:
        content = await file.read()
        
        if file.content_type == "application/pdf":
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = ""
                tables_products = []
                for page in pdf.pages:
                    text += page.extract_text() or ""
                    try:
                        tables = page.extract_tables()
                        for table in tables:
                            df = pd.DataFrame(table)
                            if not df.empty and df.shape[0] > 1:
                                table_data = df.to_dict('records')
                                tables_products.extend(normalize_product_data(table_data))
                    except Exception as e:
                        print(f"Failed to extract table: {e}")
                
                if tables_products:
                    products = tables_products
                else:
                    products = extract_products_from_text(text)
            
        elif file.content_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel", "text/csv"]:
            df = pd.read_excel(io.BytesIO(content)) if file.content_type.endswith('sheet') else pd.read_csv(io.StringIO(content.decode()))
            products = normalize_product_data(df.to_dict('records'))
        
        elif file.content_type == "application/json":
            data = json.loads(content.decode())
            products = normalize_product_data(data if isinstance(data, list) else [data])
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        for p in products:
            product_database.append(Product(**p))

        return {"message": f"File processed successfully, {len(products)} products imported.", "products": products}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/api/add_product", response_model=Product)
def add_product_endpoint(new_product: QuickAddProduct):
    """
    Adds a new product manually to the database.
    """
    product = Product(
        id=str(uuid.uuid4()),
        name=new_product.name,
        price=new_product.price,
        brand=new_product.brand or "Generic",
        category=new_product.category or "Miscellaneous"
    )
    product_database.append(product)
    return product

@app.get("/api/product_database", response_model=List[Product])
def get_products():
    """
    Retrieves all products from the database.
    """
    return product_database

@app.post("/api/generate_boq", response_model=Boq)
def generate_boq_endpoint(config: BoqConfig):
    """
    Generates a BOQ based on project configuration and stored products.
    """
    if not product_database:
        raise HTTPException(status_code=400, detail="No products available to generate a BOQ.")

    # Apply advanced business logic here
    # 1. Product selection based on requirements (simple keyword matching for now)
    selected_products = []
    req_words = set(config.requirements.lower().split())
    
    for product in product_database:
        product_words = set(product.name.lower().split() + product.category.lower().split())
        if not req_words or not product_words.isdisjoint(req_words):
             selected_products.append(product)

    # Fallback if no matches
    if not selected_products:
        import random
        selected_products = random.sample(product_database, min(10, len(product_database)))

    # 2. Quantity calculation (simple heuristic based on category)
    boq_items = []
    for product in selected_products:
        quantity = 1
        if "speaker" in product.name.lower() or "microphone" in product.name.lower():
            quantity = 2
        
        boq_items.append({
            "id": str(uuid.uuid4()),
            "productId": product.id,
            "name": product.name,
            "brand": product.brand,
            "category": product.category,
            "quantity": quantity,
            "unit": "pc",
            "unitPrice": product.price,
            "totalPrice": quantity * product.price,
            "description": f"{product.name} - {product.brand}"
        })
    
    # 3. Apply pricing strategy and contingency
    total_cost = 0
    contingency_multiplier = 1 + (config.contingency / 100)
    for item in boq_items:
        item['totalPrice'] = item['quantity'] * item['unitPrice'] * contingency_multiplier
        total_cost += item['totalPrice']

    # 4. Generate summary
    summary = {
        "totalCost": total_cost,
        "totalQuantity": sum(item['quantity'] for item in boq_items),
        "itemCount": len(boq_items),
        "categoryCount": len(set(item['category'] for item in boq_items)),
        "categoryBreakdown": {}
    }

    # Final BOQ
    boq = {
        "id": str(uuid.uuid4()),
        "projectName": config.projectName,
        "clientName": config.clientName,
        "generatedAt": datetime.now().isoformat(),
        "items": boq_items,
        "summary": summary,
        "metadata": config.dict()
    }
    
    return Boq(**boq)
