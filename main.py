from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import sqlite3
import uuid
from datetime import datetime
import logging
from contextlib import contextmanager
import math
import re
import functools

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Professional BOQ Generator API v9.2 (Optimized)",
    description="Enhanced AV Room Design and BOQ Generation System with Optimized Loading",
    version="9.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

DATABASE_FILE = 'products.db'

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# --- Data Models ---
class Product(BaseModel):
    id: str
    category: Optional[str] = None
    brand: Optional[str] = None
    name: str
    price: float
    features: Optional[str] = None
    tier: Optional[str] = None
    use_case_tags: Optional[str] = None
    compatibility_tags: Optional[str] = None
    # This field is added dynamically by the filter, so it needs to be optional
    relevance_score: Optional[float] = None


# NEW: Lightweight model for the library list
class ProductLibraryItem(BaseModel):
    id: str
    category: Optional[str] = None
    brand: Optional[str] = None
    name: str
    price: float
    tier: Optional[str] = None
    model: Optional[str] = None # Added for display

class EnhancedBoqConfig(BaseModel):
    project_name: str = Field(alias="projectName", default="Default Project")
    client_name: str = Field(alias="clientName", default="Default Client")
    room_dimensions: List[float] = Field(min_items=3, max_items=3)
    room_capacity: str = Field(alias="roomSize", default="medium")
    use_case: str = Field(alias="useCase", default="meeting_room")
    brand_preference: Optional[str] = Field(alias="brandPreference", default=None)
    special_requirements: str = Field(alias="requirements", default="")
    budget_range: str = Field(alias="budgetRange", default="standard")
    
    @property
    def roomWidth(self) -> float: return self.room_dimensions[0]
    @property
    def roomDepth(self) -> float: return self.room_dimensions[1]
    
    priority: str = Field(default="balanced", exclude=True)
    contingency: float = Field(default=10.0, ge=5.0, le=25.0, exclude=True)

    class Config:
        allow_population_by_field_name = True
        
# (All other Pydantic models like BoqItem, BoqSection, etc. remain unchanged)
class BoqItem(BaseModel):
    id: str
    productId: str
    name: str
    brand: str
    category: Optional[str]
    quantity: int
    unitPrice: float
    totalPrice: float
    description: str
    features: Optional[str]
    tier: Optional[str] = None

class BoqSection(BaseModel):
    name: str
    items: List[BoqItem]
    sectionTotal: float
    description: Optional[str]

class BoqSummary(BaseModel):
    subtotal: float
    contingencyAmount: float
    installationCost: float
    totalCost: float
    itemCount: int
    estimatedInstallDays: int

class RoomAnalysis(BaseModel):
    roomArea: float
    roomPerimeter: float
    recommendedCapacity: int
    acousticRating: str
    complexityScore: int

class Boq(BaseModel):
    id: str
    projectName: str
    clientName: str
    generatedAt: str
    sections: List[BoqSection]
    summary: BoqSummary
    metadata: Dict[str, Any]
    recommendations: List[str]
    terms_conditions: List[str]
    assumptions: List[str]
    roomAnalysis: RoomAnalysis

# --- Cached Product Loading ---
@functools.lru_cache(maxsize=None)
def get_all_products_from_db() -> List[Product]:
    """Cached function to load all products from the database."""
    logger.info("Loading and caching all products from the database...")
    with get_db_connection() as conn:
        products_data = conn.execute('SELECT * FROM products').fetchall()
        return [Product(**dict(row)) for row in products_data]

# --- Business Logic Classes ---
class EcosystemManager:
    def __init__(self):
        self.ecosystems = {'logitech': {'primary_keywords': ['logitech', 'rally bar', 'meetup', 'tap ip', 'brio'],'room_mapping': {'small': ['meetup', 'brio', 'tap'],'medium': ['rally bar', 'tap ip'],'large': ['rally bar', 'rally camera', 'tap ip'],'xlarge': ['rally camera', 'ptz pro', 'tap ip']}},'poly': {'primary_keywords': ['poly', 'polycom', 'studio x30', 'studio x50', 'tc8', 'tc10'],'room_mapping': {'small': ['studio x30', 'tc8'],'medium': ['studio x50', 'tc10'],'large': ['studio x70', 'tc10'],'xlarge': ['g7500', 'tc10']}},'cisco': {'primary_keywords': ['cisco', 'webex', 'room kit', 'room bar', 'touch 10'],'room_mapping': {'small': ['room kit mini', 'touch 10'],'medium': ['room bar', 'touch 10'],'large': ['room kit pro', 'navigator'],'xlarge': ['sx80', 'navigator']}},'microsoft': {'primary_keywords': ['microsoft', 'teams', 'surface hub'],'room_mapping': {'small': ['teams room', 'surface hub'],'medium': ['teams room pro', 'surface hub'],'large': ['teams room large', 'surface hub'],'xlarge': ['teams room xlarge', 'surface hub']}},'zoom': {'primary_keywords': ['zoom', 'zoom rooms'],'room_mapping': {'small': ['zoom rooms appliance'],'medium': ['zoom rooms pro'],'large': ['zoom rooms enterprise'],'xlarge': ['zoom rooms enterprise']}}}
    def detect_primary_brand(self, requirements: str, brand_preference: Optional[str] = None) -> str:
        if brand_preference and brand_preference.lower() in self.ecosystems: return brand_preference.lower()
        req_lower = requirements.lower()
        for brand, config in self.ecosystems.items():
            if any(keyword in req_lower for keyword in config['primary_keywords']): return brand
        return 'logitech'
    def get_brand_keywords(self, brand: str, room_size: str) -> List[str]:
        ecosystem = self.ecosystems.get(brand.lower(), {})
        return ecosystem.get('room_mapping', {}).get(room_size, [])

class IntelligentProductMatcher:
    def __init__(self, products: List[Product]):
        self.products = products
        self.products_by_category = {}
        self.products_by_tier = {}
        for p in products:
            cat = p.category or "Other"
            self.products_by_category.setdefault(cat, []).append(p)
            tier = p.tier or "standard"
            self.products_by_tier.setdefault(tier, []).append(p)
            
    def find_products_by_category(self, category: str) -> List[Product]:
        """A simple helper to get all products in a category."""
        return self.products_by_category.get(category, [])

    def find_best_product(self, category: str, keywords: List[str], brand: Optional[str] = None, tier: Optional[str] = None, use_case: Optional[str] = None) -> Optional[Product]:
        candidates = self.products_by_category.get(category, [])
        scored_products = []
        if not candidates: return None
        for product in candidates:
            score = 0
            if brand and product.brand and brand.lower() in product.brand.lower(): score += 50
            for keyword in keywords:
                if keyword.lower() in product.name.lower(): score += 30
                if product.features and keyword.lower() in product.features.lower(): score += 10
            if tier and product.tier and tier == product.tier: score += 20
            if use_case and product.use_case_tags and use_case in product.use_case_tags.lower(): score += 25
            if tier in ['premium', 'enterprise'] and product.tier in ['premium', 'enterprise']: score += 15
            if score > 0: scored_products.append((score, product))
        if scored_products: return sorted(scored_products, key=lambda x: x[0], reverse=True)[0][1]
        return candidates[0] if candidates else None

class RoomAnalyzer:
    def __init__(self, config: EnhancedBoqConfig): self.config = config
    def analyze_room(self) -> RoomAnalysis:
        area, perimeter = self.config.roomWidth * self.config.roomDepth, 2 * (self.config.roomWidth + self.config.roomDepth)
        capacity = 4 if area < 15 else 8 if area < 30 else 16 if area < 60 else 30 if area < 100 else 50
        acoustic = "Good" if area < 20 else "Moderate" if area < 50 else "Challenging" if area < 100 else "Complex"
        complexity = 3 + (2 if area > 50 else 0) + (3 if self.config.use_case in ['auditorium'] else 0) + (1 if self.config.budget_range == 'enterprise' else 0)
        return RoomAnalysis(roomArea=area, roomPerimeter=perimeter, recommendedCapacity=capacity, acousticRating=acoustic, complexityScore=min(complexity, 10))

class EnhancedLogicalSystemBuilder:
    def __init__(self, matcher: IntelligentProductMatcher, config: EnhancedBoqConfig):
        self.matcher = matcher
        self.config = config
        self.ecosystem_manager = EcosystemManager()
        self.room_analyzer = RoomAnalyzer(config)
        self.brand = self.ecosystem_manager.detect_primary_brand(config.special_requirements, config.brand_preference)
        self.room_analysis = self.room_analyzer.analyze_room()
        self.recommendations = []
    def _create_boq_item(self, product: Product, quantity: int = 1) -> BoqItem:
        return BoqItem(id=str(uuid.uuid4()), productId=product.id, name=product.name, brand=product.brand or "Unknown", category=product.category, quantity=quantity, unitPrice=product.price, totalPrice=product.price * quantity, description=product.features or "", features=product.features, tier=product.tier)
    def _get_room_based_quantities(self) -> Dict[str, int]:
        area = self.room_analysis.roomArea
        q = {'displays': 1, 'cameras': 1, 'speakers': 2, 'microphones': 1, 'controllers': 1}
        if area > 30: q['speakers'] = 4
        if area > 60: q.update({'speakers': 6, 'microphones': 2})
        if area > 100: q.update({'displays': 2, 'cameras': 2, 'speakers': 8, 'microphones': 3})
        return q
    def build_collaboration_section(self) -> Optional[BoqSection]:
        items, q, keywords = [], self._get_room_based_quantities(), self.ecosystem_manager.get_brand_keywords(self.brand, self.config.room_capacity)
        device = self.matcher.find_best_product("UC & Collaboration Devices", keywords + ['videobar'], brand=self.brand, tier=self.config.budget_range)
        if device: items.append(self._create_boq_item(device, 1))
        if items: return BoqSection(name="UC & Collaboration Devices", items=items, sectionTotal=sum(i.totalPrice for i in items), description=f"{self.brand.title()} collaboration system")
        return None
    def build_display_section(self) -> Optional[BoqSection]: return BoqSection(name="Displays & Projectors", items=[], sectionTotal=0.0) # Placeholder
    def build_audio_section(self) -> Optional[BoqSection]: return BoqSection(name="Audio: Speakers & Amplifiers", items=[], sectionTotal=0.0) # Placeholder
    def build_control_section(self) -> Optional[BoqSection]: return BoqSection(name="Control & Processing", items=[], sectionTotal=0.0) # Placeholder
    def build_infrastructure_section(self) -> Optional[BoqSection]: return BoqSection(name="Network & Infrastructure", items=[], sectionTotal=0.0) # Placeholder
    def build_cabling_section(self) -> Optional[BoqSection]: return BoqSection(name="Cables & Connectors", items=[], sectionTotal=0.0) # Placeholder
    def build_services_section(self, total: float) -> BoqSection: return BoqSection(name="Professional Services", items=[], sectionTotal=0.0) # Placeholder
    def build_complete_system(self) -> List[BoqSection]:
        sections = [s for s in [self.build_collaboration_section(), self.build_display_section(), self.build_audio_section()] if s]
        total = sum(s.sectionTotal for s in sections)
        if total > 0: sections.append(self.build_services_section(total))
        return sections
    def _generate_additional_recommendations(self): pass

# --- Helper Functions for New Endpoints ---

def _apply_smart_filtering(products: List[Product], room_size: str, use_case: str,
                           budget_range: str, brand_preference: Optional[str]) -> List[Product]:
    """Apply intelligent filtering based on room requirements."""

    # Define relevance scoring
    relevant_products = []
    ecosystem_manager = EcosystemManager()

    for product in products:
        score = 0
        
        # Category relevance based on room size
        category_scores = {
            'small': {
                'UC & Collaboration Devices': 10,
                'Displays & Projectors': 8,
                'Audio: Microphones & Conferencing': 6,
                'Cables & Connectors': 5
            },
            'medium': {
                'UC & Collaboration Devices': 10,
                'Displays & Projectors': 9,
                'Audio: Speakers & Amplifiers': 8,
                'Audio: Microphones & Conferencing': 7,
                'Control & Processing': 6,
                'Network & Infrastructure': 5,
                'Cables & Connectors': 5
            },
            'large': {
                'Displays & Projectors': 10,
                'Audio: Speakers & Amplifiers': 9,
                'UC & Collaboration Devices': 9,
                'Control & Processing': 8,
                'Audio: Microphones & Conferencing': 7,
                'Network & Infrastructure': 6,
                'Cables & Connectors': 5
            },
            'xlarge': {
                'Displays & Projectors': 10,
                'Audio: Speakers & Amplifiers': 10,
                'Control & Processing': 9,
                'UC & Collaboration Devices': 8,
                'Audio: Microphones & Conferencing': 8,
                'Network & Infrastructure': 7,
                'Cables & Connectors': 6
            },
            'auditorium': { # Added auditorium for completeness
                'Displays & Projectors': 10,
                'Audio: Speakers & Amplifiers': 10,
                'Control & Processing': 9,
                'Audio: Microphones & Conferencing': 9,
                'Network & Infrastructure': 8,
                'UC & Collaboration Devices': 7,
                'Cables & Connectors': 6
            }
        }
        
        # Get category score
        category_score = category_scores.get(room_size, {}).get(product.category, 3)
        score += category_score
        
        # Brand preference
        if brand_preference and product.brand:
            if brand_preference.lower() in product.brand.lower():
                score += 15
        
        # Budget tier matching
        if product.tier:
            if product.tier == budget_range:
                score += 10
            elif budget_range == 'premium' and product.tier in ['standard', 'premium']:
                score += 5
            elif budget_range == 'enterprise' and product.tier in ['premium', 'enterprise']:
                score += 8
        
        # Use case relevance
        if product.use_case_tags and use_case in product.use_case_tags.lower():
            score += 12
        
        # Only include products with reasonable relevance
        if score >= 8:  # Threshold for inclusion
            product.relevance_score = score
            relevant_products.append(product)
    
    # Sort by relevance score (highest first)
    relevant_products.sort(key=lambda p: p.relevance_score, reverse=True)
    
    return relevant_products

def _get_category_recommendations(builder: EnhancedLogicalSystemBuilder,
                                  matcher: IntelligentProductMatcher) -> Dict[str, List[Dict]]:
    """Get top recommended products by category."""
    
    recommendations = {}
    
    # Get room-specific quantities
    quantities = builder._get_room_based_quantities()
    brand_keywords = builder.ecosystem_manager.get_brand_keywords(
        builder.brand, builder.config.room_capacity
    )
    
    # Define category priorities and search criteria
    category_configs = {
        'UC & Collaboration Devices': {
            'keywords': brand_keywords + ['videobar', 'all-in-one', 'camera', 'controller'],
            'max_items': 3,
            'priority': 1
        },
        'Displays & Projectors': {
            'keywords': ['display', '4k', 'projector', 'interactive'],
            'max_items': 2,
            'priority': 2
        },
        'Audio: Speakers & Amplifiers': {
            'keywords': ['speaker', 'amplifier', 'ceiling', 'array'],
            'max_items': 3,
            'priority': 3
        },
        'Audio: Microphones & Conferencing': {
            'keywords': ['microphone', 'conferencing', 'array'],
            'max_items': 2,
            'priority': 4
        },
        'Control & Processing': {
            'keywords': ['processor', 'control', 'matrix'],
            'max_items': 2,
            'priority': 5
        }
    }
    
    for category, config in category_configs.items():
        # Use the matcher's find_products_by_category method
        products = matcher.find_products_by_category(category)[:config['max_items']]
        
        if products:
            recommendations[category] = [
                {
                    'id': p.id,
                    'name': p.name,
                    'brand': p.brand or 'Unknown',
                    'price': p.price,
                    'tier': p.tier,
                    'features': p.features,
                    'recommended_quantity': quantities.get(category.split(':')[0].lower().replace(' ', '_'), 1),
                    'priority': config['priority']
                }
                for p in products
            ]
    
    return recommendations


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Enhanced BOQ API v9.2", "status": "operational"}

@app.get("/api/products", response_model=List[Product])
def get_products():
    # This endpoint remains for full data access if needed
    return get_all_products_from_db()

# NEW, OPTIMIZED ENDPOINT FOR THE LIBRARY
@app.get("/api/products/library", response_model=List[ProductLibraryItem])
def get_products_for_library():
    """Get a lightweight list of products suitable for the UI library."""
    logger.info("Fetching lightweight product list for UI library.")
    with get_db_connection() as conn:
        # Only select fields needed for the library view
        products_data = conn.execute(
            'SELECT id, category, brand, name, price, tier, model FROM products'
        ).fetchall()
        return [ProductLibraryItem(**dict(row)) for row in products_data]

# --- NEW ENDPOINTS ---
@app.get("/api/products/filtered")
def get_filtered_products(
    room_size: str = Query("medium", regex="^(small|medium|large|xlarge|auditorium)$"),
    use_case: str = Query("meeting_room"),
    budget_range: str = Query("standard"),
    brand_preference: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get filtered and paginated products based on room requirements."""
    try:
        products = get_all_products_from_db()
        
        # Apply intelligent filtering
        filtered_products = _apply_smart_filtering(
            products, room_size, use_case, budget_range, brand_preference
        )
        
        # Apply additional filters
        if category:
            filtered_products = [p for p in filtered_products if p.category == category]
        
        if search:
            search_lower = search.lower()
            filtered_products = [p for p in filtered_products if 
                search_lower in p.name.lower() or 
                search_lower in (p.brand or "").lower() or
                search_lower in (p.category or "").lower()
            ]
        
        # Apply pagination
        total_count = len(filtered_products)
        paginated_products = filtered_products[offset:offset + limit]
        
        return {
            "products": [p.dict() for p in paginated_products],
            "total_count": total_count,
            "has_more": offset + limit < total_count,
            "categories": sorted(list(set(p.category for p in filtered_products if p.category)))
        }
        
    except Exception as e:
        logger.error(f"Error filtering products: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to filter products: {str(e)}")

@app.get("/api/products/recommendations")
def get_product_recommendations(
    room_dimensions: str = Query(..., description="Format: 'width,depth,height'"),
    room_size: str = Query("medium"),
    use_case: str = Query("meeting_room"),
    budget_range: str = Query("standard"),
    brand_preference: Optional[str] = Query(None)
):
    """Get AI-recommended products for specific room requirements."""
    try:
        # Parse room dimensions
        try:
            dimensions = [float(x.strip()) for x in room_dimensions.split(',')]
            if len(dimensions) != 3:
                raise ValueError("Invalid room dimensions format. Expected 'width,depth,height'.")
        except ValueError as ve:
             raise HTTPException(status_code=400, detail=str(ve))

        width, depth, height = dimensions
        
        # Create a temporary config for analysis
        config = EnhancedBoqConfig(
            room_dimensions=dimensions,
            room_capacity=room_size,
            use_case=use_case,
            budget_range=budget_range,
            brand_preference=brand_preference,
            special_requirements="",
            project_name="Temp",
            client_name="Temp"
        )
        
        products = get_all_products_from_db()
        matcher = IntelligentProductMatcher(products)
        builder = EnhancedLogicalSystemBuilder(matcher, config)
        
        # Get recommended products by category
        recommendations = _get_category_recommendations(builder, matcher)
        
        return {
            "recommendations": recommendations,
            "room_analysis": builder.room_analysis.dict(),
            "total_categories": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@app.post("/api/generate-boq")
def generate_enhanced_boq(config: EnhancedBoqConfig):
    try:
        products = get_all_products_from_db()
        if not products: raise HTTPException(status_code=500, detail="No products found in database")
        
        matcher = IntelligentProductMatcher(products)
        builder = EnhancedLogicalSystemBuilder(matcher, config)
        
        sections = builder.build_complete_system()
        if not sections: raise HTTPException(status_code=500, detail="Unable to generate BOQ - no suitable products found")
        
        subtotal = sum(section.sectionTotal for section in sections)
        total_cost = subtotal * (1 + config.contingency / 100)

        frontend_categories = {}
        for section in sections:
            if section.items:
                section_items_for_frontend = []
                for item in section.items:
                    item_dict = item.dict()
                    item_dict['unit_price'] = item_dict.pop('unitPrice')
                    item_dict['total_price'] = item_dict.pop('totalPrice')
                    section_items_for_frontend.append(item_dict)

                frontend_categories[section.name] = {
                    "items": section_items_for_frontend,
                    "total_cost": section.sectionTotal
                }
        
        compatible_response = {
            "status": "success", "boq": {
                "project_name": config.project_name, "client_name": config.client_name,
                "categories": frontend_categories, "recommendations": builder.recommendations,
            }
        }
        
        logger.info(f"Generated BOQ for {config.project_name} - Total: ${total_cost:.2f}")
        return compatible_response
        
    except Exception as e:
        logger.error(f"Error generating BOQ: {str(e)}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": f"BOQ generation failed: {str(e)}"})

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    try:
        with get_db_connection() as conn:
            product_count = conn.execute('SELECT COUNT(*) FROM products').fetchone()[0]
        return {"status": "healthy", "version": "9.2.0", "database": "connected", "product_count": product_count, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "degraded", "error": str(e), "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
