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
    title="Professional BOQ Generator API v9.1 (Compatible)",
    description="Enhanced AV Room Design and BOQ Generation System with Frontend Compatibility Layer",
    version="9.1.0"
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

# --- Cached Product Loading ---
@functools.lru_cache(maxsize=None)
def get_all_products_from_db() -> List['Product']:
    """Cached function to load all products from the database."""
    logger.info("Loading and caching all products from the database...")
    with get_db_connection() as conn:
        products_data = conn.execute('SELECT * FROM products').fetchall()
        return [Product(**dict(row)) for row in products_data]

# --- Data Models (Aligned with actual DB schema) ---
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

class EnhancedBoqConfig(BaseModel):
    # Match the frontend's JSON payload structure using aliases
    project_name: str = Field(alias="projectName", default="Default Project")
    client_name: str = Field(alias="clientName", default="Default Client")
    room_dimensions: List[float] = Field(min_items=3, max_items=3)
    room_capacity: str = Field(alias="roomSize", default="medium")
    use_case: str = Field(alias="useCase", default="meeting_room")
    brand_preference: Optional[str] = Field(alias="brandPreference", default=None)
    special_requirements: str = Field(alias="requirements", default="")
    budget_range: str = Field(alias="budgetRange", default="standard")

    # Add internal properties for easier access so the rest of the logic doesn't need to change
    @property
    def roomWidth(self) -> float:
        return self.room_dimensions[0]

    @property
    def roomDepth(self) -> float:
        return self.room_dimensions[1]
    
    # Internal-only fields that are not part of the request body
    priority: str = Field(default="balanced", exclude=True)
    contingency: float = Field(default=10.0, ge=5.0, le=25.0, exclude=True)

    class Config:
        allow_population_by_field_name = True
        
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

class RoomValidation(BaseModel):
    valid: bool
    warnings: List[str]
    suggestions: List[str]
    complexity_score: int

# --- Enhanced Ecosystem Management ---
class EcosystemManager:
    def __init__(self):
        self.ecosystems = {
            'logitech': {
                'primary_keywords': ['logitech', 'rally bar', 'meetup', 'tap ip', 'brio'],
                'room_mapping': {
                    'small': ['meetup', 'brio', 'tap'],
                    'medium': ['rally bar', 'tap ip'],
                    'large': ['rally bar', 'rally camera', 'tap ip'],
                    'xlarge': ['rally camera', 'ptz pro', 'tap ip']
                }
            },
            'poly': {
                'primary_keywords': ['poly', 'polycom', 'studio x30', 'studio x50', 'tc8', 'tc10'],
                'room_mapping': {
                    'small': ['studio x30', 'tc8'],
                    'medium': ['studio x50', 'tc10'],
                    'large': ['studio x70', 'tc10'],
                    'xlarge': ['g7500', 'tc10']
                }
            },
            'cisco': {
                'primary_keywords': ['cisco', 'webex', 'room kit', 'room bar', 'touch 10'],
                'room_mapping': {
                    'small': ['room kit mini', 'touch 10'],
                    'medium': ['room bar', 'touch 10'],
                    'large': ['room kit pro', 'navigator'],
                    'xlarge': ['sx80', 'navigator']
                }
            },
            'microsoft': {
                'primary_keywords': ['microsoft', 'teams', 'surface hub'],
                'room_mapping': {
                    'small': ['teams room', 'surface hub'],
                    'medium': ['teams room pro', 'surface hub'],
                    'large': ['teams room large', 'surface hub'],
                    'xlarge': ['teams room xlarge', 'surface hub']
                }
            },
            'zoom': {
                'primary_keywords': ['zoom', 'zoom rooms'],
                'room_mapping': {
                    'small': ['zoom rooms appliance'],
                    'medium': ['zoom rooms pro'],
                    'large': ['zoom rooms enterprise'],
                    'xlarge': ['zoom rooms enterprise']
                }
            }
        }
    
    def detect_primary_brand(self, requirements: str, brand_preference: Optional[str] = None) -> str:
        if brand_preference and brand_preference in self.ecosystems:
            return brand_preference
        
        req_lower = requirements.lower()
        for brand, config in self.ecosystems.items():
            if any(keyword in req_lower for keyword in config['primary_keywords']):
                return brand
        return 'logitech'  # Default fallback
    
    def get_brand_keywords(self, brand: str, room_size: str) -> List[str]:
        ecosystem = self.ecosystems.get(brand.lower(), {})
        return ecosystem.get('room_mapping', {}).get(room_size, [])

# --- Enhanced Product Matching ---
class IntelligentProductMatcher:
    def __init__(self, products: List[Product]):
        self.products = products
        self.products_by_category = {}
        self.products_by_tier = {}
        
        # Index products for faster lookup
        for p in products:
            cat = p.category or "Other"
            if cat not in self.products_by_category:
                self.products_by_category[cat] = []
            self.products_by_category[cat].append(p)
            
            tier = p.tier or "standard"
            if tier not in self.products_by_tier:
                self.products_by_tier[tier] = []
            self.products_by_tier[tier].append(p)
    
    def find_best_product(self, category: str, keywords: List[str], 
                          brand: Optional[str] = None, tier: Optional[str] = None,
                          use_case: Optional[str] = None) -> Optional[Product]:
        """Enhanced product matching with multiple criteria"""
        candidates = self.products_by_category.get(category, [])
        
        if not candidates:
            return None
        
        scored_products = []
        
        for product in candidates:
            score = 0
            
            # Brand matching
            if brand and product.brand and brand.lower() in product.brand.lower():
                score += 50
            
            # Keyword matching in name
            for keyword in keywords:
                if keyword.lower() in product.name.lower():
                    score += 30
                if product.features and keyword.lower() in product.features.lower():
                    score += 10
            
            # Tier preference
            if tier and product.tier and tier == product.tier:
                score += 20
            
            # Use case matching
            if use_case and product.use_case_tags:
                if use_case in product.use_case_tags.lower():
                    score += 25
            
            # Quality tier preference for premium/enterprise
            if tier in ['premium', 'enterprise'] and product.tier in ['premium', 'enterprise']:
                score += 15
            
            if score > 0:
                scored_products.append((score, product))
        
        if scored_products:
            scored_products.sort(key=lambda x: x[0], reverse=True)
            return scored_products[0][1]
        
        # Fallback to first available product in category if no matches
        return candidates[0] if candidates else None
    
    def find_products_by_category(self, category: str) -> List[Product]:
        return self.products_by_category.get(category, [])

# --- Room Analysis Engine ---
class RoomAnalyzer:
    def __init__(self, config: EnhancedBoqConfig):
        self.config = config
    
    def analyze_room(self) -> RoomAnalysis:
        area = self.config.roomWidth * self.config.roomDepth
        perimeter = 2 * (self.config.roomWidth + self.config.roomDepth)
        
        # Capacity calculation based on area
        if area < 15: capacity = 4
        elif area < 30: capacity = 8
        elif area < 60: capacity = 16
        elif area < 100: capacity = 30
        else: capacity = 50
        
        # Acoustic rating
        if area < 20: acoustic = "Good"
        elif area < 50: acoustic = "Moderate"
        elif area < 100: acoustic = "Challenging"
        else: acoustic = "Complex"
        
        # Complexity score (1-10)
        complexity = 3  # Base complexity
        if area > 50: complexity += 2
        if self.config.use_case in ['auditorium', 'lecture_hall']: complexity += 3
        if self.config.budget_range == 'enterprise': complexity += 1
        if self.config.roomWidth / self.config.roomDepth > 3: complexity += 1  # Unusual aspect ratio
        
        return RoomAnalysis(
            roomArea=area,
            roomPerimeter=perimeter,
            recommendedCapacity=capacity,
            acousticRating=acoustic,
            complexityScore=min(complexity, 10)
        )

# --- Enhanced System Builder ---
class EnhancedLogicalSystemBuilder:
    def __init__(self, matcher: IntelligentProductMatcher, config: EnhancedBoqConfig):
        self.matcher = matcher
        self.config = config
        self.ecosystem_manager = EcosystemManager()
        self.room_analyzer = RoomAnalyzer(config)
        self.brand = self.ecosystem_manager.detect_primary_brand(
            config.special_requirements, 
            config.brand_preference
        )
        self.room_analysis = self.room_analyzer.analyze_room()
        self.recommendations = []

    def _create_boq_item(self, product: Product, quantity: int = 1) -> BoqItem:
        # Renamed internal fields to match BoqItem model for direct dict unpacking
        return BoqItem(
            id=str(uuid.uuid4()),
            productId=product.id,
            name=product.name,
            brand=product.brand or "Unknown",
            category=product.category,
            quantity=quantity,
            unitPrice=product.price,
            totalPrice=product.price * quantity,
            description=product.features or "",
            features=product.features,
            tier=product.tier
        )

    def _get_room_based_quantities(self) -> Dict[str, int]:
        """Calculate quantities based on room size and use case"""
        area = self.room_analysis.roomArea
        
        quantities = {
            'displays': 1, 'cameras': 1, 'speakers': 2,
            'microphones': 1, 'controllers': 1
        }
        
        if area > 30: quantities['speakers'] = 4
        if area > 60:
            quantities['speakers'] = 6
            quantities['microphones'] = 2
        if area > 100:
            quantities['displays'] = 2
            quantities['cameras'] = 2
            quantities['speakers'] = 8
            quantities['microphones'] = 3
            
        if self.config.use_case in ['auditorium', 'lecture_hall']:
            quantities['displays'] = max(2, quantities['displays'])
            quantities['microphones'] = max(4, quantities['microphones'])
            
        return quantities

    def build_collaboration_section(self) -> Optional[BoqSection]:
        items = []
        quantities = self._get_room_based_quantities()
        brand_keywords = self.ecosystem_manager.get_brand_keywords(self.brand, self.config.room_capacity)
        
        primary_device = self.matcher.find_best_product("UC & Collaboration Devices", brand_keywords + ['videobar', 'all-in-one', 'camera'], brand=self.brand, tier=self.config.budget_range, use_case=self.config.use_case)
        if primary_device:
            items.append(self._create_boq_item(primary_device, 1))
            self.recommendations.append(f"Selected {primary_device.name} as primary collaboration device for {self.config.room_capacity} {self.config.use_case}")

        controller = self.matcher.find_best_product("UC & Collaboration Devices", brand_keywords + ['controller', 'touch', 'tap'], brand=self.brand, tier=self.config.budget_range)
        if controller:
            items.append(self._create_boq_item(controller, 1))

        if quantities['cameras'] > 1:
            additional_camera = self.matcher.find_best_product("UC & Collaboration Devices", ['ptz', 'camera', '4k'], brand=self.brand, tier=self.config.budget_range)
            if additional_camera and additional_camera.id != primary_device.id:
                items.append(self._create_boq_item(additional_camera, quantities['cameras'] - 1))
        
        if items:
            return BoqSection(name="UC & Collaboration Devices", items=items, sectionTotal=sum(i.totalPrice for i in items), description=f"Video collaboration system optimized for {self.brand.title()} ecosystem")
        return None

    # ... Other build sections remain the same ...
    def build_display_section(self) -> Optional[BoqSection]:
        items, area = [], self.room_analysis.roomArea
        display_keywords = ['86"', '98"'] if area > 100 else ['75"', '86"'] if area > 50 else ['65"', '75"'] if area > 20 else ['55"']
        
        category = "Displays & Projectors"
        keywords = ['projector', '4k', 'laser'] if area > 80 or self.config.use_case in ['auditorium'] else ['display', '4k', 'interactive']
        
        display = self.matcher.find_best_product(category, keywords + display_keywords, tier=self.config.budget_range, use_case=self.config.use_case)
        if display:
            items.append(self._create_boq_item(display, self._get_room_based_quantities()['displays']))
        
        mount = self.matcher.find_best_product("Mounts, Racks & Enclosures", ['mount', 'bracket', 'wall mount'], tier=self.config.budget_range)
        if mount:
            items.append(self._create_boq_item(mount, self._get_room_based_quantities()['displays']))
        
        if items:
            return BoqSection(name="Displays & Projectors", items=items, sectionTotal=sum(i.totalPrice for i in items), description="Primary visual display system")
        return None

    def build_audio_section(self) -> Optional[BoqSection]:
        items, quantities = [], self._get_room_based_quantities()
        if quantities['speakers'] > 2:
            speaker = self.matcher.find_best_product("Audio: Speakers & Amplifiers", ['ceiling', 'speaker', 'array'], tier=self.config.budget_range, use_case=self.config.use_case)
            if speaker: items.append(self._create_boq_item(speaker, quantities['speakers']))
        if quantities['microphones'] > 1:
            mic = self.matcher.find_best_product("Audio: Microphones & Conferencing", ['microphone', 'array', 'ceiling'], brand=self.brand, tier=self.config.budget_range)
            if mic: items.append(self._create_boq_item(mic, quantities['microphones']))

        if items:
            return BoqSection(name="Audio: Speakers & Amplifiers", items=items, sectionTotal=sum(i.totalPrice for i in items), description="Professional audio system")
        return None
    
    def build_control_section(self) -> Optional[BoqSection]:
        items = []
        if self.config.room_capacity in ['medium', 'large', 'xlarge'] or self.config.budget_range in ['premium', 'enterprise']:
            processor = self.matcher.find_best_product("Control & Processing", ['processor', 'control', 'matrix'], tier=self.config.budget_range, use_case=self.config.use_case)
            if processor: items.append(self._create_boq_item(processor, 1))

        if items:
            return BoqSection(name="Control & Processing", items=items, sectionTotal=sum(i.totalPrice for i in items), description="Centralized control system")
        return None

    def build_infrastructure_section(self) -> Optional[BoqSection]:
        items = []
        switch = self.matcher.find_best_product("Network & Infrastructure", ['switch', 'managed', 'poe'], tier=self.config.budget_range)
        if switch: items.append(self._create_boq_item(switch, 1))
        
        if items:
            return BoqSection(name="Network & Infrastructure", items=items, sectionTotal=sum(i.totalPrice for i in items), description="Networking hardware")
        return None

    def build_cabling_section(self) -> Optional[BoqSection]:
        items = []
        hdmi = self.matcher.find_best_product("Cables & Connectors", ['hdmi', 'cable', '4k'], tier=self.config.budget_range)
        if hdmi: items.append(self._create_boq_item(hdmi, self._get_room_based_quantities()['displays'] * 2))
        
        ethernet = self.matcher.find_best_product("Cables & Connectors", ['ethernet', 'cat6', 'cable'], tier=self.config.budget_range)
        if ethernet: items.append(self._create_boq_item(ethernet, int(self.room_analysis.roomArea / 10)))

        if items:
            return BoqSection(name="Cables & Connectors", items=items, sectionTotal=sum(i.totalPrice for i in items), description="Professional grade cabling")
        return None

    def build_services_section(self, equipment_total: float) -> BoqSection:
        complexity_score = self.room_analysis.complexityScore
        base_rate = 0.20 + (0.05 if complexity_score > 6 else 0) + (0.05 if complexity_score > 8 else 0)
        service_cost = round(equipment_total * base_rate, 2)
        
        service_item = BoqItem(id=str(uuid.uuid4()), productId="SVC-PKG-01", name="Professional Services Package", brand="Professional Services", category="Services", quantity=1, unitPrice=service_cost, totalPrice=service_cost, description="Complete installation, integration, testing, and training services", features="Project management, certified technicians, documentation, training, 1-year warranty")
        return BoqSection(name="Professional Services", items=[service_item], sectionTotal=service_cost, description="Turnkey professional installation and integration services")

    def build_complete_system(self) -> List[BoqSection]:
        sections, equipment_total = [], 0.0
        
        hardware_sections = [
            self.build_collaboration_section(), self.build_display_section(),
            self.build_audio_section(), self.build_control_section(),
            self.build_infrastructure_section(), self.build_cabling_section()
        ]
        
        for section in hardware_sections:
            if section:
                sections.append(section)
                equipment_total += section.sectionTotal
        
        if equipment_total > 0:
            sections.append(self.build_services_section(equipment_total))
        
        self._generate_additional_recommendations()
        return sections

    def _generate_additional_recommendations(self):
        if self.room_analysis.roomArea > 60: self.recommendations.append("Large room detected - consider dual display setup for optimal visibility")
        if self.config.budget_range == 'enterprise': self.recommendations.append("Enterprise tier includes premium support and extended warranties")
        if self.config.use_case == 'boardroom': self.recommendations.append("Executive boardroom features include premium finishes and advanced presentation capabilities")
        if self.room_analysis.complexityScore > 7: self.recommendations.append("Complex installation detected - additional project management is included")

# --- Utility Functions ---
def generate_professional_terms(config: EnhancedBoqConfig, summary: BoqSummary) -> List[str]:
    # This function can remain as is
    return ["Standard terms and conditions apply."]

def generate_assumptions(config: EnhancedBoqConfig, room_analysis: RoomAnalysis) -> List[str]:
    # This function can remain as is
    return ["Standard project assumptions apply."]

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Enhanced BOQ API v9.1", "status": "operational"}

@app.get("/api/products", response_model=List[Product])
def get_products():
    return get_all_products_from_db()

@app.post("/api/generate-boq")
def generate_enhanced_boq(config: EnhancedBoqConfig):
    """Generate comprehensive BOQ with enhanced logic and frontend compatibility."""
    try:
        products = get_all_products_from_db()
        if not products:
            raise HTTPException(status_code=500, detail="No products found in database")
        
        matcher = IntelligentProductMatcher(products)
        builder = EnhancedLogicalSystemBuilder(matcher, config)
        
        sections = builder.build_complete_system()
        if not sections:
            raise HTTPException(status_code=500, detail="Unable to generate BOQ - no suitable products found")
        
        subtotal = sum(section.sectionTotal for section in sections)
        contingency_amount = round(subtotal * (config.contingency / 100), 2)
        total_cost = subtotal + contingency_amount
        item_count = sum(len(section.items) for section in sections)
        complexity_score = builder.room_analysis.complexityScore
        base_days = 2 + (1 if complexity_score > 6 else 0) + (2 if complexity_score > 8 else 0)

        summary = BoqSummary(subtotal=subtotal, contingencyAmount=contingency_amount, installationCost=0, totalCost=total_cost, itemCount=item_count, estimatedInstallDays=base_days)
        
        # --- Compatibility Transformation for Frontend ---
        frontend_categories = {}
        for section in sections:
            if section.items:
                # The frontend expects a dictionary of items, let's adapt
                section_items_for_frontend = []
                for item in section.items:
                    item_dict = item.dict()
                    # The frontend expects 'unit_price' and 'total_price'
                    item_dict['unit_price'] = item_dict.pop('unitPrice')
                    item_dict['total_price'] = item_dict.pop('totalPrice')
                    section_items_for_frontend.append(item_dict)

                frontend_categories[section.name] = {
                    "items": section_items_for_frontend,
                    "total_cost": section.sectionTotal
                }
        
        compatible_response = {
            "status": "success",
            "boq": {
                "project_name": config.project_name,
                "client_name": config.client_name,
                "categories": frontend_categories,
                "recommendations": builder.recommendations,
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
        return {"status": "healthy", "version": "9.1.0", "database": "connected", "product_count": product_count, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "degraded", "error": str(e), "timestamp": datetime.now().isoformat()}

# Other endpoints can remain as they were in the original file
# ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
