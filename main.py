# main.py (Production-Ready v4.0 - Logical System Builder)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Set
import sqlite3
import uuid
from datetime import datetime, timedelta
import random
import logging
import os
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Logical BOQ Generator API v4.0",
    description="Production-ready BOQ generation with intelligent system compatibility",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

DATABASE_FILE = 'products.db'

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Enhanced Data Models (same as before)
class Product(BaseModel):
    id: str
    name: str
    price: float = Field(gt=0, description="Product price must be positive")
    brand: Optional[str] = "Generic"
    category: Optional[str] = "Miscellaneous"
    features: Optional[str] = None
    specifications: Optional[str] = None
    warranty_years: Optional[int] = 1
    model_number: Optional[str] = None
    availability: Optional[str] = "In Stock"

class BoqConfig(BaseModel):
    projectName: str = Field(min_length=1, max_length=200)
    clientName: str = Field(min_length=1, max_length=200)
    projectLocation: Optional[str] = None
    projectDate: Optional[str] = None
    requirements: str = ""
    roomSize: Optional[str] = "medium"
    useCase: Optional[str] = "meeting_room"
    budgetRange: str
    priority: str
    contingency: float = Field(ge=0, le=50, default=15)
    
    @validator('roomSize')
    def validate_room_size(cls, v):
        valid_sizes = ['small', 'medium', 'large', 'xlarge']
        if v not in valid_sizes:
            raise ValueError(f'Room size must be one of: {valid_sizes}')
        return v

class BoqItem(BaseModel):
    id: str
    productId: str
    name: str
    brand: str
    category: str
    quantity: int = Field(gt=0)
    unit: str = "pcs"
    unitPrice: float = Field(gt=0)
    totalPrice: float = Field(gt=0)
    description: str
    features: Optional[str] = None
    specifications: Optional[str] = None
    warranty: Optional[str] = "1 Year"
    leadTime: Optional[str] = "2-3 weeks"

class BoqSection(BaseModel):
    name: str
    items: List[BoqItem]
    sectionTotal: float
    description: Optional[str] = None

class BoqSummary(BaseModel):
    subtotal: float
    contingencyAmount: float
    totalCost: float
    itemCount: int
    averageItemCost: float
    estimatedDelivery: str

class Boq(BaseModel):
    id: str
    projectName: str
    clientName: str
    generatedAt: str
    validUntil: str
    sections: List[BoqSection]
    summary: BoqSummary
    metadata: Dict[str, Any]
    recommendations: List[str] = []
    terms_conditions: List[str] = []
    assumptions: List[str] = []

# Smart Product Ecosystem Manager
class EcosystemManager:
    """Manages brand compatibility and system logic"""
    
    def __init__(self):
        # Define compatible ecosystems - this is the key intelligence
        self.ecosystems = {
            'logitech': {
                'primary_devices': ['rally bar', 'rally bar plus', 'rally bar mini', 'meetup'],
                'controllers': ['tap ip', 'tap', 'touch'],
                'accessories': ['rally mic pod', 'rally speaker', 'rally camera'],
                'compatible_brands': ['logitech'],
                'room_mapping': {
                    'small': ['meetup', 'rally bar mini'],
                    'medium': ['rally bar', 'rally bar mini'],
                    'large': ['rally bar', 'rally bar plus'],
                    'xlarge': ['rally bar plus']
                }
            },
            'poly': {
                'primary_devices': ['studio p5', 'studio p15', 'studio p21', 'studio x30', 'studio x50', 'studio x70'],
                'controllers': ['tc10', 'tc8'],
                'accessories': ['sync 20', 'sync 40', 'sync 60'],
                'compatible_brands': ['poly', 'polycom'],
                'room_mapping': {
                    'small': ['studio p5', 'studio x30'],
                    'medium': ['studio p15', 'studio x50'],
                    'large': ['studio p21', 'studio x50'],
                    'xlarge': ['studio x70']
                }
            },
            'cisco': {
                'primary_devices': ['webex room kit', 'webex room kit plus', 'webex room kit pro', 'webex board'],
                'controllers': ['touch 10', 'touch controller'],
                'accessories': ['webex quad camera', 'webex speaker'],
                'compatible_brands': ['cisco', 'webex'],
                'room_mapping': {
                    'small': ['webex room kit'],
                    'medium': ['webex room kit', 'webex room kit plus'],
                    'large': ['webex room kit plus', 'webex room kit pro'],
                    'xlarge': ['webex room kit pro', 'webex board']
                }
            },
            'microsoft': {
                'primary_devices': ['teams room', 'surface hub'],
                'controllers': ['teams touch', 'surface touch'],
                'accessories': ['teams camera', 'teams speaker'],
                'compatible_brands': ['microsoft', 'teams'],
                'room_mapping': {
                    'small': ['teams room'],
                    'medium': ['teams room', 'surface hub'],
                    'large': ['surface hub'],
                    'xlarge': ['surface hub']
                }
            }
        }
        
        # Display compatibility (most displays work with any VC system)
        self.display_brands = ['samsung', 'lg', 'sony', 'sharp', 'nec']
        
        # Universal accessories
        self.universal_accessories = ['wall mount', 'hdmi cable', 'ethernet cable', 'power cable']

    def detect_primary_brand(self, requirements: str) -> Optional[str]:
        """Detect the primary brand preference from requirements"""
        req_lower = requirements.lower()
        
        # Brand keywords with priority scoring
        brand_scores = {}
        
        for brand, config in self.ecosystems.items():
            score = 0
            if brand in req_lower:
                score += 10
            
            for device in config['primary_devices']:
                if device in req_lower:
                    score += 5
            
            for controller in config['controllers']:
                if controller in req_lower:
                    score += 3
            
            if score > 0:
                brand_scores[brand] = score
        
        return max(brand_scores, key=brand_scores.get) if brand_scores else None

    def get_compatible_ecosystem(self, brand: str) -> Dict[str, Any]:
        """Get the complete ecosystem configuration for a brand"""
        brand_lower = brand.lower()
        
        for eco_brand, config in self.ecosystems.items():
            if brand_lower in config['compatible_brands'] or brand_lower == eco_brand:
                return config
        
        return None

    def is_compatible(self, device_brand: str, controller_brand: str) -> bool:
        """Check if a device and controller are compatible"""
        device_eco = self.get_compatible_ecosystem(device_brand)
        controller_eco = self.get_compatible_ecosystem(controller_brand)
        
        if not device_eco or not controller_eco:
            return False
        
        # Same ecosystem = compatible
        return device_eco == controller_eco

# Intelligent Product Matcher
class IntelligentProductMatcher:
    def __init__(self, products: List[Product], ecosystem_manager: EcosystemManager):
        self.products = products
        self.ecosystem_manager = ecosystem_manager
        self.products_by_category = {}
        self.products_by_brand = {}
        
        # Organize products
        for product in products:
            # By category
            if product.category:
                if product.category not in self.products_by_category:
                    self.products_by_category[product.category] = []
                self.products_by_category[product.category].append(product)
            
            # By brand
            if product.brand:
                brand_lower = product.brand.lower()
                if brand_lower not in self.products_by_brand:
                    self.products_by_brand[brand_lower] = []
                self.products_by_brand[brand_lower].append(product)

    def find_primary_vc_device(self, brand: str, room_size: str) -> Optional[Product]:
        """Find the primary video conferencing device for a brand and room size"""
        ecosystem = self.ecosystem_manager.get_compatible_ecosystem(brand)
        if not ecosystem:
            logger.warning(f"No ecosystem found for brand: {brand}")
            return None
        
        # Get appropriate devices for room size
        suitable_devices = ecosystem['room_mapping'].get(room_size, [])
        
        # Find matching products
        brand_products = self.products_by_brand.get(brand.lower(), [])
        vc_products = [p for p in brand_products if p.category == 'UC & Collaboration Devices']
        
        for device_name in suitable_devices:
            for product in vc_products:
                if any(keyword in product.name.lower() for keyword in device_name.split()):
                    logger.info(f"Found primary VC device: {product.name} for {room_size} room")
                    return product
        
        # Fallback: any VC device from the brand
        if vc_products:
            logger.warning(f"Using fallback VC device for {brand}")
            return vc_products[0]
        
        return None

    def find_compatible_controller(self, primary_device: Product) -> Optional[Product]:
        """Find a controller compatible with the primary device"""
        if not primary_device:
            return None
        
        ecosystem = self.ecosystem_manager.get_compatible_ecosystem(primary_device.brand)
        if not ecosystem:
            return None
        
        # Find controllers from the same ecosystem
        brand_products = self.products_by_brand.get(primary_device.brand.lower(), [])
        controller_candidates = []
        
        for product in brand_products:
            if product.category == 'UC & Collaboration Devices':
                product_name_lower = product.name.lower()
                if any(controller in product_name_lower for controller in ecosystem['controllers']):
                    controller_candidates.append(product)
        
        if controller_candidates:
            # Prefer touch controllers
            touch_controllers = [c for c in controller_candidates if 'touch' in c.name.lower()]
            if touch_controllers:
                return touch_controllers[0]
            return controller_candidates[0]
        
        return None

    def find_compatible_accessories(self, primary_device: Product, room_size: str) -> List[Product]:
        """Find accessories compatible with the primary device"""
        if not primary_device:
            return []
        
        ecosystem = self.ecosystem_manager.get_compatible_ecosystem(primary_device.brand)
        if not ecosystem:
            return []
        
        accessories = []
        brand_products = self.products_by_brand.get(primary_device.brand.lower(), [])
        
        # For large/xlarge rooms, add expansion microphones
        if room_size in ['large', 'xlarge']:
            for product in brand_products:
                if product.category == 'UC & Collaboration Devices':
                    product_name_lower = product.name.lower()
                    if any(acc in product_name_lower for acc in ecosystem['accessories']):
                        if 'mic' in product_name_lower or 'microphone' in product_name_lower:
                            accessories.append(product)
                            break  # Only one type of expansion mic needed
        
        return accessories

    def find_display(self, room_size: str, budget_range: str) -> Optional[Product]:
        """Find an appropriate display for the room size"""
        display_products = self.products_by_category.get('Displays & Projectors', [])
        
        # Size preferences by room
        size_preferences = {
            'small': ['55', '50', '48'],
            'medium': ['65', '60', '55'],
            'large': ['75', '70', '65'],
            'xlarge': ['85', '82', '75']
        }
        
        preferred_sizes = size_preferences.get(room_size, ['65'])
        
        # Find display matching size preference
        for size in preferred_sizes:
            for product in display_products:
                if size in product.name and ('display' in product.name.lower() or 'monitor' in product.name.lower()):
                    return product
        
        # Fallback: any display
        return display_products[0] if display_products else None

    def find_mount(self) -> Optional[Product]:
        """Find a wall mount"""
        mount_products = self.products_by_category.get('Mounts, Racks & Enclosures', [])
        
        for product in mount_products:
            if 'wall' in product.name.lower() and 'mount' in product.name.lower():
                return product
        
        return mount_products[0] if mount_products else None

    def find_cables(self, cable_type: str = 'hdmi') -> Optional[Product]:
        """Find cables"""
        cable_products = self.products_by_category.get('Cables & Connectors', [])
        
        for product in cable_products:
            if cable_type.lower() in product.name.lower():
                return product
        
        return cable_products[0] if cable_products else None

# Logical System Builder
class LogicalSystemBuilder:
    def __init__(self, matcher: IntelligentProductMatcher, room_size: str, budget_range: str):
        self.matcher = matcher
        self.room_size = room_size
        self.budget_range = budget_range
        self.recommendations = []
        self.assumptions = []
        self.selected_ecosystem = None

    def _create_item(self, product: Product, qty: int = 1, custom_description: str = None) -> Optional[BoqItem]:
        if not product:
            return None
        
        description = custom_description or f"{product.brand} {product.name}"
        
        return BoqItem(
            id=str(uuid.uuid4()),
            productId=product.id,
            name=product.name,
            brand=product.brand or "Generic",
            category=product.category or "Miscellaneous",
            quantity=qty,
            unitPrice=product.price,
            totalPrice=product.price * qty,
            description=description,
            features=product.features,
            specifications=product.specifications,
            warranty=f"{product.warranty_years} Year(s)" if product.warranty_years else "1 Year",
            leadTime="2-3 weeks"
        )

    def build_logical_system(self, primary_brand: str = None) -> List[BoqSection]:
        """Build a logically consistent AV system"""
        sections = []
        
        if not primary_brand:
            # Default to Logitech if no preference
            primary_brand = 'logitech'
        
        self.selected_ecosystem = primary_brand
        logger.info(f"Building {primary_brand} ecosystem for {self.room_size} room")
        
        # 1. Video Conferencing System (Primary)
        vc_section = self._build_vc_system(primary_brand)
        if vc_section:
            sections.append(vc_section)
        
        # 2. Display & Mounting System
        display_section = self._build_display_system()
        if display_section:
            sections.append(display_section)
        
        # 3. Audio Enhancement (only for large rooms and if compatible)
        if self.room_size in ['large', 'xlarge']:
            audio_section = self._build_audio_enhancement(primary_brand)
            if audio_section:
                sections.append(audio_section)
        
        # 4. Connectivity & Infrastructure
        infra_section = self._build_infrastructure()
        if infra_section:
            sections.append(infra_section)
        
        # 5. Professional Services
        services_section = self._build_services(sections)
        if services_section:
            sections.append(services_section)
        
        return sections

    def _build_vc_system(self, brand: str) -> Optional[BoqSection]:
        """Build the core video conferencing system"""
        items = []
        
        # Find primary device
        primary_device = self.matcher.find_primary_vc_device(brand, self.room_size)
        if not primary_device:
            logger.error(f"No primary VC device found for {brand}")
            return None
        
        items.append(self._create_item(
            primary_device, 1, 
            f"{primary_device.brand} {primary_device.name} (Primary Video Conferencing System)"
        ))
        
        # Find compatible controller
        controller = self.matcher.find_compatible_controller(primary_device)
        if controller:
            items.append(self._create_item(
                controller, 1,
                f"{controller.brand} {controller.name} (Touch Controller)"
            ))
        else:
            self.assumptions.append(f"Touch controller included with {primary_device.name}")
        
        if items:
            return BoqSection(
                name="Video Conferencing System",
                items=items,
                sectionTotal=sum(item.totalPrice for item in items),
                description=f"Complete {brand.title()} video conferencing solution for {self.room_size} rooms"
            )
        
        return None

    def _build_audio_enhancement(self, brand: str) -> Optional[BoqSection]:
        """Build audio enhancement for large rooms"""
        items = []
        
        # Find expansion microphones from the same ecosystem
        accessories = self.matcher.find_compatible_accessories(
            self.matcher.find_primary_vc_device(brand, self.room_size), 
            self.room_size
        )
        
        for accessory in accessories:
            qty = 2 if self.room_size == 'xlarge' else 1
            items.append(self._create_item(
                accessory, qty,
                f"{accessory.brand} {accessory.name} (Expansion Microphones)"
            ))
        
        if items:
            return BoqSection(
                name="Audio Enhancement System",
                items=items,
                sectionTotal=sum(item.totalPrice for item in items),
                description=f"Enhanced audio coverage for {self.room_size} meeting spaces"
            )
        
        return None

    def _build_display_system(self) -> Optional[BoqSection]:
        """Build display and mounting system"""
        items = []
        
        # Find display
        display = self.matcher.find_display(self.room_size, self.budget_range)
        if display:
            items.append(self._create_item(display, 1))
        
        # Find mount
        mount = self.matcher.find_mount()
        if mount:
            items.append(self._create_item(mount, 1))
        
        # Find HDMI cables
        hdmi_cable = self.matcher.find_cables('hdmi')
        if hdmi_cable:
            cable_qty = 2  # USB-C to HDMI + backup
            items.append(self._create_item(hdmi_cable, cable_qty, "HDMI Cables (USB-C to HDMI + backup)"))
        
        if items:
            size_desc = "Professional display with mounting hardware"
            if display:
                size_match = None
                for size in ['85', '82', '75', '70', '65', '60', '55', '50']:
                    if size in display.name:
                        size_match = f'{size}" '
                        break
                if size_match:
                    size_desc = f"{size_match}{size_desc}"
            
            return BoqSection(
                name="Display & Mounting System",
                items=items,
                sectionTotal=sum(item.totalPrice for item in items),
                description=size_desc
            )
        
        return None

    def _build_infrastructure(self) -> Optional[BoqSection]:
        """Build networking and power infrastructure"""
        items = []
        
        # Network switch (if available)
        network_products = self.matcher.products_by_category.get('Networking Equipment', [])
        if network_products:
            switch = next((p for p in network_products if 'switch' in p.name.lower()), None)
            if switch:
                items.append(self._create_item(switch, 1))
        
        # UPS for premium installations
        if self.budget_range in ['premium', 'enterprise']:
            power_products = self.matcher.products_by_category.get('Power & Connectivity', [])
            if power_products:
                ups = next((p for p in power_products if 'ups' in p.name.lower()), None)
                if ups:
                    items.append(self._create_item(ups, 1))
        
        if items:
            return BoqSection(
                name="Network & Infrastructure",
                items=items,
                sectionTotal=sum(item.totalPrice for item in items),
                description="Network infrastructure and power management"
            )
        
        return None

    def _build_services(self, equipment_sections: List[BoqSection]) -> Optional[BoqSection]:
        """Build professional services section"""
        if not equipment_sections:
            return None
        
        # Calculate equipment total
        equipment_total = sum(section.sectionTotal for section in equipment_sections)
        
        # Installation cost: 15% of equipment for standard, 20% for premium
        if self.budget_range in ['premium', 'enterprise']:
            install_rate = 0.20
            service_warranty = "2 Years"
            description = "Premium installation with advanced configuration and extended support"
        else:
            install_rate = 0.15
            service_warranty = "1 Year"
            description = "Professional installation, configuration, and user training"
        
        installation_cost = equipment_total * install_rate
        
        installation_item = BoqItem(
            id=str(uuid.uuid4()),
            productId="service-installation",
            name="Professional Installation & Configuration",
            brand="Professional Services",
            category="Services",
            quantity=1,
            unit="project",
            unitPrice=installation_cost,
            totalPrice=installation_cost,
            description=description,
            features="System installation, configuration, testing, user training, documentation",
            specifications=f"Certified technicians, {service_warranty} service warranty",
            warranty=service_warranty,
            leadTime="1-2 weeks after equipment delivery"
        )
        
        return BoqSection(
            name="Professional Installation Services",
            items=[installation_item],
            sectionTotal=installation_cost,
            description="Complete installation and configuration services"
        )

    def generate_intelligent_recommendations(self) -> List[str]:
        """Generate intelligent recommendations based on the selected system"""
        recommendations = []
        
        if self.selected_ecosystem:
            ecosystem_name = self.selected_ecosystem.title()
            recommendations.append(f"Complete {ecosystem_name} ecosystem ensures optimal compatibility and performance")
            recommendations.append(f"Single-vendor solution simplifies support and maintenance")
        
        # Room-specific recommendations
        room_configs = {
            'small': "Optimized for 4-8 person huddle rooms and small meeting spaces",
            'medium': "Ideal for 8-16 person standard meeting rooms",
            'large': "Designed for 16-30 person boardrooms with enhanced audio coverage",
            'xlarge': "Enterprise solution for 30+ person large conference rooms and auditoriums"
        }
        
        if self.room_size in room_configs:
            recommendations.append(room_configs[self.room_size])
        
        # Technical recommendations
        recommendations.append("Ensure minimum 10Mbps upload/download bandwidth per concurrent video call")
        recommendations.append("Network QoS configuration recommended for optimal video quality")
        
        if self.room_size in ['large', 'xlarge']:
            recommendations.append("Consider acoustic treatment for optimal audio performance in large spaces")
        
        # Installation recommendations
        recommendations.append("Professional calibration included for optimal camera framing and audio levels")
        recommendations.append("System testing with your preferred video conferencing platform included")
        
        return recommendations

# API Endpoints (same as before, but updated generator call)
@app.get("/")
def read_root():
    return {
        "message": "Logical BOQ Generator API v4.0",
        "status": "operational",
        "features": ["Intelligent ecosystem matching", "Brand compatibility logic", "Production-ready systems"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
def health_check():
    try:
        with get_db_connection() as conn:
            conn.execute('SELECT 1').fetchone()
        return {"status": "healthy", "database": "connected", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}
        )

@app.get("/api/products", response_model=List[Product])
def get_products():
    try:
        with get_db_connection() as conn:
            products = conn.execute('SELECT * FROM products ORDER BY category, brand, name').fetchall()
            return [Product(**dict(row)) for row in products]
    except Exception as e:
        logger.error(f"Error fetching products: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/api/categories")
def get_categories():
    try:
        with get_db_connection() as conn:
            categories = conn.execute('SELECT DISTINCT category FROM products WHERE category IS NOT NULL ORDER BY category').fetchall()
            return {"categories": [row[0] for row in categories]}
    except Exception as e:
        logger.error(f"Error fetching categories: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/api/generate_boq", response_model=Boq)
def generate_boq_endpoint(config: BoqConfig):
    try:
        # Fetch products from database
        with get_db_connection() as conn:
            products_data = conn.execute('SELECT * FROM products').fetchall()
            products = [Product(**dict(p)) for p in products_data]
        
        if not products:
            raise HTTPException(status_code=400, detail="No products available in database")
        
        # Initialize intelligent systems
        ecosystem_manager = EcosystemManager()
        matcher = IntelligentProductMatcher(products, ecosystem_manager)
        builder = LogicalSystemBuilder(matcher, config.roomSize or "medium", config.budgetRange)
        
        # Detect brand preference intelligently
        primary_brand = ecosystem_manager.detect_primary_brand(config.requirements)
        if not primary_brand:
            # Default to Logitech if no clear preference
            primary_brand = 'logitech'
            logger.info("No brand preference detected, defaulting to Logitech ecosystem")
        else:
            logger.info(f"Detected brand preference: {primary_brand}")
        
        # Build logical system
        sections = builder.build_logical_system(primary_brand)
        
        if not sections:
            raise HTTPException(status_code=400, detail="Unable to generate logical BOQ with available products")
        
        # Calculate totals
        subtotal = sum(section.sectionTotal for section in sections)
        contingency_amount = subtotal * (config.contingency / 100)
        total_cost = subtotal + contingency_amount
        item_count = sum(len(section.items) for section in sections)
        average_item_cost = subtotal / item_count if item_count > 0 else 0
        
        # Generate intelligent recommendations
        recommendations = builder.generate_intelligent_recommendations()
        
        # Enhanced assumptions
        assumptions = [
            "Prices valid for 30 days from quotation date",
            "All components from single ecosystem for guaranteed compatibility",
            "Installation site has adequate power (110-240V) and network infrastructure",
            "Client provides necessary building access during installation",
            "System tested with Microsoft Teams, Zoom, and WebEx platforms",
            "User training session included with professional installation"
        ]
        
        # Professional terms
        terms_conditions = [
            "Payment Terms: 50% advance, 50% upon completion",
            "Delivery: 2-4 weeks from purchase order confirmation",
            "Warranty: Manufacturer warranty plus 1-year service coverage",
            "Support: Dedicated technical support for 12 months",
            "Installation: Certified technician installation recommended",
            "Testing: Complete system commissioning and user acceptance testing included"
        ]
        
        # Create summary
        summary = BoqSummary(
            subtotal=subtotal,
            contingencyAmount=contingency_amount,
            totalCost=total_cost,
            itemCount=item_count,
            averageItemCost=average_item_cost,
            estimatedDelivery=(datetime.now() + timedelta(weeks=3)).strftime("%Y-%m-%d")
        )
        
        # Create final BOQ
        boq = Boq(
            id=str(uuid.uuid4()),
            projectName=config.projectName,
            clientName=config.clientName,
            generatedAt=datetime.now().isoformat(),
            validUntil=(datetime.now() + timedelta(days=30)).isoformat(),
            sections=sections,
            summary=summary,
            metadata=config.dict(),
            recommendations=recommendations,
            terms_conditions=terms_conditions,
            assumptions=assumptions
        )
        
        logger.info(f"Logical BOQ generated: {primary_brand} ecosystem, {len(sections)} sections, ${total_cost:.2f}")
        return boq
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating BOQ: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Continuing from the exception handler...

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Path {request.url.path} not found",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# Additional API endpoints for enhanced functionality
@app.get("/api/brands")
def get_brands():
    """Get all available brands"""
    try:
        with get_db_connection() as conn:
            brands = conn.execute('SELECT DISTINCT brand FROM products WHERE brand IS NOT NULL ORDER BY brand').fetchall()
            return {"brands": [row[0] for row in brands]}
    except Exception as e:
        logger.error(f"Error fetching brands: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/api/products/{category}")
def get_products_by_category(category: str):
    """Get products by category"""
    try:
        with get_db_connection() as conn:
            products = conn.execute(
                'SELECT * FROM products WHERE category = ? ORDER BY brand, name', 
                (category,)
            ).fetchall()
            return {"products": [Product(**dict(row)) for row in products]}
    except Exception as e:
        logger.error(f"Error fetching products by category: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/api/ecosystems")
def get_ecosystems():
    """Get information about available ecosystems"""
    ecosystem_manager = EcosystemManager()
    return {
        "ecosystems": {
            brand: {
                "primary_devices": config["primary_devices"],
                "controllers": config["controllers"],
                "room_support": list(config["room_mapping"].keys())
            }
            for brand, config in ecosystem_manager.ecosystems.items()
        }
    }

@app.post("/api/validate_config")
def validate_config(config: BoqConfig):
    """Validate BOQ configuration before generation"""
    try:
        # Perform validation logic
        errors = []
        warnings = []
        
        # Check budget range
        valid_budgets = ['economy', 'standard', 'premium', 'enterprise']
        if config.budgetRange not in valid_budgets:
            errors.append(f"Budget range must be one of: {valid_budgets}")
        
        # Check priority
        valid_priorities = ['low', 'medium', 'high', 'urgent']
        if config.priority not in valid_priorities:
            errors.append(f"Priority must be one of: {valid_priorities}")
        
        # Check room size and use case compatibility
        if config.roomSize == 'small' and config.useCase in ['auditorium', 'large_conference']:
            warnings.append("Small room size may not be suitable for large conference use case")
        
        # Check requirements for brand detection
        ecosystem_manager = EcosystemManager()
        detected_brand = ecosystem_manager.detect_primary_brand(config.requirements)
        
        validation_result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "detected_brand": detected_brand,
            "recommendations": []
        }
        
        if detected_brand:
            validation_result["recommendations"].append(f"Detected {detected_brand.title()} ecosystem preference")
        else:
            validation_result["recommendations"].append("No specific brand preference detected, will use Logitech as default")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating config: {str(e)}")
        raise HTTPException(status_code=500, detail="Validation error")

@app.get("/api/boq/{boq_id}")
def get_boq(boq_id: str):
    """Get a specific BOQ by ID (placeholder for future database storage)"""
    # This would typically fetch from a BOQ storage table
    # For now, return a not implemented response
    raise HTTPException(
        status_code=501, 
        detail="BOQ retrieval not yet implemented - BOQs are generated on-demand"
    )

@app.delete("/api/products/{product_id}")
def delete_product(product_id: str):
    """Delete a product (admin function)"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM products WHERE id = ?', (product_id,))
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Product not found")
            
            conn.commit()
            logger.info(f"Product {product_id} deleted")
            return {"message": "Product deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting product: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

@app.put("/api/products/{product_id}")
def update_product(product_id: str, product: Product):
    """Update a product (admin function)"""
    try:
        # Ensure the ID matches
        if product.id != product_id:
            raise HTTPException(status_code=400, detail="Product ID mismatch")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE products 
                SET name=?, price=?, brand=?, category=?, features=?, specifications=?, 
                    warranty_years=?, model_number=?, availability=?
                WHERE id=?
            ''', (
                product.name, product.price, product.brand, product.category,
                product.features, product.specifications, product.warranty_years,
                product.model_number, product.availability, product_id
            ))
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Product not found")
            
            conn.commit()
            logger.info(f"Product {product_id} updated")
            return {"message": "Product updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating product: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/api/products")
def create_product(product: Product):
    """Create a new product (admin function)"""
    try:
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO products (id, name, price, brand, category, features, specifications, 
                                    warranty_years, model_number, availability)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                product.id, product.name, product.price, product.brand, product.category,
                product.features, product.specifications, product.warranty_years,
                product.model_number, product.availability
            ))
            conn.commit()
            logger.info(f"Product {product.id} created")
            return {"message": "Product created successfully", "product": product}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Product ID already exists")
    except Exception as e:
        logger.error(f"Error creating product: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

# Database initialization
def init_database():
    """Initialize the database with required tables"""
    try:
        with get_db_connection() as conn:
            # Create products table if it doesn't exist
            conn.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    price REAL NOT NULL,
                    brand TEXT,
                    category TEXT,
                    features TEXT,
                    specifications TEXT,
                    warranty_years INTEGER DEFAULT 1,
                    model_number TEXT,
                    availability TEXT DEFAULT 'In Stock',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_category ON products(category)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_brand ON products(brand)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_name ON products(name)')
            
            conn.commit()
            logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Logical BOQ Generator API v4.0")
    init_database()
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Logical BOQ Generator API v4.0")

# Development server runner
if __name__ == "__main__":
    import uvicorn
    
    # Check if running in production
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.environ.get("ENV") == "development",
        log_level="info"
    )
