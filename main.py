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

# Enhanced Product Matcher with budget logic
class EnhancedProductMatcher(IntelligentProductMatcher):
    def __init__(self, products: List[Product], ecosystem_manager: EcosystemManager):
        super().__init__(products, ecosystem_manager)
        self.price_thresholds = {
            'economy': (0, 15000),
            'standard': (10000, 35000), 
            'premium': (30000, 75000),
            'enterprise': (70000, float('inf'))
        }
    
    def find_optimal_system(self, config: BoqConfig) -> Dict[str, List[Product]]:
        """Find optimal product mix within budget constraints"""
        budget_min, budget_max = self.price_thresholds[config.budgetRange]
        
        # Core system components (must-have)
        brand = self.detect_brand_preference(config.requirements, config.budgetRange)
        core_system = {
            'primary_vc': self.find_primary_vc_device(brand, config.roomSize),
            'controller': None,
            'display': self.find_display(config.roomSize, config.budgetRange),
            'mount': self.find_mount(),
            'cables': self.find_cables('hdmi')
        }
        
        # Set controller based on primary device
        if core_system['primary_vc']:
            core_system['controller'] = self.find_compatible_controller(core_system['primary_vc'])
        
        # Filter for products within budget
        core_products_filtered = []
        for p in core_system.values():
            if p and p.price <= budget_max:
                core_products_filtered.append(p)

        core_cost = sum(p.price for p in core_products_filtered if p)
        
        # Optional enhancements based on remaining budget
        remaining_budget = budget_max - core_cost
        enhancements = []
        
        if remaining_budget > 1000 and core_system['primary_vc']:
            if config.roomSize in ['large', 'xlarge']:
                expansion_mics = self.find_compatible_accessories(core_system['primary_vc'], config.roomSize)
                enhancements.extend(expansion_mics)
        
        if remaining_budget > 2000:
            wireless_present = self.find_wireless_presentation_system()
            if wireless_present and wireless_present.price <= remaining_budget:
                enhancements.append(wireless_present)
                
        return {
            'core': core_products_filtered,
            'enhancements': enhancements,
            'estimated_total': core_cost + sum(p.price for p in enhancements)
        }
    
    def find_wireless_presentation_system(self) -> Optional[Product]:
        """Find wireless presentation solutions"""
        wireless_products = [
            p for p in self.products 
            if 'wireless' in p.name.lower() and 
            ('present' in p.name.lower() or 'cast' in p.name.lower() or 'airplay' in p.name.lower())
        ]
        return wireless_products[0] if wireless_products else None
    
    def detect_brand_preference(self, requirements: str, budget_range: str) -> str:
        """Enhanced brand detection with fallback logic"""
        detected = self.ecosystem_manager.detect_primary_brand(requirements)
        if detected:
            return detected
            
        # Fallback based on budget
        budget_defaults = {
            'economy': 'logitech',
            'standard': 'logitech', 
            'premium': 'poly',
            'enterprise': 'cisco'
        }
        return budget_defaults.get(budget_range, 'logitech')

# Advanced System Validation for Production BOQ
class SystemValidator:
    """Validates system completeness and compatibility"""
    
    def __init__(self):
        # Essential components for different use cases
        self.essential_components = {
            'meeting_room': ['primary_vc', 'display', 'mount', 'cables'],
            'boardroom': ['primary_vc', 'controller', 'display', 'mount', 'premium_audio'],
            'auditorium': ['primary_vc', 'multiple_displays', 'premium_audio', 'wireless_mics'],
            'telepresence': ['dual_cameras', 'premium_displays', 'acoustic_treatment']
        }
        
        # Compatibility rules
        self.compatibility_rules = {
            'hdmi_4k': self._check_4k_compatibility,
            'network_requirements': self._check_network_requirements,
            'power_requirements': self._check_power_requirements
        }
    
    def validate_system(self, sections: List[BoqSection], use_case: str) -> Dict[str, Any]:
        """Comprehensive system validation"""
        all_items = []
        for section in sections:
            all_items.extend(section.items)
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'missing_components': [],
            'recommendations': [],
            'compatibility_issues': []
        }
        
        # Check essential components
        essential = self.essential_components.get(use_case, [])
        missing = self._check_missing_components(all_items, essential)
        if missing:
            validation_results['missing_components'] = missing
            validation_results['errors'].extend([f"Missing essential component: {comp}" for comp in missing])
            validation_results['is_valid'] = False
        
        # Check compatibility
        compatibility_issues = self._check_compatibility(all_items)
        if compatibility_issues:
            validation_results['compatibility_issues'] = compatibility_issues
            validation_results['warnings'].extend(compatibility_issues)
        
        # Check for redundancy
        redundant_items = self._check_redundancy(all_items)
        if redundant_items:
            validation_results['warnings'].extend([f"Potentially redundant: {item}" for item in redundant_items])
        
        # Generate smart recommendations
        recommendations = self._generate_system_recommendations(all_items, use_case)
        validation_results['recommendations'] = recommendations
        
        return validation_results
    
    def _check_missing_components(self, items: List[BoqItem], essential: List[str]) -> List[str]:
        """Check for missing essential components"""
        found_components = set()
        
        for item in items:
            item_type = self._classify_item(item)
            found_components.add(item_type)
        
        return [comp for comp in essential if comp not in found_components]
    
    def _classify_item(self, item: BoqItem) -> str:
        """Classify item type for validation"""
        name_lower = item.name.lower()
        
        if 'rally' in name_lower or 'studio' in name_lower or 'room kit' in name_lower or 'surface hub' in name_lower:
            return 'primary_vc'
        elif 'tap' in name_lower or 'touch' in name_lower:
            return 'controller'
        elif 'display' in name_lower or 'monitor' in name_lower:
            return 'display'
        elif 'mount' in name_lower:
            return 'mount'
        elif 'cable' in name_lower:
            return 'cables'
        elif 'mic' in name_lower and 'wireless' in name_lower:
            return 'wireless_mics'
        elif 'speaker' in name_lower or 'audio' in name_lower:
            return 'premium_audio'
        
        return 'accessory'
    
    def _check_compatibility(self, items: List[BoqItem]) -> List[str]:
        """Check for compatibility issues"""
        issues = []
        
        # Find primary VC device and check ecosystem consistency
        primary_vc = next((item for item in items if self._classify_item(item) == 'primary_vc'), None)
        if primary_vc:
            ecosystem_brand = primary_vc.brand.lower()
            
            # Check if controllers match ecosystem
            controllers = [item for item in items if self._classify_item(item) == 'controller']
            for controller in controllers:
                if controller.brand.lower() != ecosystem_brand:
                    issues.append(f"Controller {controller.name} may not be compatible with {primary_vc.name}")
        
        return issues
    
    def _check_redundancy(self, items: List[BoqItem]) -> List[str]:
        """Check for potentially redundant items"""
        redundant = []
        
        # Check for multiple controllers
        controllers = [item for item in items if self._classify_item(item) == 'controller']
        if len(controllers) > 1:
            redundant.extend([f"Multiple controllers: {c.name}" for c in controllers[1:]])
        
        return redundant
    
    def _generate_system_recommendations(self, items: List[BoqItem], use_case: str) -> List[str]:
        """Generate intelligent system recommendations"""
        recommendations = []
        
        total_cost = sum(item.totalPrice for item in items)
        
        if total_cost > 50000:
            recommendations.append("Consider extended warranty for high-value system")
            
        if use_case in ['boardroom', 'auditorium']:
            recommendations.append("Professional acoustic treatment recommended for optimal audio quality")
            
        # Check for missing wireless presentation
        has_wireless = any('wireless' in item.name.lower() and 'present' in item.name.lower() for item in items)
        if not has_wireless and use_case in ['meeting_room', 'boardroom']:
            recommendations.append("Consider adding wireless presentation solution for BYOD support")
            
        return recommendations
    
    def _check_4k_compatibility(self, display: BoqItem, vc_device: BoqItem) -> bool:
        """Check if 4K display is compatible with VC device"""
        display_is_4k = '4k' in display.name.lower() or 'uhd' in display.name.lower()
        vc_supports_4k = '4k' in vc_device.name.lower() or vc_device.brand.lower() in ['logitech', 'poly', 'cisco']
        return not display_is_4k or vc_supports_4k

    def _check_network_requirements(self, system: List[BoqItem]) -> bool:
        """Placeholder for network checks"""
        return True # Not implemented, but signature is here

    def _check_power_requirements(self, items: List[BoqItem]) -> bool:
        """Placeholder for power checks"""
        return True # Not implemented, but signature is here

# Logical System Builder
class LogicalSystemBuilder:
    def __init__(self, matcher: EnhancedProductMatcher, room_size: str, budget_range: str, use_case: str = "meeting_room"):
        self.matcher = matcher
        self.room_size = room_size
        self.budget_range = budget_range
        self.use_case = use_case  # Add use_case parameter
        self.recommendations = []
        self.assumptions = []
        self.selected_ecosystem = None
        
        # Define use case specific requirements
        self.use_case_configs = {
            'meeting_room': {
                'primary_focus': 'video_conferencing',
                'audio_priority': 'standard',
                'display_requirement': 'single',
                'interaction_style': 'collaborative',
                'additional_accessories': []
            },
            'boardroom': {
                'primary_focus': 'presentation',
                'audio_priority': 'high',
                'display_requirement': 'premium',
                'interaction_style': 'executive',
                'additional_accessories': ['wireless_presentation', 'premium_audio']
            },
            'huddle_room': {
                'primary_focus': 'quick_meetings',
                'audio_priority': 'standard',
                'display_requirement': 'compact',
                'interaction_style': 'informal',
                'additional_accessories': []
            },
            'training_room': {
                'primary_focus': 'education',
                'audio_priority': 'high',
                'display_requirement': 'large_display',
                'interaction_style': 'instructor_led',
                'additional_accessories': ['wireless_microphone', 'recording_equipment']
            },
            'auditorium': {
                'primary_focus': 'large_presentation',
                'audio_priority': 'premium',
                'display_requirement': 'multiple_displays',
                'interaction_style': 'broadcast',
                'additional_accessories': ['professional_audio', 'lighting', 'recording_equipment']
            },
            'telepresence': {
                'primary_focus': 'immersive_vc',
                'audio_priority': 'premium',
                'display_requirement': 'multiple_premium',
                'interaction_style': 'immersive',
                'additional_accessories': ['premium_lighting', 'acoustic_treatment']
            },
            'lecture_hall': {
                'primary_focus': 'education_large',
                'audio_priority': 'premium',
                'display_requirement': 'projector_plus_displays',
                'interaction_style': 'one_to_many',
                'additional_accessories': ['wireless_microphone', 'document_camera', 'recording_equipment']
            },
            'creative_space': {
                'primary_focus': 'collaboration',
                'audio_priority': 'standard',
                'display_requirement': 'interactive',
                'interaction_style': 'creative',
                'additional_accessories': ['digital_whiteboard', 'wireless_presentation']
            }
        }

    def get_use_case_config(self):
        """Get configuration for the current use case"""
        return self.use_case_configs.get(self.use_case, self.use_case_configs['meeting_room'])

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
        """Build a logically consistent AV system based on use case"""
        sections = []
        use_case_config = self.get_use_case_config()
        
        if not primary_brand:
            primary_brand = 'logitech'
        
        self.selected_ecosystem = primary_brand
        logger.info(f"Building {primary_brand} ecosystem for {self.room_size} room - Use case: {self.use_case}")
        
        # Build sections based on use case requirements
        if use_case_config['primary_focus'] in ['video_conferencing', 'quick_meetings', 'immersive_vc']:
            # Video conferencing focused systems
            vc_section = self._build_vc_system(primary_brand, use_case_config)
            if vc_section:
                sections.append(vc_section)
        
        elif use_case_config['primary_focus'] in ['presentation', 'education', 'large_presentation', 'education_large']:
            # Presentation focused systems
            vc_section = self._build_presentation_system(primary_brand, use_case_config)
            if vc_section:
                sections.append(vc_section)
        
        elif use_case_config['primary_focus'] == 'collaboration':
            # Collaboration focused systems
            vc_section = self._build_collaboration_system(primary_brand, use_case_config)
            if vc_section:
                sections.append(vc_section)

        # Display system - varies by use case
        display_section = self._build_display_system_by_use_case(use_case_config)
        if display_section:
            sections.append(display_section)
        
        # Audio system - varies by use case
        if use_case_config['audio_priority'] in ['high', 'premium']:
            audio_section = self._build_enhanced_audio_system(primary_brand, use_case_config)
            if audio_section:
                sections.append(audio_section)
        
        # Additional accessories based on use case
        if use_case_config['additional_accessories']:
            accessories_section = self._build_use_case_accessories(use_case_config)
            if accessories_section:
                sections.append(accessories_section)
        
        # Infrastructure
        infra_section = self._build_infrastructure_by_use_case(use_case_config)
        if infra_section:
            sections.append(infra_section)
        
        # Services
        services_section = self._build_services(sections)
        if services_section:
            sections.append(services_section)
        
        return sections

    def _build_vc_system(self, brand: str, use_case_config: dict) -> Optional[BoqSection]:
        """Build video conferencing system based on use case"""
        items = []
        
        # Find primary device
        primary_device = self.matcher.find_primary_vc_device(brand, self.room_size)
        if not primary_device:
            logger.error(f"No primary VC device found for {brand}")
            return None
        
        # Quantity based on use case
        if self.use_case == 'telepresence':
            qty = 2  # Dual camera setup for telepresence
            description = f"{primary_device.brand} {primary_device.name} (Dual Camera Telepresence Setup)"
        else:
            qty = 1
            description = f"{primary_device.brand} {primary_device.name} (Primary Video Conferencing System)"
        
        items.append(self._create_item(primary_device, qty, description))
        
        # Controller based on use case
        controller = self.matcher.find_compatible_controller(primary_device)
        if controller:
            if self.use_case in ['boardroom', 'telepresence']:
                # Premium controller for executive use
                items.append(self._create_item(
                    controller, 1,
                    f"{controller.brand} {controller.name} (Executive Touch Controller)"
                ))
            else:
                items.append(self._create_item(
                    controller, 1,
                    f"{controller.brand} {controller.name} (Touch Controller)"
                ))
        
        if items:
            section_name = "Video Conferencing System"
            if self.use_case == 'telepresence':
                section_name = "Telepresence System"
            elif self.use_case == 'huddle_room':
                section_name = "Huddle Room AV System"
            
            return BoqSection(
                name=section_name,
                items=items,
                sectionTotal=sum(item.totalPrice for item in items),
                description=f"Complete {brand.title()} solution optimized for {self.use_case.replace('_', ' ')}"
            )
        
        return None

    def _build_presentation_system(self, brand: str, use_case_config: dict) -> Optional[BoqSection]:
        """Build presentation-focused system"""
        items = []
        
        # For presentation use cases, we might need different devices
        primary_device = self.matcher.find_primary_vc_device(brand, self.room_size)
        if primary_device:
            if self.use_case in ['auditorium', 'lecture_hall']:
                description = f"{primary_device.brand} {primary_device.name} (Professional Presentation System)"
            else:
                description = f"{primary_device.brand} {primary_device.name} (Presentation & Collaboration System)"
            
            items.append(self._create_item(primary_device, 1, description))
        
        # Add presentation-specific controllers
        controller = self.matcher.find_compatible_controller(primary_device)
        if controller and self.use_case in ['training_room', 'lecture_hall']:
            items.append(self._create_item(
                controller, 1,
                f"{controller.brand} {controller.name} (Instructor Control Panel)"
            ))
        
        return BoqSection(
            name="Presentation System",
            items=items,
            sectionTotal=sum(item.totalPrice for item in items),
            description=f"Professional presentation system for {self.use_case.replace('_', ' ')}"
        ) if items else None

    def _build_collaboration_system(self, brand: str, use_case_config: dict) -> Optional[BoqSection]:
        """Build collaboration-focused system"""
        items = []
        
        primary_device = self.matcher.find_primary_vc_device(brand, self.room_size)
        if primary_device:
            items.append(self._create_item(
                primary_device, 1,
                f"{primary_device.brand} {primary_device.name} (Creative Collaboration Hub)"
            ))
        
        return BoqSection(
            name="Collaboration System",
            items=items,
            sectionTotal=sum(item.totalPrice for item in items),
            description="Interactive collaboration system with wireless connectivity"
        ) if items else None

    def _build_display_system_by_use_case(self, use_case_config: dict) -> Optional[BoqSection]:
        """Build display system based on use case requirements"""
        items = []
        
        display_req = use_case_config['display_requirement']
        
        if display_req == 'multiple_displays':
            # Multiple displays for auditorium
            display = self.matcher.find_display(self.room_size, self.budget_range)
            if display:
                items.append(self._create_item(display, 3, "Primary Display Array (3x displays)"))
                
                mount = self.matcher.find_mount()
                if mount:
                    items.append(self._create_item(mount, 3, "Display Mounts (3x units)"))
        
        elif display_req == 'multiple_premium':
            # Premium displays for telepresence
            display = self.matcher.find_display('large', 'premium')
            if display:
                items.append(self._create_item(display, 2, "Premium Telepresence Displays (2x units)"))
        
        elif display_req == 'projector_plus_displays':
            # Projector + displays for lecture hall
            display = self.matcher.find_display(self.room_size, self.budget_range)
            if display:
                items.append(self._create_item(display, 1, "Primary Display"))
            
            # Add projector (would need projector products in database)
            projector_products = self.matcher.products_by_category.get('Displays & Projectors', [])
            projector = next((p for p in projector_products if 'projector' in p.name.lower()), None)
            if projector:
                items.append(self._create_item(projector, 1, "Lecture Hall Projector"))
        
        elif display_req == 'interactive':
            # Interactive display for creative spaces
            display = self.matcher.find_display(self.room_size, self.budget_range)
            if display:
                items.append(self._create_item(display, 1, "Interactive Collaboration Display"))
        
        else:
            # Standard display
            display = self.matcher.find_display(self.room_size, self.budget_range)
            if display:
                size_descriptor = "Compact" if display_req == 'compact' else "Professional"
                items.append(self._create_item(display, 1, f"{size_descriptor} Display"))
            
            mount = self.matcher.find_mount()
            if mount:
                items.append(self._create_item(mount, 1))
        
        # Add cables based on setup complexity
        hdmi_cable = self.matcher.find_cables('hdmi')
        if hdmi_cable:
            cable_qty = 3 if 'multiple' in display_req else 2
            items.append(self._create_item(hdmi_cable, cable_qty, f"Display Cables ({cable_qty}x units)"))
        
        if items:
            return BoqSection(
                name="Display & Mounting System",
                items=items,
                sectionTotal=sum(item.totalPrice for item in items),
                description=f"Display solution optimized for {self.use_case.replace('_', ' ')}"
            )
        
        return None

    def _build_enhanced_audio_system(self, brand: str, use_case_config: dict) -> Optional[BoqSection]:
        """Build enhanced audio system for high-priority audio use cases"""
        items = []
        
        if use_case_config['audio_priority'] == 'premium':
            # Premium audio for auditorium, telepresence
            accessories = self.matcher.find_compatible_accessories(
                self.matcher.find_primary_vc_device(brand, self.room_size),  
                self.room_size
            )
            
            for accessory in accessories:
                if 'mic' in accessory.name.lower():
                    if self.use_case == 'auditorium':
                        items.append(self._create_item(accessory, 4, "Ceiling Microphone Array (4x units)"))
                    elif self.use_case == 'telepresence':
                        items.append(self._create_item(accessory, 2, "Premium Table Microphones (2x units)"))
                    else:
                        items.append(self._create_item(accessory, 2, "Enhanced Audio Pickup (2x units)"))
        
        elif use_case_config['audio_priority'] == 'high':
            # High quality audio for boardroom, training
            accessories = self.matcher.find_compatible_accessories(
                self.matcher.find_primary_vc_device(brand, self.room_size),  
                self.room_size
            )
            
            for accessory in accessories:
                if 'mic' in accessory.name.lower():
                    qty = 2 if self.use_case == 'training_room' else 1
                    items.append(self._create_item(accessory, qty, f"Enhanced Microphones ({qty}x units)"))
        
        if items:
            return BoqSection(
                name="Enhanced Audio System",
                items=items,
                sectionTotal=sum(item.totalPrice for item in items),
                description=f"Professional audio enhancement for {self.use_case.replace('_', ' ')}"
            )
        
        return None

    def _build_use_case_accessories(self, use_case_config: dict) -> Optional[BoqSection]:
        """Build accessories specific to use case"""
        items = []
        
        # This would be expanded with actual products from your database
        # For now, creating placeholder items for demonstration
        
        for accessory_type in use_case_config['additional_accessories']:
            if accessory_type == 'wireless_presentation':
                # Find wireless presentation device
                wireless_products = [p for p in self.matcher.products if 'wireless' in p.name.lower() and 'present' in p.name.lower()]
                if wireless_products:
                    items.append(self._create_item(wireless_products[0], 1, "Wireless Presentation System"))
            
            elif accessory_type == 'wireless_microphone':
                # Find wireless microphone
                mic_products = [p for p in self.matcher.products if 'wireless' in p.name.lower() and 'mic' in p.name.lower()]
                if mic_products:
                    items.append(self._create_item(mic_products[0], 1, "Wireless Instructor Microphone"))
            
            elif accessory_type == 'digital_whiteboard':
                # Digital whiteboard for creative spaces
                whiteboard_products = [p for p in self.matcher.products if 'whiteboard' in p.name.lower() or 'interactive' in p.name.lower()]
                if whiteboard_products:
                    items.append(self._create_item(whiteboard_products[0], 1, "Digital Whiteboard"))
        
        if items:
            return BoqSection(
                name="Specialized Accessories",
                items=items,
                sectionTotal=sum(item.totalPrice for item in items),
                description=f"Use case specific accessories for {self.use_case.replace('_', ' ')}"
            )
        
        return None

    def _build_infrastructure_by_use_case(self, use_case_config: dict) -> Optional[BoqSection]:
        """Build infrastructure based on use case complexity"""
        items = []
        
        # Network infrastructure varies by use case
        network_products = self.matcher.products_by_category.get('Networking Equipment', [])
        
        if self.use_case in ['auditorium', 'lecture_hall', 'telepresence']:
            # High-end infrastructure for complex use cases
            managed_switch = next((p for p in network_products if 'managed' in p.name.lower() and 'switch' in p.name.lower()), None)
            if managed_switch:
                items.append(self._create_item(managed_switch, 1, "Managed Network Switch (Professional Grade)"))
        else:
            # Standard infrastructure
            switch = next((p for p in network_products if 'switch' in p.name.lower()), None)
            if switch:
                items.append(self._create_item(switch, 1))
        
        # Power infrastructure
        if self.use_case in ['auditorium', 'telepresence'] or self.budget_range in ['premium', 'enterprise']:
            power_products = self.matcher.products_by_category.get('Power & Connectivity', [])
            if power_products:
                ups = next((p for p in power_products if 'ups' in p.name.lower()), None)
                if ups:
                    ups_description = "Enterprise UPS System" if self.use_case == 'auditorium' else "Professional UPS System"
                    items.append(self._create_item(ups, 1, ups_description))
        
        if items:
            return BoqSection(
                name="Network & Infrastructure",
                items=items,
                sectionTotal=sum(item.totalPrice for item in items),
                description=f"Infrastructure optimized for {self.use_case.replace('_', ' ')} requirements"
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
        """Generate use case specific recommendations"""
        recommendations = []
        use_case_config = self.get_use_case_config()
        
        # Base ecosystem recommendation
        if self.selected_ecosystem:
            ecosystem_name = self.selected_ecosystem.title()
            recommendations.append(f"Complete {ecosystem_name} ecosystem ensures optimal compatibility and performance")
        
        # Use case specific recommendations
        use_case_recommendations = {
            'meeting_room': [
                "Standard meeting room setup ideal for 4-16 participants",
                "Optimized for hybrid meetings with remote participants"
            ],
            'boardroom': [
                "Executive boardroom configuration with premium audio/video quality",
                "Enhanced presentation capabilities for C-level meetings",
                "Professional aesthetics suitable for client presentations"
            ],
            'huddle_room': [
                "Compact solution perfect for quick team huddles and 1-on-1s",
                "Easy to use interface for spontaneous meetings",
                "Space-efficient design maximizes room utilization"
            ],
            'training_room': [
                "Instructor-focused controls for effective training delivery",
                "Enhanced audio pickup for interactive learning sessions",
                "Recording capabilities for training documentation"
            ],
            'auditorium': [
                "Large venue solution supporting 100+ participants",
                "Professional-grade audio system with wireless microphone support",
                "Multiple display configuration for optimal viewing angles"
            ],
            'telepresence': [
                "Immersive telepresence experience with dual-camera setup",
                "Premium audio processing for natural conversation flow",
                "Dedicated lighting recommendations for optimal video quality"
            ],
            'lecture_hall': [
                "Education-optimized setup with instructor presentation tools",
                "Hybrid learning support for in-person and remote students",
                "Document camera integration for real-time content sharing"
            ],
            'creative_space': [
                "Interactive collaboration tools for creative workflows",
                "Wireless connectivity for seamless device integration",
                "Flexible configuration supporting various creative processes"
            ]
        }
        
        specific_recs = use_case_recommendations.get(self.use_case, [])
        recommendations.extend(specific_recs)
        
        # Technical recommendations based on use case complexity
        if use_case_config['audio_priority'] == 'premium':
            recommendations.append("Professional acoustic treatment recommended for optimal audio quality")
        
        if use_case_config['display_requirement'].startswith('multiple'):
            recommendations.append("Multiple display synchronization and calibration included in installation")
        
        if self.use_case in ['auditorium', 'lecture_hall']:
            recommendations.append("Consider lighting control integration for optimal presentation visibility")
            recommendations.append("Assisted listening system compatibility available upon request")
        
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
        # Use the EnhancedProductMatcher
        matcher = EnhancedProductMatcher(products, ecosystem_manager)
        # Pass use_case to the builder
        builder = LogicalSystemBuilder(
            matcher,  
            config.roomSize or "medium",  
            config.budgetRange,
            config.useCase or "meeting_room"
        )
        
        # Detect brand preference intelligently
        primary_brand = matcher.detect_brand_preference(config.requirements, config.budgetRange)
        if not primary_brand:
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
        
        # Run system validation
        validator = SystemValidator()
        validation_results = validator.validate_system(sections, config.useCase)

        # Generate intelligent recommendations
        recommendations = builder.generate_intelligent_recommendations()
        
        # Add validation recommendations to the final list
        recommendations.extend(validation_results['recommendations'])
        # Add validation warnings as recommendations
        recommendations.extend([f"Warning: {w}" for w in validation_results['warnings']])

        # Enhanced assumptions based on use case
        base_assumptions = [
            "Prices valid for 30 days from quotation date",
            "All components from single ecosystem for guaranteed compatibility",
            "Installation site has adequate power (110-240V) and network infrastructure",
            "Client provides necessary building access during installation"
        ]
        
        use_case_assumptions = {
            'auditorium': [
                "Venue has appropriate acoustic properties or acoustic treatment budget allocated",
                "Professional lighting control system integration may require additional consultation"
            ],
            'telepresence': [
                "Dedicated network bandwidth of minimum 20Mbps up/down per session",
                "Controlled lighting environment or additional lighting equipment may be required"
            ],
            'training_room': [
                "Recording and streaming capabilities may require additional network configuration",
                "Content management system integration available as optional service"
            ]
        }
        
        assumptions = base_assumptions + use_case_assumptions.get(config.useCase or "meeting_room", [])
        
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
        
        # Enhanced metadata with use case info
        enhanced_metadata = config.dict()
        enhanced_metadata['use_case_config'] = builder.get_use_case_config()
        enhanced_metadata['selected_ecosystem'] = builder.selected_ecosystem
        
        # Add validation errors to a new field in metadata
        if validation_results['errors']:
            enhanced_metadata['validation_errors'] = validation_results['errors']
            logger.error(f"BOQ generation failed validation: {validation_results['errors']}")
            raise HTTPException(status_code=400, detail="Generated BOQ failed validation checks: " + ", ".join(validation_results['errors']))
        
        # Create final BOQ
        boq = Boq(
            id=str(uuid.uuid4()),
            projectName=config.projectName,
            clientName=config.clientName,
            generatedAt=datetime.now().isoformat(),
            validUntil=(datetime.now() + timedelta(days=30)).isoformat(),
            sections=sections,
            summary=summary,
            metadata=enhanced_metadata,
            recommendations=recommendations,
            terms_conditions=terms_conditions,
            assumptions=assumptions
        )
        
        logger.info(f"Use case specific BOQ generated: {config.useCase}, {primary_brand} ecosystem, {len(sections)} sections, ${total_cost:.2f}")
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
