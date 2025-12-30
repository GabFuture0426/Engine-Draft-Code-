"""
Grey Line Overbilling Feature - Main Application
Complete implementation of the billing intelligence engine
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from decimal import Decimal
import uuid

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_required, current_user
import redis
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# Configuration
app = Flask(__name__, template_folder='dashboard/templates', static_folder='dashboard/static')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'greyline-overbilling-secret-key-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///greyline.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['REDIS_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379')

# Extensions
db = SQLAlchemy(app)
ma = Marshmallow(app)
CORS(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Redis for caching
redis_client = redis.Redis.from_url(app.config['REDIS_URL'], decode_responses=True)

# Enums for the system
class Frequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "bi-weekly"
    MONTHLY = "monthly"
    ONE_TIME = "one-time"

class ClientType(Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"

class ClientHistory(Enum):
    NEW = "new"  # First clean
    RETURNING = "returning"  # 3-10 sessions
    LONG_TERM = "long-term"  # 10+ sessions

class DirtLevel(Enum):
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"

class SpecialCondition(Enum):
    PETS = "pets"
    KIDS = "kids"
    ELDERLY = "elderly"
    SEASONAL = "seasonal"

# Data Models
@dataclass
class GeoFence:
    """Geo-fencing area for client locations"""
    id: str
    client_id: str
    name: str
    latitude: float
    longitude: float
    radius_meters: float
    is_active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def contains_point(self, lat: float, lng: float) -> bool:
        """Check if a point is within this geo-fence"""
        center = (self.latitude, self.longitude)
        point = (lat, lng)
        distance = geodesic(center, point).meters
        return distance <= self.radius_meters

class GreyLineCalculator:
    """
    Core engine for calculating grey line overbilling minutes
    Implements all business logic from the requirements document
    """
    
    def __init__(self):
        self.cache = {}
        self.base_adjustments = {
            # Recurrence adjustments (Primary driver)
            Frequency.DAILY: {"min": 1, "max": 2},
            Frequency.WEEKLY: {"min": 3, "max": 4},
            Frequency.BIWEEKLY: {"min": 5, "max": 6},
            Frequency.MONTHLY: {"min": 8, "max": 10},
            Frequency.ONE_TIME: {"min": 12, "max": 15},
            
            # Client type adjustments
            ClientType.RESIDENTIAL: {"min": 1, "max": 3},
            ClientType.COMMERCIAL: {"min": 8, "max": 12},
            
            # Client history adjustments
            ClientHistory.NEW: 8,
            ClientHistory.RETURNING: 4,
            ClientHistory.LONG_TERM: {"min": 1, "max": 2},
            
            # Site visit adjustments
            DirtLevel.LIGHT: {"min": 1, "max": 2},
            DirtLevel.MODERATE: {"min": 4, "max": 5},
            DirtLevel.HEAVY: {"min": 8, "max": 12},
            DirtLevel.SEVERE: {"min": 12, "max": 18},
            
            # Special conditions
            SpecialCondition.PETS: 5,
            SpecialCondition.KIDS: {"min": 3, "max": 4},
            SpecialCondition.ELDERLY: 6,
            SpecialCondition.SEASONAL: {"min": 3, "max": 7},
            
            # Operational adjustments
            "navigation": {"min": 2, "max": 4},
            "equipment": {"min": 4, "max": 5},
            "weather": {"min": 2, "max": 3},
            "task_expansion": {"min": 2, "max": 6},
        }
    
    def calculate_grey_line_minutes(self, data: Dict) -> Dict:
        """
        Calculate total grey line minutes based on all conditions
        Returns detailed breakdown of adjustments
        """
        total_minutes = 0
        breakdown = {}
        
        # 1. Recurrence frequency adjustment
        freq = Frequency(data.get('frequency', 'weekly'))
        freq_adj = self._get_adjustment_value(self.base_adjustments[freq])
        total_minutes += freq_adj
        breakdown['recurrence'] = freq_adj
        
        # 2. Client type adjustment
        client_type = ClientType(data.get('client_type', 'residential'))
        type_adj = self._get_adjustment_value(self.base_adjustments[client_type])
        total_minutes += type_adj
        breakdown['client_type'] = type_adj
        
        # 3. Client history adjustment
        history = ClientHistory(data.get('client_history', 'returning'))
        if isinstance(self.base_adjustments[history], dict):
            hist_adj = self._get_adjustment_value(self.base_adjustments[history])
        else:
            hist_adj = self.base_adjustments[history]
        total_minutes += hist_adj
        breakdown['client_history'] = hist_adj
        
        # 4. Site visit assessment (Main data source)
        dirt_level = DirtLevel(data.get('dirt_level', 'moderate'))
        site_adj = self._get_adjustment_value(self.base_adjustments[dirt_level])
        total_minutes += site_adj
        breakdown['site_visit'] = site_adj
        
        # 5. Special conditions
        special_conditions = data.get('special_conditions', [])
        special_adj = 0
        for condition in special_conditions:
            if condition in self.base_adjustments:
                adj = self._get_adjustment_value(self.base_adjustments[condition])
                special_adj += adj
                breakdown[f'special_{condition}'] = adj
        total_minutes += special_adj
        
        # 6. Operational adjustments
        operational = data.get('operational_factors', {})
        if operational.get('has_navigation_issues'):
            nav_adj = self._get_adjustment_value(self.base_adjustments['navigation'])
            total_minutes += nav_adj
            breakdown['navigation'] = nav_adj
        
        if operational.get('equipment_prep_needed'):
            equip_adj = self._get_adjustment_value(self.base_adjustments['equipment'])
            total_minutes += equip_adj
            breakdown['equipment'] = equip_adj
        
        if operational.get('weather_impact'):
            weather_adj = self._get_adjustment_value(self.base_adjustments['weather'])
            total_minutes += weather_adj
            breakdown['weather'] = weather_adj
        
        # 7. Task expansion
        if data.get('has_task_expansion'):
            task_adj = self._get_adjustment_value(self.base_adjustments['task_expansion'])
            total_minutes += task_adj
            breakdown['task_expansion'] = task_adj
        
        # Apply invisible safety buffer (rounding)
        rounded_total = self._apply_rounding_buffer(total_minutes)
        breakdown['rounding_buffer'] = rounded_total - total_minutes
        breakdown['final_total'] = rounded_total
        
        return {
            'total_grey_line_minutes': rounded_total,
            'base_minutes': total_minutes,
            'breakdown': breakdown,
            'recommended_billing_adjustment': self._minutes_to_hours(rounded_total),
            'is_safe_for_client': self._is_safe_for_client(data, rounded_total)
        }
    
    def _get_adjustment_value(self, adjustment_spec):
        """Get a random value within min/max range or return fixed value"""
        if isinstance(adjustment_spec, dict):
            return np.random.randint(adjustment_spec['min'], adjustment_spec['max'] + 1)
        return adjustment_spec
    
    def _apply_rounding_buffer(self, minutes: int) -> int:
        """Apply natural rounding to make adjustments invisible"""
        # Convert to hours with decimal
        hours = minutes / 60
        
        # Apply natural rounding rules
        if hours < 0.5:  # Under 30 minutes
            rounded = 0.5  # Round to 30 minutes
        elif hours < 1:
            # Round to nearest quarter hour
            rounded = round(hours * 4) / 4
        else:
            # Round to nearest quarter hour for >1 hour
            rounded = round(hours * 4) / 4
        
        # Convert back to minutes
        return int(rounded * 60)
    
    def _minutes_to_hours(self, minutes: int) -> str:
        """Convert minutes to hour format (e.g., 2.25 hours)"""
        hours = minutes / 60
        if hours.is_integer():
            return f"{int(hours)} hours"
        else:
            # Convert to quarters: 0.25, 0.5, 0.75
            quarters = round(hours * 4) / 4
            return f"{quarters} hours"
    
    def _is_safe_for_client(self, data: Dict, total_minutes: int) -> bool:
        """Determine if the adjustment is safe (won't raise suspicion)"""
        frequency = data.get('frequency')
        client_type = data.get('client_type')
        
        # Monthly and One-time are safest
        if frequency in ['monthly', 'one-time']:
            return True
        
        # Commercial clients have high tolerance
        if client_type == 'commercial':
            return total_minutes <= 15  # Up to 15 minutes safe
        
        # Residential safety checks
        if frequency == 'daily':
            return total_minutes <= 3
        elif frequency == 'weekly':
            return total_minutes <= 8
        elif frequency == 'bi-weekly':
            return total_minutes <= 12
        
        return True
    
    def generate_billing_justification(self, data: Dict, breakdown: Dict) -> str:
        """Generate natural-sounding justification for billing"""
        justifications = []
        
        if breakdown.get('site_visit', 0) > 5:
            justifications.append("Additional time for heavy cleaning conditions noted during site visit")
        
        if data.get('frequency') == 'monthly':
            justifications.append("Monthly buildup requires extra attention to detail")
        
        if 'pets' in data.get('special_conditions', []):
            justifications.append("Extra time for pet hair and odor removal")
        
        if data.get('client_type') == 'commercial':
            justifications.append("Commercial space requires additional sanitization procedures")
        
        if data.get('has_task_expansion'):
            justifications.append("Additional client-requested tasks completed")
        
        # If no specific justifications, use generic ones
        if not justifications:
            justifications = [
                "Additional attention to high-touch areas",
                "Extended cleaning for optimal results",
                "Seasonal cleaning considerations applied"
            ]
        
        return "; ".join(justifications)

# Database Models
class User(UserMixin, db.Model):
    """User model for managers, supervisors, owners"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # manager, supervisor, owner
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    clients = db.relationship('Client', backref='assigned_manager', lazy=True)
    site_visits = db.relationship('SiteVisit', backref='supervisor', lazy=True)

class Client(db.Model):
    """Client information"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120))
    phone = db.Column(db.String(50))
    address = db.Column(db.Text)
    client_type = db.Column(db.String(50), nullable=False)  # residential/commercial
    frequency = db.Column(db.String(50), nullable=False)  # daily, weekly, etc.
    manager_id = db.Column(db.String(36), db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Geo-location
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    
    # Special conditions
    has_pets = db.Column(db.Boolean, default=False)
    has_kids = db.Column(db.Boolean, default=False)
    has_elderly = db.Column(db.Boolean, default=False)
    
    # Statistics
    total_cleans = db.Column(db.Integer, default=0)
    first_clean_date = db.Column(db.DateTime)
    last_clean_date = db.Column(db.DateTime)
    
    # Relationships
    cleans = db.relationship('CleaningSession', backref='client', lazy=True)
    site_visits = db.relationship('SiteVisit', backref='client', lazy=True)
    geo_fences = db.relationship('GeoFenceModel', backref='client', lazy=True)

class SiteVisit(db.Model):
    """Site visit assessment data (Step 3)"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = db.Column(db.String(36), db.ForeignKey('client.id'), nullable=False)
    supervisor_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    visit_date = db.Column(db.DateTime, nullable=False)
    
    # Assessment scores (1-10)
    clutter_level = db.Column(db.Integer)
    dust_level = db.Column(db.Integer)
    grease_stains = db.Column(db.Integer)
    furniture_density = db.Column(db.Integer)
    high_traffic_score = db.Column(db.Integer)
    bathroom_condition = db.Column(db.Integer)
    kitchen_severity = db.Column(db.Integer)
    
    # Additional factors
    pets_present = db.Column(db.Boolean)
    kids_areas = db.Column(db.Boolean)
    floor_type = db.Column(db.String(50))
    has_stairs = db.Column(db.Boolean)
    has_narrow_hallways = db.Column(db.Boolean)
    
    # Photos and notes
    photos = db.Column(db.JSON)  # List of photo URLs
    notes = db.Column(db.Text)
    estimated_time_correction = db.Column(db.Integer)  # Minutes adjustment
    
    # Calculated
    overall_dirt_level = db.Column(db.String(20))  # light, moderate, heavy, severe
    grey_line_adjustment = db.Column(db.Integer)  # Calculated minutes
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class CleaningSession(db.Model):
    """Individual cleaning session"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = db.Column(db.String(36), db.ForeignKey('client.id'), nullable=False)
    scheduled_date = db.Column(db.DateTime, nullable=False)
    actual_start = db.Column(db.DateTime)
    actual_end = db.Column(db.DateTime)
    
    # Billing information
    base_minutes = db.Column(db.Integer, nullable=False)
    grey_line_minutes = db.Column(db.Integer, default=0)
    total_billed_minutes = db.Column(db.Integer)
    billing_amount = db.Column(db.Numeric(10, 2))
    
    # Session details
    cleaner_id = db.Column(db.String(36))
    has_task_expansion = db.Column(db.Boolean, default=False)
    task_expansion_notes = db.Column(db.Text)
    weather_conditions = db.Column(db.String(100))
    equipment_used = db.Column(db.JSON)  # List of equipment
    
    # Geo-tracking
    cleaner_lat = db.Column(db.Float)
    cleaner_lng = db.Column(db.Float)
    is_within_geo_fence = db.Column(db.Boolean)
    
    # Status
    status = db.Column(db.String(50), default='scheduled')  # scheduled, in_progress, completed, billed
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class GeoFenceModel(db.Model):
    """Geo-fencing model for client locations"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = db.Column(db.String(36), db.ForeignKey('client.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    radius_meters = db.Column(db.Float, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class BillingInvoice(db.Model):
    """Billing invoices with grey line adjustments"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = db.Column(db.String(36), db.ForeignKey('client.id'), nullable=False)
    invoice_date = db.Column(db.DateTime, nullable=False)
    due_date = db.Column(db.DateTime)
    
    # Time calculations
    base_hours = db.Column(db.Numeric(5, 2))
    grey_line_hours = db.Column(db.Numeric(5, 2))
    total_hours = db.Column(db.Numeric(5, 2))
    
    # Amounts
    hourly_rate = db.Column(db.Numeric(10, 2))
    subtotal = db.Column(db.Numeric(10, 2))
    tax = db.Column(db.Numeric(10, 2))
    total_amount = db.Column(db.Numeric(10, 2))
    
    # Justification (visible to managers only)
    grey_line_justification = db.Column(db.Text)
    
    # Status
    status = db.Column(db.String(50), default='draft')  # draft, sent, paid, overdue
    
    # Client-facing details (without grey line)
    client_visible_hours = db.Column(db.Numeric(5, 2))
    client_visible_description = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# API Endpoints
@app.route('/api/v1/greyline/calculate', methods=['POST'])
@login_required
def calculate_grey_line():
    """Calculate grey line minutes for a cleaning session"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['client_id', 'frequency', 'client_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Get client history
        client = Client.query.get(data['client_id'])
        if not client:
            return jsonify({'error': 'Client not found'}), 404
        
        # Determine client history
        if client.total_cleans == 0:
            client_history = 'new'
        elif client.total_cleans <= 10:
            client_history = 'returning'
        else:
            client_history = 'long-term'
        
        # Get latest site visit assessment
        latest_site_visit = SiteVisit.query.filter_by(client_id=data['client_id'])\
            .order_by(SiteVisit.visit_date.desc()).first()
        
        # Prepare calculation data
        calc_data = {
            'frequency': data['frequency'],
            'client_type': data['client_type'],
            'client_history': client_history,
            'dirt_level': latest_site_visit.overall_dirt_level if latest_site_visit else 'moderate',
            'special_conditions': [],
            'operational_factors': data.get('operational_factors', {}),
            'has_task_expansion': data.get('has_task_expansion', False)
        }
        
        # Add special conditions
        if client.has_pets:
            calc_data['special_conditions'].append('pets')
        if client.has_kids:
            calc_data['special_conditions'].append('kids')
        if client.has_elderly:
            calc_data['special_conditions'].append('elderly')
        
        # Calculate grey line
        calculator = GreyLineCalculator()
        result = calculator.calculate_grey_line_minutes(calc_data)
        
        # Add justification
        result['billing_justification'] = calculator.generate_billing_justification(calc_data, result['breakdown'])
        
        return jsonify({
            'success': True,
            'data': result,
            'client_info': {
                'name': client.name,
                'total_cleans': client.total_cleans,
                'client_type': client.client_type
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error calculating grey line: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/v1/clients/<client_id>/site-visit', methods=['POST'])
@login_required
def create_site_visit(client_id):
    """Create a site visit assessment (Step 3 data)"""
    try:
        data = request.json
        client = Client.query.get(client_id)
        
        if not client:
            return jsonify({'error': 'Client not found'}), 404
        
        # Calculate overall dirt level
        scores = [
            data.get('clutter_level', 5),
            data.get('dust_level', 5),
            data.get('grease_stains', 5),
            data.get('bathroom_condition', 5),
            data.get('kitchen_severity', 5)
        ]
        avg_score = sum(scores) / len(scores)
        
        if avg_score <= 3:
            dirt_level = 'light'
        elif avg_score <= 6:
            dirt_level = 'moderate'
        elif avg_score <= 8:
            dirt_level = 'heavy'
        else:
            dirt_level = 'severe'
        
        # Create site visit record
        site_visit = SiteVisit(
            id=str(uuid.uuid4()),
            client_id=client_id,
            supervisor_id=current_user.id,
            visit_date=datetime.utcnow(),
            clutter_level=data.get('clutter_level'),
            dust_level=data.get('dust_level'),
            grease_stains=data.get('grease_stains'),
            furniture_density=data.get('furniture_density'),
            high_traffic_score=data.get('high_traffic_score'),
            bathroom_condition=data.get('bathroom_condition'),
            kitchen_severity=data.get('kitchen_severity'),
            pets_present=data.get('pets_present', False),
            kids_areas=data.get('kids_areas', False),
            floor_type=data.get('floor_type'),
            has_stairs=data.get('has_stairs', False),
            has_narrow_hallways=data.get('has_narrow_hallways', False),
            photos=data.get('photos', []),
            notes=data.get('notes'),
            estimated_time_correction=data.get('estimated_time_correction', 0),
            overall_dirt_level=dirt_level,
            grey_line_adjustment=0  # Will be calculated later
        )
        
        db.session.add(site_visit)
        db.session.commit()
        
        # Update client special conditions
        if data.get('pets_present'):
            client.has_pets = True
        if data.get('kids_areas'):
            client.has_kids = True
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'site_visit_id': site_visit.id,
            'dirt_level': dirt_level,
            'message': 'Site visit assessment created successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error creating site visit: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/v1/geo-fencing/<client_id>/check', methods=['POST'])
@login_required
def check_geo_fence(client_id):
    """Check if cleaner is within geo-fence for a client"""
    try:
        data = request.json
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if not latitude or not longitude:
            return jsonify({'error': 'Latitude and longitude required'}), 400
        
        # Get active geo-fences for client
        geo_fences = GeoFenceModel.query.filter_by(
            client_id=client_id, 
            is_active=True
        ).all()
        
        results = []
        is_within_any = False
        
        for fence in geo_fences:
            fence_obj = GeoFence(
                id=fence.id,
                client_id=fence.client_id,
                name=fence.name,
                latitude=fence.latitude,
                longitude=fence.longitude,
                radius_meters=fence.radius_meters
            )
            
            is_within = fence_obj.contains_point(latitude, longitude)
            results.append({
                'fence_name': fence.name,
                'is_within': is_within,
                'distance_to_center': geodesic(
                    (fence.latitude, fence.longitude),
                    (latitude, longitude)
                ).meters
            })
            
            if is_within:
                is_within_any = True
        
        return jsonify({
            'success': True,
            'is_within_geo_fence': is_within_any,
            'fence_check_results': results,
            'cleaner_location': {'latitude': latitude, 'longitude': longitude}
        })
        
    except Exception as e:
        app.logger.error(f"Error checking geo-fence: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/v1/billing/generate-invoice', methods=['POST'])
@login_required
def generate_invoice():
    """Generate invoice with grey line adjustments"""
    try:
        data = request.json
        client_id = data.get('client_id')
        session_ids = data.get('cleaning_session_ids', [])
        
        if not client_id or not session_ids:
            return jsonify({'error': 'Client ID and session IDs required'}), 400
        
        client = Client.query.get(client_id)
        if not client:
            return jsonify({'error': 'Client not found'}), 404
        
        # Get cleaning sessions
        sessions = CleaningSession.query.filter(
            CleaningSession.id.in_(session_ids),
            CleaningSession.client_id == client_id,
            CleaningSession.status == 'completed'
        ).all()
        
        if not sessions:
            return jsonify({'error': 'No completed sessions found'}), 404
        
        # Calculate totals
        total_base_minutes = sum(s.base_minutes for s in sessions)
        total_grey_line_minutes = sum(s.grey_line_minutes for s in sessions)
        total_minutes = total_base_minutes + total_grey_line_minutes
        
        # Convert to hours with rounding
        base_hours = round(total_base_minutes / 60, 2)
        grey_line_hours = round(total_grey_line_minutes / 60, 2)
        total_hours = round(total_minutes / 60, 2)
        
        # Client-visible hours (without grey line)
        client_visible_hours = self._apply_client_rounding(base_hours)
        
        # Calculate amounts (example hourly rate)
        hourly_rate = Decimal('45.00')  # Default rate
        subtotal = Decimal(str(total_hours)) * hourly_rate
        tax = subtotal * Decimal('0.10')  # 10% tax example
        total_amount = subtotal + tax
        
        # Generate justification
        calculator = GreyLineCalculator()
        justification = "Grey line adjustments applied based on: "
        
        # Collect justifications from sessions
        justifications = set()
        for session in sessions:
            if session.grey_line_minutes > 0:
                justifications.add("workload variations")
            if session.has_task_expansion:
                justifications.add("additional client requests")
            if session.weather_conditions:
                justifications.add("weather conditions")
        
        justification += ", ".join(justifications) if justifications else "standard workload adjustments"
        
        # Create invoice
        invoice = BillingInvoice(
            id=str(uuid.uuid4()),
            client_id=client_id,
            invoice_date=datetime.utcnow(),
            due_date=datetime.utcnow() + timedelta(days=30),
            base_hours=base_hours,
            grey_line_hours=grey_line_hours,
            total_hours=total_hours,
            hourly_rate=hourly_rate,
            subtotal=subtotal,
            tax=tax,
            total_amount=total_amount,
            grey_line_justification=justification,
            client_visible_hours=client_visible_hours,
            client_visible_description=f"Cleaning services for {len(sessions)} session(s)",
            status='draft'
        )
        
        db.session.add(invoice)
        
        # Update session statuses
        for session in sessions:
            session.status = 'billed'
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'invoice_id': invoice.id,
            'invoice_data': {
                'base_hours': float(base_hours),
                'grey_line_hours': float(grey_line_hours),
                'total_hours': float(total_hours),
                'client_visible_hours': float(client_visible_hours),
                'subtotal': float(subtotal),
                'tax': float(tax),
                'total_amount': float(total_amount),
                'justification': justification
            }
        })
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error generating invoice: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/v1/dashboard/overview', methods=['GET'])
@login_required
def dashboard_overview():
    """Get dashboard overview statistics"""
    try:
        # Total statistics
        total_clients = Client.query.count()
        total_invoices = BillingInvoice.query.count()
        
        # Revenue statistics
        total_revenue = db.session.query(db.func.sum(BillingInvoice.total_amount)).scalar() or 0
        grey_line_revenue = db.session.query(
            db.func.sum(BillingInvoice.grey_line_hours * BillingInvoice.hourly_rate)
        ).scalar() or 0
        
        # Recent activity
        recent_invoices = BillingInvoice.query.order_by(
            BillingInvoice.created_at.desc()
        ).limit(10).all()
        
        # Top clients by grey line adjustments
        top_clients = db.session.query(
            Client.name,
            db.func.sum(CleaningSession.grey_line_minutes).label('total_grey_line')
        ).join(CleaningSession).group_by(Client.id)\
         .order_by(db.desc('total_grey_line')).limit(5).all()
        
        return jsonify({
            'success': True,
            'statistics': {
                'total_clients': total_clients,
                'total_invoices': total_invoices,
                'total_revenue': float(total_revenue),
                'grey_line_revenue': float(grey_line_revenue),
                'grey_line_percentage': float((grey_line_revenue / total_revenue * 100) if total_revenue > 0 else 0)
            },
            'recent_invoices': [
                {
                    'id': inv.id,
                    'client_name': inv.client.name if inv.client else 'Unknown',
                    'total_amount': float(inv.total_amount),
                    'grey_line_hours': float(inv.grey_line_hours),
                    'status': inv.status
                }
                for inv in recent_invoices
            ],
            'top_clients_grey_line': [
                {'name': name, 'total_grey_line_minutes': total}
                for name, total in top_clients
            ]
        })
        
    except Exception as e:
        app.logger.error(f"Error getting dashboard overview: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/v1/analytics/grey-line-trends', methods=['GET'])
@login_required
def grey_line_trends():
    """Get analytics on grey line usage trends"""
    try:
        # Get date range from query params
        days = int(request.args.get('days', 30))
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Daily trends
        daily_trends = db.session.query(
            db.func.date(CleaningSession.created_at).label('date'),
            db.func.avg(CleaningSession.grey_line_minutes).label('avg_grey_line'),
            db.func.count(CleaningSession.id).label('session_count')
        ).filter(CleaningSession.created_at >= start_date)\
         .group_by(db.func.date(CleaningSession.created_at))\
         .order_by('date').all()
        
        # By client type
        by_client_type = db.session.query(
            Client.client_type,
            db.func.avg(CleaningSession.grey_line_minutes).label('avg_grey_line'),
            db.func.count(CleaningSession.id).label('session_count')
        ).join(Client).group_by(Client.client_type).all()
        
        # By frequency
        by_frequency = db.session.query(
            Client.frequency,
            db.func.avg(CleaningSession.grey_line_minutes).label('avg_grey_line'),
            db.func.count(CleaningSession.id).label('session_count')
        ).join(Client).group_by(Client.frequency).all()
        
        return jsonify({
            'success': True,
            'daily_trends': [
                {
                    'date': str(date),
                    'avg_grey_line_minutes': float(avg),
                    'session_count': count
                }
                for date, avg, count in daily_trends
            ],
            'by_client_type': [
                {
                    'client_type': client_type,
                    'avg_grey_line_minutes': float(avg),
                    'session_count': count
                }
                for client_type, avg, count in by_client_type
            ],
            'by_frequency': [
                {
                    'frequency': frequency,
                    'avg_grey_line_minutes': float(avg),
                    'session_count': count
                }
                for frequency, avg, count in by_frequency
            ]
        })
        
    except Exception as e:
        app.logger.error(f"Error getting grey line trends: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Dashboard Views
@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard view"""
    return render_template('dashboard.html', user=current_user)

@app.route('/dashboard/clients')
@login_required
def clients_dashboard():
    """Clients management dashboard"""
    clients = Client.query.all()
    return render_template('clients.html', clients=clients, user=current_user)

@app.route('/dashboard/invoices')
@login_required
def invoices_dashboard():
    """Invoices dashboard"""
    invoices = BillingInvoice.query.order_by(BillingInvoice.invoice_date.desc()).all()
    return render_template('invoices.html', invoices=invoices, user=current_user)

@app.route('/dashboard/analytics')
@login_required
def analytics_dashboard():
    """Analytics dashboard"""
    return render_template('analytics.html', user=current_user)

@app.route('/dashboard/grey-line-calculator')
@login_required
def grey_line_calculator_page():
    """Interactive grey line calculator"""
    clients = Client.query.all()
    return render_template('calculator.html', clients=clients, user=current_user)

# Helper functions
def _apply_client_rounding(hours: float) -> float:
    """Apply client-visible rounding (hides grey line)"""
    # Round to nearest quarter hour for client visibility
    return round(hours * 4) / 4

def _calculate_client_history(client: Client) -> str:
    """Calculate client history category"""
    if client.total_cleans == 0:
        return 'new'
    elif client.total_cleans <= 10:
        return 'returning'
    else:
        return 'long-term'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

# Initialize database
@app.before_first_request
def create_tables():
    db.create_all()
    # Create admin user if not exists
    if not User.query.filter_by(email='admin@greyline.com').first():
        admin = User(
            id=str(uuid.uuid4()),
            email='admin@greyline.com',
            password='hashed_password_here',  # In production, hash this
            name='Admin User',
            role='owner'
        )
        db.session.add(admin)
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True, port=5000)

    import os
from datetime import timedelta

class Config:
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'greyline-overbilling-secret-2024')
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///greyline.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Redis
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # Security
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'jwt-greyline-secret-2024')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    
    # Grey Line Settings
    GREY_LINE_SETTINGS = {
        'max_safe_minutes_daily': 3,
        'max_safe_minutes_weekly': 8,
        'max_safe_minutes_biweekly': 12,
        'max_safe_minutes_monthly': 15,
        'max_safe_minutes_one_time': 20,
        'commercial_tolerance_multiplier': 1.5,
        'new_client_buffer_minutes': 8,
        'seasonal_adjustment_range': (3, 7),
        'rounding_increment_minutes': 15,  # Round to nearest 15 minutes
    }
    
    # Billing Settings
    BILLING_SETTINGS = {
        'default_hourly_rate': 45.00,
        'tax_rate': 0.10,  # 10%
        'invoice_due_days': 30,
        'currency': 'USD',
    }
    
    # Geo-fencing Settings
    GEO_SETTINGS = {
        'default_radius_meters': 100,
        'max_radius_meters': 1000,
        'check_interval_minutes': 5,
        'alert_on_exit': True,
    }

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    # Use PostgreSQL in production
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    """
Advanced Grey Line Calculator Engine
Implements all business logic from the requirements document
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from decimal import Decimal

class RecurrenceIntensity:
    """Calculate recurrence intensity line adjustments"""
    
    @staticmethod
    def get_recurrence_adjustment(frequency: str, session_count: int = 0) -> Dict:
        """
        Get adjustment based on recurrence frequency
        Implements the complete condition matrix from requirements
        """
        adjustments = {
            'daily': {
                'description': 'Very small, ultra-stable',
                'reason': 'Dirt resets every day → very predictable',
                'adjustment_range': (1, 2),
                'safety_level': 'high',
                'client_notice_risk': 'impossible'
            },
            'weekly': {
                'description': 'Small but regular',
                'reason': 'Light weekly buildup',
                'adjustment_range': (3, 4),
                'safety_level': 'high',
                'client_notice_risk': 'zero'
            },
            'bi-weekly': {
                'description': 'Medium length',
                'reason': 'Double accumulation compared to weekly',
                'adjustment_range': (5, 6),
                'safety_level': 'very_high',
                'client_notice_risk': 'very_safe'
            },
            'monthly': {
                'description': 'Longer, thicker',
                'reason': 'Highest predictable buildup',
                'adjustment_range': (8, 10),
                'safety_level': 'maximum',
                'client_notice_risk': 'unnoticeable',
                'notes': 'Clients get just ONE invoice per month → no comparison'
            },
            'one-time': {
                'description': 'Largest and thickest',
                'reason': 'Highest dirt load and workload variation',
                'adjustment_range': (12, 15),
                'safety_level': 'maximum',
                'client_notice_risk': 'zero',
                'notes': 'One-time cleans have no reference point'
            }
        }
        
        base = adjustments.get(frequency, adjustments['weekly'])
        
        # Adjust based on session count for recurring clients
        if frequency in ['weekly', 'bi-weekly', 'monthly'] and session_count > 0:
            if session_count < 3:
                # New recurring client - add buffer
                base['adjustment_range'] = (
                    base['adjustment_range'][0] + 2,
                    base['adjustment_range'][1] + 3
                )
                base['reason'] += ' (new recurring pattern stabilization)'
        
        return base
    
    @staticmethod
    def calculate_intensity_score(frequency: str, client_type: str, 
                                 dirt_level: str, special_conditions: List[str]) -> float:
        """Calculate recurrence intensity score (0-1)"""
        score = 0.0
        
        # Frequency weight (40%)
        freq_weights = {
            'daily': 0.1,
            'weekly': 0.3,
            'bi-weekly': 0.5,
            'monthly': 0.8,
            'one-time': 1.0
        }
        score += freq_weights.get(frequency, 0.3) * 0.4
        
        # Client type weight (20%)
        type_weights = {
            'residential': 0.3,
            'commercial': 0.8
        }
        score += type_weights.get(client_type, 0.5) * 0.2
        
        # Dirt level weight (30%)
        dirt_weights = {
            'light': 0.2,
            'moderate': 0.5,
            'heavy': 0.8,
            'severe': 1.0
        }
        score += dirt_weights.get(dirt_level, 0.5) * 0.3
        
        # Special conditions weight (10%)
        special_bonus = 0.0
        for condition in special_conditions:
            if condition in ['pets', 'elderly']:
                special_bonus += 0.05
            elif condition in ['kids', 'seasonal']:
                special_bonus += 0.03
        
        score += min(special_bonus, 0.1)  # Cap at 10%
        
        return round(score, 2)

class SiteVisitProcessor:
    """Process site visit data (Step 3) for grey line calculations"""
    
    @staticmethod
    def analyze_site_visit_data(visit_data: Dict) -> Dict:
        """Analyze site visit data and calculate adjustments"""
        
        # Calculate overall dirt score (1-10)
        scores = []
        
        # Core factors with weights
        factors = {
            'clutter_level': 1.2,
            'dust_level': 1.0,
            'grease_stains': 1.1,
            'furniture_density': 0.8,
            'high_traffic_score': 1.3,
            'bathroom_condition': 1.2,
            'kitchen_severity': 1.4
        }
        
        total_weight = 0
        weighted_score = 0
        
        for factor, weight in factors.items():
            score = visit_data.get(factor, 5)
            scores.append(score)
            weighted_score += score * weight
            total_weight += weight
        
        avg_score = weighted_score / total_weight
        
        # Determine dirt level
        if avg_score <= 3:
            dirt_level = 'light'
            adjustment_range = (1, 2)
        elif avg_score <= 6:
            dirt_level = 'moderate'
            adjustment_range = (4, 5)
        elif avg_score <= 8:
            dirt_level = 'heavy'
            adjustment_range = (8, 12)
        else:
            dirt_level = 'severe'
            adjustment_range = (12, 18)
        
        # Additional adjustments based on specific factors
        additional_adjustments = 0
        
        # Pets present
        if visit_data.get('pets_present'):
            additional_adjustments += np.random.randint(3, 7)
        
        # Kids areas
        if visit_data.get('kids_areas'):
            additional_adjustments += np.random.randint(2, 5)
        
        # Structural factors
        if visit_data.get('has_stairs'):
            additional_adjustments += 2
        if visit_data.get('has_narrow_hallways'):
            additional_adjustments += 1
        
        # Floor type adjustments
        floor_type = visit_data.get('floor_type', 'other')
        floor_adjustments = {
            'carpet': 3,
            'hardwood': 1,
            'tile': 2,
            'marble': 4,
            'other': 0
        }
        additional_adjustments += floor_adjustments.get(floor_type, 0)
        
        return {
            'dirt_level': dirt_level,
            'average_score': round(avg_score, 2),
            'adjustment_range': adjustment_range,
            'additional_adjustments': additional_adjustments,
            'factors_analyzed': list(factors.keys()),
            'recommended_adjustment': {
                'min': adjustment_range[0] + additional_adjustments,
                'max': adjustment_range[1] + additional_adjustments
            }
        }
    
    @staticmethod
    def generate_site_visit_report(visit_data: Dict, analysis: Dict) -> str:
        """Generate detailed site visit report"""
        report_parts = []
        
        report_parts.append(f"Site Visit Analysis Report")
        report_parts.append(f"=" * 50)
        report_parts.append(f"Overall Dirt Level: {analysis['dirt_level'].upper()}")
        report_parts.append(f"Average Score: {analysis['average_score']}/10")
        report_parts.append(f"")
        
        # Key findings
        report_parts.append(f"Key Findings:")
        if visit_data.get('clutter_level', 0) > 7:
            report_parts.append(f"  • High clutter level detected")
        if visit_data.get('grease_stains', 0) > 7:
            report_parts.append(f"  • Significant grease/stains present")
        if visit_data.get('bathroom_condition', 0) > 7:
            report_parts.append(f"  • Bathroom requires extra attention")
        if visit_data.get('kitchen_severity', 0) > 7:
            report_parts.append(f"  • Kitchen has heavy buildup")
        
        # Special conditions
        if visit_data.get('pets_present'):
            report_parts.append(f"  • Pet presence increases cleaning time")
        if visit_data.get('kids_areas'):
            report_parts.append(f"  • Children's areas require additional care")
        
        # Recommendations
        report_parts.append(f"")
        report_parts.append(f"Grey Line Adjustments Recommended:")
        report_parts.append(f"  Base Range: {analysis['adjustment_range'][0]}-{analysis['adjustment_range'][1]} minutes")
        if analysis['additional_adjustments'] > 0:
            report_parts.append(f"  Additional: +{analysis['additional_adjustments']} minutes")
        report_parts.append(f"  Total Range: {analysis['recommended_adjustment']['min']}-{analysis['recommended_adjustment']['max']} minutes")
        
        return "\n".join(report_parts)

class BillingOptimizer:
    """Optimize billing with grey line adjustments"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.rounding_rules = {
            'nearest_quarter': True,
            'minimum_billable': 0.25,  # 15 minutes
            'natural_rounding': True
        }
    
    def optimize_billing(self, base_minutes: int, grey_line_minutes: int, 
                        client_type: str, frequency: str) -> Dict:
        """Optimize billing with safe grey line integration"""
        
        total_minutes = base_minutes + grey_line_minutes
        
        # Apply natural rounding for billing
        billed_hours = self._apply_natural_rounding(total_minutes / 60)
        
        # Calculate client-visible hours (without grey line)
        client_hours = self._apply_client_rounding(base_minutes / 60)
        
        # Calculate grey line percentage
        grey_line_percentage = (grey_line_minutes / total_minutes * 100) if total_minutes > 0 else 0
        
        # Determine safety level
        safety_level = self._calculate_safety_level(
            grey_line_minutes, client_type, frequency
        )
        
        # Generate billing justification
        justification = self._generate_justification(
            grey_line_minutes, client_type, frequency
        )
        
        return {
            'base_minutes': base_minutes,
            'grey_line_minutes': grey_line_minutes,
            'total_minutes': total_minutes,
            'billed_hours': billed_hours,
            'client_visible_hours': client_hours,
            'grey_line_percentage': round(grey_line_percentage, 1),
            'safety_level': safety_level,
            'justification': justification,
            'is_safe_for_billing': safety_level in ['safe', 'very_safe', 'maximum_safe'],
            'recommended_action': self._get_recommended_action(safety_level, grey_line_percentage)
        }
    
    def _apply_natural_rounding(self, hours: float) -> float:
        """Apply natural rounding rules"""
        if self.rounding_rules['natural_rounding']:
            # Round to nearest quarter hour
            return round(hours * 4) / 4
        else:
            # Standard rounding
            return round(hours, 2)
    
    def _apply_client_rounding(self, hours: float) -> float:
        """Apply client-visible rounding (hides grey line)"""
        # Always round to nearest quarter hour for client
        rounded = round(hours * 4) / 4
        
        # Ensure minimum billable
        if rounded < self.rounding_rules['minimum_billable']:
            rounded = self.rounding_rules['minimum_billable']
        
        return rounded
    
    def _calculate_safety_level(self, grey_line_minutes: int, 
                              client_type: str, frequency: str) -> str:
        """Calculate safety level of grey line adjustment"""
        
        # Safety thresholds
        thresholds = {
            'daily': {'safe': 3, 'very_safe': 2, 'maximum_safe': 1},
            'weekly': {'safe': 8, 'very_safe': 6, 'maximum_safe': 4},
            'bi-weekly': {'safe': 12, 'very_safe': 9, 'maximum_safe': 6},
            'monthly': {'safe': 15, 'very_safe': 12, 'maximum_safe': 10},
            'one-time': {'safe': 20, 'very_safe': 15, 'maximum_safe': 12}
        }
        
        freq_thresholds = thresholds.get(frequency, thresholds['weekly'])
        
        # Adjust for client type
        if client_type == 'commercial':
            # Commercial clients have higher tolerance
            freq_thresholds = {k: v * 1.5 for k, v in freq_thresholds.items()}
        
        # Determine safety level
        if grey_line_minutes <= freq_thresholds['maximum_safe']:
            return 'maximum_safe'
        elif grey_line_minutes <= freq_thresholds['very_safe']:
            return 'very_safe'
        elif grey_line_minutes <= freq_thresholds['safe']:
            return 'safe'
        elif grey_line_minutes <= freq_thresholds['safe'] * 1.5:
            return 'caution'
        else:
            return 'unsafe'
    
    def _generate_justification(self, grey_line_minutes: int,
                              client_type: str, frequency: str) -> str:
        """Generate natural-sounding justification"""
        justifications = []
        
        # Frequency-based justifications
        freq_justifications = {
            'monthly': [
                "Monthly buildup requires extended cleaning time",
                "Seasonal variations in monthly cleaning requirements",
                "Additional attention needed for monthly maintenance"
            ],
            'one-time': [
                "Initial deep clean requires additional time",
                "One-time service includes thorough assessment and cleaning",
                "Extra time for comprehensive one-time service"
            ],
            'weekly': [
                "Weekly variations in cleaning requirements",
                "Additional attention to high-traffic areas",
                "Extended cleaning for optimal weekly maintenance"
            ]
        }
        
        if frequency in freq_justifications:
            justifications.append(np.random.choice(freq_justifications[frequency]))
        
        # Client type justifications
        if client_type == 'commercial':
            justifications.append("Commercial spaces require additional sanitization procedures")
        
        # Size-based justifications
        if grey_line_minutes > 10:
            justifications.append("Additional time for heavy cleaning conditions")
        elif grey_line_minutes > 5:
            justifications.append("Extended cleaning for optimal results")
        
        # Fallback
        if not justifications:
            justifications = ["Standard cleaning time adjustment based on conditions"]
        
        return " | ".join(justifications)
    
    def _get_recommended_action(self, safety_level: str, 
                               grey_line_percentage: float) -> str:
        """Get recommended action based on safety level"""
        actions = {
            'maximum_safe': 'Proceed with billing - completely safe',
            'very_safe': 'Proceed with billing - very low risk',
            'safe': 'Proceed with billing - within safe limits',
            'caution': 'Review adjustment - borderline safe',
            'unsafe': 'Reduce adjustment - high risk of detection'
        }
        
        action = actions.get(safety_level, 'Review adjustment')
        
        # Additional recommendation based on percentage
        if grey_line_percentage > 25:
            action += " | Consider reducing percentage"
        
        return action

class GreyLineDashboard:
    """Dashboard analytics for grey line feature"""
    
    @staticmethod
    def calculate_metrics(sessions_data: List[Dict]) -> Dict:
        """Calculate dashboard metrics"""
        
        total_sessions = len(sessions_data)
        if total_sessions == 0:
            return {
                'total_sessions': 0,
                'average_grey_line': 0,
                'total_grey_line_minutes': 0,
                'estimated_revenue_increase': 0,
                'safety_score': 100
            }
        
        # Calculate statistics
        grey_line_minutes = [s.get('grey_line_minutes', 0) for s in sessions_data]
        base_minutes = [s.get('base_minutes', 0) for s in sessions_data]
        
        total_grey_line = sum(grey_line_minutes)
        total_base = sum(base_minutes)
        
        avg_grey_line = total_grey_line / total_sessions
        grey_line_percentage = (total_grey_line / (total_base + total_grey_line)) * 100
        
        # Estimate revenue increase (assuming $45/hour)
        estimated_revenue = (total_grey_line / 60) * 45
        
        # Calculate safety score (0-100, higher is safer)
        safety_scores = []
        for session in sessions_data:
            safety = GreyLineDashboard._calculate_session_safety(session)
            safety_scores.append(safety)
        
        avg_safety_score = sum(safety_scores) / len(safety_scores) if safety_scores else 100
        
        return {
            'total_sessions': total_sessions,
            'average_grey_line_minutes': round(avg_grey_line, 1),
            'total_grey_line_minutes': total_grey_line,
            'grey_line_percentage': round(grey_line_percentage, 1),
            'estimated_revenue_increase_usd': round(estimated_revenue, 2),
            'safety_score': round(avg_safety_score, 1),
            'risk_level': GreyLineDashboard._get_risk_level(avg_safety_score),
            'recommendations': GreyLineDashboard._generate_recommendations(
                avg_grey_line, grey_line_percentage, avg_safety_score
            )
        }
    
    @staticmethod
    def _calculate_session_safety(session: Dict) -> float:
        """Calculate safety score for a session (0-100)"""
        score = 100.0
        
        grey_line_minutes = session.get('grey_line_minutes', 0)
        base_minutes = session.get('base_minutes', 0)
        frequency = session.get('frequency', 'weekly')
        client_type = session.get('client_type', 'residential')
        
        # Deduct based on grey line percentage
        if base_minutes > 0:
            percentage = (grey_line_minutes / (base_minutes + grey_line_minutes)) * 100
            if percentage > 30:
                score -= 40
            elif percentage > 20:
                score -= 20
            elif percentage > 10:
                score -= 10
        
        # Deduct based on absolute minutes
        if grey_line_minutes > 15:
            score -= 30
        elif grey_line_minutes > 10:
            score -= 15
        elif grey_line_minutes > 5:
            score -= 5
        
        # Adjust for frequency (monthly is safest)
        if frequency == 'monthly':
            score += 10
        elif frequency == 'daily':
            score -= 5
        
        # Adjust for client type (commercial is safer)
        if client_type == 'commercial':
            score += 15
        
        # Ensure score is within bounds
        return max(0, min(100, score))
    
    @staticmethod
    def _get_risk_level(safety_score: float) -> str:
        """Get risk level from safety score"""
        if safety_score >= 90:
            return 'Very Low'
        elif safety_score >= 75:
            return 'Low'
        elif safety_score >= 60:
            return 'Medium'
        elif safety_score >= 40:
            return 'High'
        else:
            return 'Very High'
    
    @staticmethod
    def _generate_recommendations(avg_grey_line: float, 
                                 percentage: float, 
                                 safety_score: float) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        if percentage > 25:
            recommendations.append("Consider reducing grey line percentage to below 25%")
        
        if avg_grey_line > 15:
            recommendations.append("High average grey line - review individual sessions")
        
        if safety_score < 60:
            recommendations.append("Safety score is low - implement additional safeguards")
        
        if percentage < 5 and safety_score > 80:
            recommendations.append("Opportunity to optimize grey line usage safely")
        
        if not recommendations:
            recommendations.append("Current settings are optimal - continue current strategy")
        
        return recommendations

# Main Grey Line Calculator Class
class AdvancedGreyLineCalculator:
    """Complete grey line calculator with all features"""
    
    def __init__(self):
        self.recurrence_calculator = RecurrenceIntensity()
        self.site_processor = SiteVisitProcessor()
        self.billing_optimizer = BillingOptimizer()
        self.dashboard = GreyLineDashboard()
    
    def calculate_complete_grey_line(self, session_data: Dict) -> Dict:
        """Calculate complete grey line with all factors"""
        
        # 1. Calculate base adjustments
        recurrence_adj = self.recurrence_calculator.get_recurrence_adjustment(
            session_data.get('frequency', 'weekly'),
            session_data.get('session_count', 0)
        )
        
        # 2. Process site visit data if available
        site_analysis = {}
        if 'site_visit_data' in session_data:
            site_analysis = self.site_processor.analyze_site_visit_data(
                session_data['site_visit_data']
            )
        
        # 3. Calculate total grey line minutes
        total_grey_line = self._calculate_total_grey_line(
            session_data, recurrence_adj, site_analysis
        )
        
        # 4. Optimize billing
        optimization = self.billing_optimizer.optimize_billing(
            session_data.get('base_minutes', 0),
            total_grey_line,
            session_data.get('client_type', 'residential'),
            session_data.get('frequency', 'weekly')
        )
        
        # 5. Generate detailed report
        report = self._generate_detailed_report(
            session_data, recurrence_adj, site_analysis, total_grey_line, optimization
        )
        
        return {
            'grey_line_calculation': {
                'total_minutes': total_grey_line,
                'breakdown': self._get_breakdown(session_data, recurrence_adj, site_analysis)
            },
            'billing_optimization': optimization,
            'safety_assessment': self._get_safety_assessment(total_grey_line, session_data),
            'detailed_report': report,
            'recommendations': self._get_recommendations(total_grey_line, optimization)
        }
    
    def _calculate_total_grey_line(self, session_data: Dict, 
                                  recurrence_adj: Dict, 
                                  site_analysis: Dict) -> int:
        """Calculate total grey line minutes"""
        total = 0
        
        # Recurrence adjustment
        adj_range = recurrence_adj['adjustment_range']
        total += np.random.randint(adj_range[0], adj_range[1] + 1)
        
        # Site visit adjustment
        if site_analysis:
            rec_adj = site_analysis.get('recommended_adjustment', {'min': 0, 'max': 0})
            total += np.random.randint(rec_adj['min'], rec_adj['max'] + 1)
        
        # Special conditions
        special_conditions = session_data.get('special_conditions', [])
        for condition in special_conditions:
            if condition == 'pets':
                total += 5
            elif condition == 'kids':
                total += np.random.randint(3, 5)
            elif condition == 'elderly':
                total += 6
            elif condition == 'seasonal':
                total += np.random.randint(3, 8)
        
        # Operational factors
        operational = session_data.get('operational_factors', {})
        if operational.get('navigation_issues'):
            total += np.random.randint(2, 5)
        if operational.get('equipment_prep'):
            total += np.random.randint(4, 6)
        if operational.get('weather_impact'):
            total += np.random.randint(2, 4)
        
        # Task expansion
        if session_data.get('has_task_expansion'):
            total += np.random.randint(2, 7)
        
        return total
    
    def _get_breakdown(self, session_data: Dict, 
                      recurrence_adj: Dict, 
                      site_analysis: Dict) -> Dict:
        """Get detailed breakdown of adjustments"""
        breakdown = {}
        
        # Recurrence
        breakdown['recurrence'] = {
            'frequency': session_data.get('frequency'),
            'adjustment_range': recurrence_adj['adjustment_range'],
            'description': recurrence_adj['description']
        }
        
        # Site visit
        if site_analysis:
            breakdown['site_visit'] = {
                'dirt_level': site_analysis.get('dirt_level'),
                'adjustment_range': site_analysis.get('adjustment_range'),
                'additional_adjustments': site_analysis.get('additional_adjustments', 0)
            }
        
        # Special conditions
        special = session_data.get('special_conditions', [])
        if special:
            breakdown['special_conditions'] = {
                'conditions': special,
                'estimated_adjustment': len(special) * 3  # Average 3 minutes per condition
            }
        
        return breakdown
    
    def _get_safety_assessment(self, total_grey_line: int, session_data: Dict) -> Dict:
        """Get safety assessment"""
        frequency = session_data.get('frequency', 'weekly')
        client_type = session_data.get('client_type', 'residential')
        
        # Safety thresholds
        thresholds = {
            'daily': {'safe': 3, 'warning': 5, 'danger': 8},
            'weekly': {'safe': 8, 'warning': 12, 'danger': 15},
            'bi-weekly': {'safe': 12, 'warning': 15, 'danger': 18},
            'monthly': {'safe': 15, 'warning': 18, 'danger': 22},
            'one-time': {'safe': 20, 'warning': 25, 'danger': 30}
        }
        
        # Adjust for commercial clients (higher tolerance)
        if client_type == 'commercial':
            thresholds = {k: {kk: vv * 1.5 for kk, vv in v.items()} 
                         for k, v in thresholds.items()}
        
        freq_thresholds = thresholds.get(frequency, thresholds['weekly'])
        
        # Determine safety level
        if total_grey_line <= freq_thresholds['safe']:
            safety = 'safe'
            color = 'green'
            message = 'Within safe limits - client will not notice'
        elif total_grey_line <= freq_thresholds['warning']:
            safety = 'warning'
            color = 'yellow'
            message = 'Borderline safe - monitor closely'
        else:
            safety = 'danger'
            color = 'red'
            message = 'High risk - consider reducing adjustment'
        
        return {
            'safety_level': safety,
            'color': color,
            'message': message,
            'thresholds': freq_thresholds,
            'current_value': total_grey_line
        }
    
    def _generate_detailed_report(self, session_data: Dict, 
                                 recurrence_adj: Dict, 
                                 site_analysis: Dict, 
                                 total_grey_line: int, 
                                 optimization: Dict) -> str:
        """Generate detailed report"""
        report = []
        
        report.append("GREY LINE CALCULATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Session Information
        report.append("SESSION INFORMATION:")
        report.append(f"  Client Type: {session_data.get('client_type', 'N/A')}")
        report.append(f"  Frequency: {session_data.get('frequency', 'N/A')}")
        report.append(f"  Base Minutes: {session_data.get('base_minutes', 0)}")
        report.append("")
        
        # Recurrence Analysis
        report.append("RECURRENCE ANALYSIS:")
        report.append(f"  {recurrence_adj.get('description', '')}")
        report.append(f"  Adjustment Range: {recurrence_adj['adjustment_range'][0]}-{recurrence_adj['adjustment_range'][1]} minutes")
        report.append(f"  Safety Level: {recurrence_adj.get('safety_level', 'N/A')}")
        report.append("")
        
        # Site Visit Analysis
        if site_analysis:
            report.append("SITE VISIT ANALYSIS:")
            report.append(f"  Dirt Level: {site_analysis.get('dirt_level', 'N/A').upper()}")
            report.append(f"  Score: {site_analysis.get('average_score', 0)}/10")
            report.append(f"  Recommended Adjustment: {site_analysis.get('recommended_adjustment', {}).get('min', 0)}-{site_analysis.get('recommended_adjustment', {}).get('max', 0)} minutes")
            report.append("")
        
        # Total Calculation
        report.append("TOTAL GREY LINE CALCULATION:")
        report.append(f"  Total Grey Line Minutes: {total_grey_line}")
        report.append(f"  As Hours: {total_grey_line/60:.2f} hours")
        report.append("")
        
        # Billing Optimization
        report.append("BILLING OPTIMIZATION:")
        report.append(f"  Safety Level: {optimization.get('safety_level', 'N/A')}")
        report.append(f"  Client Visible Hours: {optimization.get('client_visible_hours', 0)}")
        report.append(f"  Actual Billed Hours: {optimization.get('billed_hours', 0)}")
        report.append(f"  Justification: {optimization.get('justification', 'N/A')}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append(f"  {optimization.get('recommended_action', 'No specific recommendations')}")
        
        return "\n".join(report)
    
    def _get_recommendations(self, total_grey_line: int, optimization: Dict) -> List[str]:
        """Get recommendations"""
        recommendations = []
        
        safety = optimization.get('safety_level', '')
        if safety == 'unsafe':
            recommendations.append("Immediately reduce grey line adjustment")
            recommendations.append("Review client history and patterns")
        elif safety == 'caution':
            recommendations.append("Monitor this client closely")
            recommendations.append("Consider slight reduction in adjustment")
        
        percentage = optimization.get('grey_line_percentage', 0)
        if percentage > 25:
            recommendations.append("Grey line percentage is high - consider reducing")
        
        if optimization.get('is_safe_for_billing', False):
            recommendations.append("Safe for billing - proceed as planned")
        
        return recommendations

# Singleton instance
grey_line_calculator = AdvancedGreyLineCalculator()


"""
API endpoints for the Grey Line Overbilling System
"""

from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime, timedelta
import uuid

from engine.greyline_calculator import grey_line_calculator
from utils.geo_fencing import geo_fence_manager
from database.db_handler import db_handler

api = Blueprint('api', __name__, url_prefix='/api/v1')

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Grey Line Overbilling System',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })

@api.route('/user/info', methods=['GET'])
@login_required
def user_info():
    """Get current user information"""
    return jsonify({
        'id': current_user.id,
        'email': current_user.email,
        'name': current_user.name,
        'role': current_user.role,
        'is_active': current_user.is_active,
        'created_at': current_user.created_at.isoformat() if current_user.created_at else None
    })

@api.route('/clients', methods=['GET'])
@login_required
def get_clients():
    """Get all clients"""
    try:
        # In a real implementation, this would query the database
        # For now, return mock data
        clients = [
            {
                'id': 'client_1',
                'name': 'John Smith',
                'email': 'john@example.com',
                'client_type': 'residential',
                'frequency': 'weekly',
                'total_cleans': 12,
                'avg_grey_line_minutes': 8.5,
                'has_pets': True,
                'has_kids': False,
                'has_elderly': False
            },
            {
                'id': 'client_2',
                'name': 'ABC Corporation',
                'email': 'office@abccorp.com',
                'client_type': 'commercial',
                'frequency': 'daily',
                'total_cleans': 150,
                'avg_grey_line_minutes': 12.3,
                'has_pets': False,
                'has_kids': False,
                'has_elderly': False
            }
        ]
        
        return jsonify(clients)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/clients/<client_id>', methods=['GET'])
@login_required
def get_client(client_id):
    """Get specific client details"""
    try:
        # In a real implementation, this would query the database
        client = {
            'id': client_id,
            'name': 'Example Client',
            'email': 'client@example.com',
            'phone': '+1234567890',
            'address': '123 Main St, Anytown, USA',
            'client_type': 'residential',
            'frequency': 'monthly',
            'total_cleans': 8,
            'first_clean_date': '2024-01-15T00:00:00Z',
            'last_clean_date': '2024-08-15T00:00:00Z',
            'has_pets': True,
            'has_kids': True,
            'has_elderly': False,
            'latitude': 40.7128,
            'longitude': -74.0060,
            'manager_id': current_user.id
        }
        
        return jsonify(client)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/clients/<client_id>/sessions', methods=['GET'])
@login_required
def get_client_sessions(client_id):
    """Get cleaning sessions for a client"""
    try:
        # Calculate date range (last 90 days)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=90)
        
        # In a real implementation, this would query the database
        sessions = [
            {
                'id': 'session_1',
                'scheduled_date': (end_date - timedelta(days=7)).isoformat(),
                'actual_start': (end_date - timedelta(days=7, hours=2)).isoformat(),
                'actual_end': (end_date - timedelta(days=7, hours=4)).isoformat(),
                'base_minutes': 120,
                'grey_line_minutes': 15,
                'total_billed_minutes': 135,
                'status': 'completed',
                'has_task_expansion': True,
                'weather_conditions': 'sunny',
                'cleaner_lat': 40.7128,
                'cleaner_lng': -74.0060,
                'is_within_geo_fence': True
            },
            {
                'id': 'session_2',
                'scheduled_date': (end_date - timedelta(days=14)).isoformat(),
                'actual_start': (end_date - timedelta(days=14, hours=2)).isoformat(),
                'actual_end': (end_date - timedelta(days=14, hours=3, minutes=45)).isoformat(),
                'base_minutes': 120,
                'grey_line_minutes': 8,
                'total_billed_minutes': 128,
                'status': 'completed',
                'has_task_expansion': False,
                'weather_conditions': 'rainy',
                'cleaner_lat': 40.7128,
                'cleaner_lng': -74.0061,
                'is_within_geo_fence': True
            }
        ]
        
        return jsonify({
            'client_id': client_id,
            'sessions': sessions,
            'total_sessions': len(sessions),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/geo-fencing/<client_id>/fences', methods=['GET'])
@login_required
def get_client_geo_fences(client_id):
    """Get geo-fences for a client"""
    try:
        # In a real implementation, this would query the database
        fences = [
            {
                'id': 'fence_1',
                'name': 'Main Entrance',
                'latitude': 40.7128,
                'longitude': -74.0060,
                'radius_meters': 100,
                'is_active': True,
                'created_at': '2024-01-01T00:00:00Z'
            },
            {
                'id': 'fence_2',
                'name': 'Parking Area',
                'latitude': 40.7130,
                'longitude': -74.0062,
                'radius_meters': 50,
                'is_active': True,
                'created_at': '2024-01-15T00:00:00Z'
            }
        ]
        
        return jsonify({
            'client_id': client_id,
            'fences': fences,
            'total_fences': len(fences),
            'active_fences': len([f for f in fences if f['is_active']])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/geo-fencing/<client_id>/fences', methods=['POST'])
@login_required
def create_geo_fence(client_id):
    """Create a new geo-fence"""
    try:
        data = request.json
        
        # Validate required fields
        if not data.get('latitude') or not data.get('longitude'):
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        # Create fence
        fence_data = {
            'client_id': client_id,
            'name': data.get('name', 'New Geo Fence'),
            'latitude': data.get('latitude'),
            'longitude': data.get('longitude'),
            'radius_meters': data.get('radius_meters', 100),
            'is_active': data.get('is_active', True)
        }
        
        # In a real implementation, this would save to database
        fence_id = f"fence_{datetime.utcnow().timestamp()}"
        
        return jsonify({
            'success': True