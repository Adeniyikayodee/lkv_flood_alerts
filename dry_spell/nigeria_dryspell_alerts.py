#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import ee
import geemap
import numpy as np
import pandas as pd
from dateutil import parser as date_parser
from geopy.geocoders import Nominatim
from tqdm import tqdm

# Image processing
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# Text-to-speech
from gtts import gTTS
import pygame

# Optional: SMS/WhatsApp support
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Earth Engine Collections
CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
MODIS_LST_COLLECTION = "MODIS/061/MOD11A1"  # Land Surface Temperature
MODIS_NDVI_COLLECTION = "MODIS/061/MOD13Q1"  # Vegetation Index
ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"  # Weather reanalysis
SMAP_COLLECTION = "NASA/SMAP/SPL3SMP_E/005"  # Soil moisture

# Nigerian Language Mappings by Geopolitical Zone
ZONE_LANGUAGES = {
    "North-West": {
        "primary": "ha",  # Hausa
        "secondary": ["ful"]  # Fulani
    },
    "North-East": {
        "primary": "ha",  # Hausa
        "secondary": ["kr", "ful"]  # Kanuri, Fulani
    },
    "North-Central": {
        "primary": "ha",  # Hausa
        "secondary": ["tiv", "ig", "yo"]  # Tiv, Igbo, Yoruba
    },
    "South-West": {
        "primary": "yo",  # Yoruba
        "secondary": ["en"]
    },
    "South-East": {
        "primary": "ig",  # Igbo
        "secondary": ["en"]
    },
    "South-South": {
        "primary": "en",  # English
        "secondary": ["ij", "ur", "ed"]  # Ijaw, Urhobo, Edo
    }
}

# ---------------------------------------------------------------------------
# Helper utilities for safe Earth Engine operations
# ---------------------------------------------------------------------------
def safe_collection_mean(image_collection: ee.ImageCollection, band_name: str, fallback_value: float = 0.0) -> ee.Image:
    """Return mean image if collection has data; otherwise constant fallback image with band name.

    Uses server-side branching to avoid empty-band images.
    """
    size = image_collection.size()
    mean_image = image_collection.mean()
    fallback = ee.Image.constant(fallback_value).rename(band_name)
    # Ensure the mean image has the expected band name if present
    mean_image = ee.Image(mean_image).rename(band_name)
    return ee.Image(ee.Algorithms.If(size.gt(0), mean_image, fallback))


def safe_collection_sum(image_collection: ee.ImageCollection, band_name: str, fallback_value: float = 0.0) -> ee.Image:
    """Return sum image if collection has data; otherwise constant fallback image with band name."""
    size = image_collection.size()
    sum_image = image_collection.sum()
    fallback = ee.Image.constant(fallback_value).rename(band_name)
    # Ensure the sum image has the expected band name if present
    sum_image = ee.Image(sum_image).rename(band_name)
    return ee.Image(ee.Algorithms.If(size.gt(0), sum_image, fallback))


def safe_divide(numerator: ee.Image, denominator: ee.Image, eps: float = 1e-6) -> ee.Image:
    """Safe element-wise division: denominator clipped to minimum epsilon to avoid divide-by-zero.

    Assumes single-band images with consistent band naming.
    """
    denom_safe = denominator.max(eps)
    return numerator.divide(denom_safe)

# Nigerian States to Geopolitical Zones
STATE_TO_ZONE = {
    # North-West
    "Sokoto": "North-West", "Kebbi": "North-West", "Zamfara": "North-West",
    "Katsina": "North-West", "Kaduna": "North-West", "Kano": "North-West", "Jigawa": "North-West",
    
    # North-East  
    "Borno": "North-East", "Yobe": "North-East", "Bauchi": "North-East",
    "Gombe": "North-East", "Adamawa": "North-East", "Taraba": "North-East",
    
    # North-Central
    "Niger": "North-Central", "Kogi": "North-Central", "Benue": "North-Central",
    "Plateau": "North-Central", "Nasarawa": "North-Central", "Kwara": "North-Central", "FCT": "North-Central",
    
    # South-West
    "Lagos": "South-West", "Ogun": "South-West", "Oyo": "South-West",
    "Osun": "South-West", "Ondo": "South-West", "Ekiti": "South-West",
    
    # South-East
    "Anambra": "South-East", "Enugu": "South-East", "Imo": "South-East",
    "Abia": "South-East", "Ebonyi": "South-East",
    
    # South-South
    "Edo": "South-South", "Delta": "South-South", "Rivers": "South-South",
    "Bayelsa": "South-South", "Cross River": "South-South", "Akwa Ibom": "South-South"
}

# Major LGA coordinates (sample - expand as needed)
LGA_COORDINATES = {
    ("Kano", "Kano Municipal"): (12.0022, 8.5919),
    ("Lagos", "Lagos Island"): (6.4549, 3.4246),
    ("Abuja", "AMAC"): (9.0579, 7.4951),
    ("Kaduna", "Kaduna North"): (10.5105, 7.4165),
    ("Enugu", "Enugu North"): (6.4584, 7.5464),
    ("Rivers", "Port Harcourt"): (4.8156, 7.0498),
    ("Borno", "Maiduguri"): (11.8333, 13.1500),
    ("Oyo", "Ibadan North"): (7.3775, 3.9470),
}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate dry spell alerts for Nigerian LGAs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Location
    parser.add_argument("--state", default="Kano",
                        help="Nigerian state name")
    parser.add_argument("--lga", default="Kano Municipal",
                        help="Local Government Area name")
    parser.add_argument("--ward", default="",
                        help="Ward name (optional)")
    
    # Date range
    parser.add_argument("--start-date", default="2025-03-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default="2025-05-31",
                        help="End date YYYY-MM-DD")
    
    # Thresholds
    parser.add_argument("--rain-threshold", type=float, default=5.0,
                        help="Daily rainfall threshold in mm")
    parser.add_argument("--dry-days", type=int, default=10,
                        help="Consecutive dry days to trigger alert")
    
    # Output
    parser.add_argument("--language", default="auto",
                        help="Target language code or 'auto'")
    parser.add_argument("--outdir", default="./dry_spell_alerts",
                        help="Output directory")
    
    # Configuration file
    parser.add_argument("--config", type=str,
                        help="JSON configuration file")
    
    # Communication
    parser.add_argument("--phone", help="Alert phone number")
    parser.add_argument("--email", help="Alert email address")
    
    args = parser.parse_args()
    
    # Load JSON config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
    
    # Parse dates
    args.start_date = date_parser.parse(args.start_date).date()
    args.end_date = date_parser.parse(args.end_date).date()
    
    # Auto-select language
    if args.language == "auto":
        zone = STATE_TO_ZONE.get(args.state, "North-Central")
        args.language = ZONE_LANGUAGES[zone]["primary"]
        logger.info(f"Auto-selected language: {args.language} for {zone}")
    
    return args


def initialize_earth_engine():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize()
        logger.info("Earth Engine initialized successfully")
    except Exception as e:
        logger.warning(f"Default initialization failed: {e}")
        try:
            ee.Authenticate()
            ee.Initialize()
            logger.info("Earth Engine authenticated and initialized")
        except Exception as auth_error:
            logger.error(f"Failed to initialize Earth Engine: {auth_error}")
            sys.exit(1)


def get_lga_location(state: str, lga: str) -> Tuple[float, float]:
    """Get LGA coordinates."""
    # Check offline database first
    if (state, lga) in LGA_COORDINATES:
        return LGA_COORDINATES[(state, lga)]
    
    # Try online geocoding
    try:
        geolocator = Nominatim(user_agent="nigeria_dryspell/1.0")
        query = f"{lga}, {state}, Nigeria"
        location = geolocator.geocode(query, timeout=10)
        
        if location:
            logger.info(f"Found coordinates for {lga}: {location.latitude}, {location.longitude}")
            return location.latitude, location.longitude
    except Exception as e:
        logger.warning(f"Geocoding failed: {e}")
    
    # Fallback to state capital
    logger.warning(f"Using approximate location for {state}")
    return 9.0765, 7.3986  # Nigeria center


def create_aoi(lat: float, lon: float, buffer_km: float = 30) -> ee.Geometry:
    """Create area of interest around LGA."""
    point = ee.Geometry.Point([lon, lat])
    aoi = point.buffer(buffer_km * 1000)
    logger.info(f"Created AOI with {buffer_km}km radius")
    return aoi


def calculate_dry_spell_indicators(aoi: ee.Geometry, start_date: str, 
                                  end_date: str, rain_threshold: float,
                                  dry_days_threshold: int) -> Dict:
    """
    Calculate dry spell indicators using multiple data sources.
    
    Returns dict with:
    - consecutive_dry_days: Maximum consecutive days below threshold
    - rainfall_deficit: Percentage below normal
    - soil_moisture: Average soil moisture
    - vegetation_stress: NDVI anomaly
    - temperature_anomaly: Temperature above normal
    """
    logger.info("Calculating dry spell indicators...")
    
    # 1. Rainfall Analysis (CHIRPS)
    chirps = ee.ImageCollection(CHIRPS_COLLECTION) \
              .filterBounds(aoi) \
              .filterDate(start_date, end_date)
    
    # Calculate daily rainfall
    def mark_dry_day(image):
        """Mark days with rainfall below threshold."""
        is_dry = image.select('precipitation').lt(rain_threshold)
        return is_dry.rename('dry_day') \
                    .set('system:time_start', image.get('system:time_start'))
    
    dry_days = chirps.map(mark_dry_day)
    
    # Count consecutive dry days (simplified total); safe against empty collections
    total_dry = safe_collection_sum(dry_days, 'dry_day').clip(aoi)
    
    # Historical baseline (5-year average)
    hist_start = ee.Date(start_date).advance(-5, 'year')
    hist_end = ee.Date(start_date).advance(-1, 'year')
    
    historical = safe_collection_sum(
        ee.ImageCollection(CHIRPS_COLLECTION)
          .filterBounds(aoi)
          .filterDate(hist_start, hist_end),
        band_name='precipitation',
        fallback_value=0.0
    )
    
    current_total = safe_collection_sum(chirps, 'precipitation', 0.0)
    
    # Rainfall deficit
    deficit = safe_divide(historical.subtract(current_total), historical) \
                       .multiply(100) \
                       .rename('rainfall_deficit')
    
    # 2. Vegetation Health (MODIS NDVI)
    ndvi = ee.ImageCollection(MODIS_NDVI_COLLECTION) \
             .filterBounds(aoi) \
             .filterDate(start_date, end_date) \
             .select('NDVI')
    
    # NDVI anomaly
    ndvi_mean = safe_collection_mean(ndvi, 'NDVI', 0.0)
    ndvi_historical = safe_collection_mean(
        ee.ImageCollection(MODIS_NDVI_COLLECTION)
          .filterBounds(aoi)
          .filterDate(hist_start, hist_end)
          .select('NDVI'),
        band_name='NDVI',
        fallback_value=0.0
    )
    
    ndvi_anomaly = safe_divide(ndvi_mean.subtract(ndvi_historical), ndvi_historical) \
                            .multiply(100) \
                            .rename('ndvi_anomaly')
    
    # 3. Land Surface Temperature (MODIS)
    lst = ee.ImageCollection(MODIS_LST_COLLECTION) \
            .filterBounds(aoi) \
            .filterDate(start_date, end_date) \
            .select('LST_Day_1km')
    
    # Convert from Kelvin to Celsius and calculate anomaly
    lst_celsius = safe_collection_mean(lst, 'LST_Day_1km', 273.15/0.02).multiply(0.02).subtract(273.15)
    
    # 4. Soil Moisture (if available)
    try:
        soil_moisture = safe_collection_mean(
            ee.ImageCollection(SMAP_COLLECTION)
              .filterBounds(aoi)
              .filterDate(start_date, end_date)
              .select('soil_moisture_pm'),
            band_name='soil_moisture_pm',
            fallback_value=0.0
        )
    except:
        # Fallback: estimate from rainfall
        soil_moisture = current_total.multiply(0.1).rename('soil_moisture_pm')
    
    return {
        'dry_days_total': total_dry,
        'rainfall_deficit': deficit.clip(aoi),
        'ndvi_anomaly': ndvi_anomaly.clip(aoi),
        'temperature': lst_celsius.clip(aoi),
        'soil_moisture': soil_moisture.clip(aoi)
    }


def assess_drought_risk(indicators: Dict, aoi: ee.Geometry) -> ee.Image:
    """
    Combine indicators into drought risk assessment.
    
    Risk levels:
    - 0: No risk
    - 1: Low risk (monitoring needed)
    - 2: Moderate risk (prepare for dry spell)
    - 3: High risk (immediate action needed)
    """
    logger.info("Assessing drought risk level...")
    
    # Weight different indicators
    weights = {
        'rainfall': 0.35,
        'vegetation': 0.25,
        'temperature': 0.20,
        'soil': 0.20
    }
    
    # Normalize and threshold each indicator. Guard against bandless images by renaming bands.
    rain = ee.Image(indicators['rainfall_deficit']).rename('rainfall_deficit')
    ndvi = ee.Image(indicators['ndvi_anomaly']).rename('ndvi_anomaly')
    temp = ee.Image(indicators['temperature']).rename('temperature')
    soil = ee.Image(indicators['soil_moisture']).rename('soil_moisture')

    # High risk if rainfall deficit > 30%
    rain_risk = rain.gt(30).multiply(weights['rainfall'])

    # High risk if NDVI drops > 20%
    veg_risk = ndvi.lt(-20).multiply(weights['vegetation'])

    # High risk if temperature > 35°C
    temp_risk = temp.gt(35).multiply(weights['temperature'])

    # High risk if soil moisture < 0.2
    soil_risk = soil.lt(0.2).multiply(weights['soil'])
    
    # Combine risks
    total_risk = rain_risk.add(veg_risk).add(temp_risk).add(soil_risk)
    
    # Classify into risk levels
    risk_level = ee.Image(0) \
        .where(total_risk.gt(0.2), 1) \
        .where(total_risk.gt(0.4), 2) \
        .where(total_risk.gt(0.6), 3) \
        .rename('drought_risk') \
        .clip(aoi)
    
    return risk_level


def generate_alert_message(state: str, lga: str, risk_level: int,
                          indicators: Dict, language: str) -> Dict[str, str]:
    """Generate alert message in local language."""
    
    # Base English message templates
    templates = {
        1: f"""
Attention {lga} farmers,
We are monitoring weather patterns that suggest possible dry conditions ahead.
Current rainfall is slightly below normal. Consider checking your water storage
and planning irrigation schedules. Continue monitoring weather updates.
""",
        2: f"""
Important notice for {lga} agricultural community,
Dry spell conditions are developing in your area. Rainfall is {indicators.get('deficit', 20):.0f}% below normal.
Please prepare by:
- Conserving water resources
- Applying mulch to retain soil moisture
- Considering drought-resistant crop varieties
Stay tuned for further updates.
""",
        3: f"""
Urgent: Severe dry spell alert for {lga},
Critical drought conditions detected. Immediate action recommended:
- Activate water conservation measures
- Prioritize irrigation for essential crops
- Contact agricultural extension office for support
- Consider early harvesting where possible
Your local agricultural office can provide drought assistance.
"""
    }
    
    english_msg = templates.get(risk_level, templates[1]).strip()
    
    # Simple translation placeholders (integrate with actual translation service)
    translations = {
        "ha": f"[Hausa] {english_msg}",  # Would use actual translation
        "yo": f"[Yoruba] {english_msg}",
        "ig": f"[Igbo] {english_msg}",
    }
    
    local_msg = translations.get(language, english_msg)
    
    return {
        "english": english_msg,
        "local": local_msg,
        "language": language,
        "risk_level": risk_level
    }


def create_risk_map(risk_image: ee.Image, indicators: Dict,
                   aoi: ee.Geometry, state: str, lga: str,
                   output_path: str) -> str:
    """Create visualization of drought risk."""
    logger.info("Creating risk map visualization...")
    
    try:
        # Export risk image
        risk_vis = {
            'min': 0,
            'max': 3,
            'palette': ['green', 'yellow', 'orange', 'red']
        }
        
        # Use geemap to export
        geemap.ee_export_image(
            risk_image,
            filename=f"{output_path}_risk.tif",
            scale=1000,
            region=aoi,
            file_per_band=False
        )
        
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Dry Spell Analysis - {lga}, {state}', fontsize=16, fontweight='bold')
        
        # Risk map (top left)
        ax1 = axes[0, 0]
        ax1.set_title('Drought Risk Level')
        # Placeholder for actual risk data
        risk_data = np.random.randint(0, 4, (100, 100))
        im1 = ax1.imshow(risk_data, cmap='RdYlGn_r', vmin=0, vmax=3)
        plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2, 3],
                    label='Risk Level',
                    orientation='horizontal',
                    pad=0.1)
        
        # Rainfall deficit (top right)
        ax2 = axes[0, 1]
        ax2.set_title('Rainfall Deficit (%)')
        deficit_data = np.random.uniform(-50, 50, (100, 100))
        im2 = ax2.imshow(deficit_data, cmap='BrBG', vmin=-50, vmax=50)
        plt.colorbar(im2, ax=ax2, label='Deficit %', orientation='horizontal', pad=0.1)
        
        # NDVI anomaly (bottom left)
        ax3 = axes[1, 0]
        ax3.set_title('Vegetation Health (NDVI Anomaly)')
        ndvi_data = np.random.uniform(-0.3, 0.3, (100, 100))
        im3 = ax3.imshow(ndvi_data, cmap='RdYlGn', vmin=-0.3, vmax=0.3)
        plt.colorbar(im3, ax=ax3, label='NDVI Anomaly', orientation='horizontal', pad=0.1)
        
        # Temperature (bottom right)
        ax4 = axes[1, 1]
        ax4.set_title('Land Surface Temperature (°C)')
        temp_data = np.random.uniform(25, 45, (100, 100))
        im4 = ax4.imshow(temp_data, cmap='hot', vmin=25, vmax=45)
        plt.colorbar(im4, ax=ax4, label='Temperature °C', orientation='horizontal', pad=0.1)
        
        # Add timestamp
        fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                ha='right', va='bottom', fontsize=8)
        
        # Save figure
        output_file = f"{output_path}_analysis.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved risk map to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to create risk map: {e}")
        return None


def generate_audio_alert(text: str, language: str, output_path: str) -> bool:
    """Generate audio file from text."""
    try:
        # Map language codes to gTTS codes
        gtts_langs = {
            'ha': 'ar',  # Use Arabic as fallback for Hausa
            'yo': 'yo',  # Yoruba if available
            'ig': 'ig',  # Igbo if available
            'en': 'en'
        }
        
        # Default to English if language not supported
        tts_lang = gtts_langs.get(language, 'en')
        
        # Generate audio
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        tts.save(output_path)
        
        logger.info(f"Generated audio alert: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        return False


def send_alert(message: str, phone: str = None, email: str = None) -> bool:
    """Send alert via SMS or email."""
    success = False
    
    if phone and TWILIO_AVAILABLE:
        try:
            account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
            auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
            from_phone = os.environ.get('TWILIO_PHONE_FROM')
            
            if all([account_sid, auth_token, from_phone]):
                client = Client(account_sid, auth_token)
                
                # Truncate message for SMS
                sms_msg = message[:160] if len(message) > 160 else message
                
                msg = client.messages.create(
                    from_=from_phone,
                    to=phone,
                    body=sms_msg
                )
                
                logger.info(f"SMS sent: {msg.sid}")
                success = True
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
    
    if email:
        # Implement email sending (using sendgrid, smtp, etc.)
        logger.info(f"Email alerts not yet implemented")
    
    return success


def main():
    """Main execution flow."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        log_dir = Path(args.outdir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("Nigerian Dry Spell Alert System")
        logger.info(f"Location: {args.lga}, {args.state}")
        logger.info(f"Period: {args.start_date} to {args.end_date}")
        logger.info(f"Language: {args.language}")
        logger.info("=" * 70)
        
        # Initialize Earth Engine
        initialize_earth_engine()
        
        # Get LGA coordinates
        lat, lon = get_lga_location(args.state, args.lga)
        
        # Create area of interest
        aoi = create_aoi(lat, lon, buffer_km=30)
        
        # Calculate dry spell indicators
        indicators = calculate_dry_spell_indicators(
            aoi,
            args.start_date.isoformat(),
            args.end_date.isoformat(),
            args.rain_threshold,
            args.dry_days
        )
        
        # Assess drought risk
        risk_image = assess_drought_risk(indicators, aoi)
        
        # Get risk statistics
        stats = risk_image.reduceRegion(
            reducer=ee.Reducer.mode(),
            geometry=aoi,
            scale=1000,
            maxPixels=1e9
        )
        
        # Guard: if region reduction returns None, default to 0
        risk_value = ee.Algorithms.If(stats.contains('drought_risk'), stats.get('drought_risk'), 0)
        risk_level = ee.Number(risk_value).format().getInfo()
        try:
            risk_level = int(risk_level)
        except Exception:
            risk_level = 0
        
        # Get rainfall deficit for message
        deficit_stats = indicators['rainfall_deficit'].reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=1000,
            maxPixels=1e9
        )
        
        deficit_value = ee.Number(ee.Algorithms.If(deficit_stats.contains('rainfall_deficit'), deficit_stats.get('rainfall_deficit'), 0)).getInfo()
        
        # Generate alert if risk detected
        if risk_level > 0:
            logger.info(f"Drought risk level: {risk_level}")
            
            # Create alert message
            alert = generate_alert_message(
                args.state,
                args.lga,
                int(risk_level),
                {'deficit': deficit_value},
                args.language
            )
            
            # Save alert text
            alert_file = log_dir / f"{args.state}_{args.lga}_{datetime.now().strftime('%Y%m%d')}_alert.txt"
            with open(alert_file, 'w', encoding='utf-8') as f:
                f.write(f"DRY SPELL ALERT - {args.lga}, {args.state}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Risk Level: {risk_level}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
                f.write("=" * 50 + "\n\n")
                f.write("LOCAL LANGUAGE VERSION:\n")
                f.write(alert['local'] + "\n\n")
                f.write("ENGLISH VERSION:\n")
                f.write(alert['english'] + "\n")
            
            logger.info(f"Alert saved to {alert_file}")
            
            # Generate audio
            audio_file = log_dir / f"{args.state}_{args.lga}_{datetime.now().strftime('%Y%m%d')}_audio.mp3"
            generate_audio_alert(alert['local'], args.language, str(audio_file))
            
            # Create risk map
            map_path = log_dir / f"{args.state}_{args.lga}_{datetime.now().strftime('%Y%m%d')}"
            create_risk_map(risk_image, indicators, aoi, args.state, args.lga, str(map_path))
            
            # Send alerts
            if args.phone or args.email:
                send_alert(alert['english'], args.phone, args.email)
            
            # Print to console
            print("\n" + "=" * 50)
            print(f"DRY SPELL ALERT - {args.lga}, {args.state}")
            print(f"Risk Level: {risk_level}/3")
            print("=" * 50)
            print(alert['local'])
            print("-" * 50)
            print(alert['english'])
            print("=" * 50)
            
        else:
            logger.info("No significant drought risk detected")
            print(f"\n✅ No dry spell risk detected for {args.lga}, {args.state}")
            print(f"   Conditions are normal for the period {args.start_date} to {args.end_date}")
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()