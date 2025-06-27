#!/usr/bin/env python3
"""
lkv_flood_alerts.py - Lake Victoria Basin Flood Alert System

Generates culturally-sensitive flood alerts for community radio stations
in the Lake Victoria Basin (Uganda, Kenya, Tanzania, Rwanda) using 
Google Earth Engine satellite data and local-language AI translation.

SETUP INSTRUCTIONS FOR APPLE SILICON MAC (M1/M2/M3/M4):
========================================================

# Create and activate Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
export CMAKE_ARGS="-DLLAMA_METAL=on"

# For WhatsApp, set environment variables:
export TWILIO_ACCOUNT_SID="your_account_sid"
export TWILIO_AUTH_TOKEN="your_auth_token"
export TWILIO_WHATSAPP_FROM="whatsapp:+14155238886"  # Twilio sandbox number

# Upgrade pip and install build helpers (required for llama-cpp)
pip install --upgrade pip wheel cmake ninja

# Install runtime dependencies
pip install -r <(python lkv_flood_alerts.py --print-requirements)

pip install "ipython<9" geemap==0.31.0 ipywidgets traitlets widgetsnbextension -U

# For Metal-accelerated llama-cpp-python (uses Apple GPU):
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Authenticate with Google Earth Engine (one-time setup)
pip install --upgrade earthengine-api
earthengine authenticate --force

# Sanity check: print pinned requirements
python - <<'PY'
import torch, numpy, platform, sys
print("Torch version :", torch.__version__)
print("NumPy version :", numpy.__version__)
print("MPS available :", torch.backends.mps.is_available())
print("Python exe    :", sys.executable)
print("Arch / OS     :", platform.platform())
PY

# Sanity check for Google Earth Engine
python - <<'PY'
import ee; ee.Initialize()
print("✅ EE ready for:", ee.AccountInfo()['email'])
PY

# set GEE with project name
Go to https://console.cloud.google.com/apis/library/earthengine.googleapis.com
Select or create a project → Enable.
earthengine set_project YOUR_PROJECT_ID

USAGE:
======

python lkv_flood_alerts.py --config params.json

or 

python lkv_flood_alerts.py \\
    --country "Kenya" \\
    --county "Kisumu" \\
    --subcounty "Nyando" \\
    --community "Obunga" \\
    --start-date 2023-06-01 \\
    --end-date 2023-06-30 \\
    --target-lang auto \\
    --outdir ./alerts

Author: Kayode Adeniyi - Lake Victoria Basin Flood Alert System


# If having issues with GEE
brew install --cask google-cloud-sdk

# Add the CLI to your shell (Homebrew usually appends this automatically):
echo 'source "$(brew --prefix)/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc"' >> ~/.zshrc
echo 'source "$(brew --prefix)/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/completion.zsh.inc"' >> ~/.zshrc
exec zsh

# After installation, run:
gcloud init 

# then set project again
earthengine set_project YOUR_PROJECT_ID

#Run 
python lkv_flood_alerts.py --params params.json

"""

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
import tempfile
import subprocess

import ee
import geemap
import numpy as np
import pandas as pd
import torch
from dateutil import parser as date_parser
from geopy.geocoders import Nominatim
from tqdm import tqdm

# Image processing and multimedia
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import rasterio
from rasterio.plot import show

# Text-to-speech for local languages
import pyttsx3
from gtts import gTTS
import pygame

# Communication libraries
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import requests

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global constants for GEE collections
CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
JRC_WATER_COLLECTION = "JRC/GSW1_4/MonthlyHistory"
SENTINEL1_COLLECTION = "COPERNICUS/S1_GRD"
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR"
LANDSAT8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
LANDSAT9_COLLECTION = "LANDSAT/LC09/C02/T1_L2"

# Default language mappings for countries
LANGUAGE_DEFAULTS = {
    "Kenya": "sw",      # Swahili
    "Tanzania": "sw",   # Swahili
    "Uganda": "en",     # English (with regional variants)
    "Rwanda": "rw"      # Kinyarwanda
}

# Language codes for TTS
TTS_LANGUAGE_CODES = {
    "sw": "sw",         # Swahili
    "lg": "lg",         # Luganda
    "rw": "rw",         # Kinyarwanda
    "xog": "xog",       # Soga
    "en": "en"          # English
}

# Offline geocoding fallbacks for major cities
OFFLINE_GEOCODES = {
    ("Kenya", "Kisumu", "Nyando", "Obunga"): (-0.1036, 34.7617),
    ("Uganda", "Kampala", "Central", "Kisenyi"): (0.3163, 32.5822),
    ("Tanzania", "Mwanza", "Ilemela", "Kirumba"): (-2.5164, 32.9175),
    ("Rwanda", "Eastern", "Nyagatare", "Matimba"): (-1.3089, 30.3289),
}


def print_requirements():
    """Print pinned requirements for reproducible installation."""
    requirements = """# Lake Victoria Basin Flood Alert System Requirements
# Generated for Python 3.11 on macOS arm64 (Apple Silicon)

# Google Earth Engine and geospatial
earthengine-api==0.1.384
geemap==0.30.2
google-auth==2.26.1
google-auth-oauthlib==1.2.0
rasterio==1.3.9
matplotlib==3.8.2
Pillow==10.2.0

# Data processing
pandas==2.1.4
numpy==1.26.2
geopy==2.4.1
tqdm==4.66.1
python-dateutil==2.8.2
psutil==5.9.8

# Machine Learning - PyTorch with MPS support
torch>=2.2.0  # macOS arm64 wheel includes Metal Performance Shaders

# LLaMA model support
transformers==4.36.2
sentencepiece==0.1.99
llama-cpp-python==0.2.27  # Build with CMAKE_ARGS="-DLLAMA_METAL=on"

# Sentiment analysis
# cardiffnlp-models-huggingface-hub==0.0.3

# Text-to-Speech
pyttsx3==2.90
gTTS==2.5.0
pygame==2.5.2

# Communication
twilio==8.11.0
requests==2.31.0

# Build tools (for llama-cpp-python)
cmake>=3.26.0
ninja>=1.11.1
"""
    print(requirements)
    sys.exit(0)


def parse_arguments():
    """
    Parse command-line arguments with optional JSON parameter file.

    Returns
    -------
    argparse.Namespace
        Parsed and validated arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate flood alerts for Lake Victoria Basin communities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Special actions
    parser.add_argument("--print-requirements", action="store_true",
                        help="Print pinned requirements and exit")
    parser.add_argument("--params", type=str,
                        help="JSON file with parameters (overrides CLI args)")
    parser.add_argument("--config", type=str,
                        help="Alias for --params (JSON parameters file)")

    # Location
    parser.add_argument("--country",   default="Kenya",
                        help="Country (Uganda/Kenya/Tanzania/Rwanda)")
    parser.add_argument("--county",    default="Kisumu",
                        help="County/District name")
    parser.add_argument("--subcounty", default="Nyando",
                        help="Subcounty/Division name")
    parser.add_argument("--community", default="Obunga",
                        help="Community/Village name")

    # Date range
    parser.add_argument("--start-date", default="2025-06-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date",   default="2025-06-30",
                        help="End date YYYY-MM-DD")

    # Output
    parser.add_argument("--target-lang", default="auto",
                        help="Target language code or 'auto'")
    parser.add_argument("--outdir",      default="./alerts",
                        help="Output directory for alerts and rasters")

    # Communication
    parser.add_argument("--leader-phone",
                        help="Community leader WhatsApp number (+ country code)")
    parser.add_argument("--radio-phone",
                        help="Radio station WhatsApp number")
    parser.add_argument("--radio-sms",
                        help="Radio station SMS number")

    # Parse CLI now
    args = parser.parse_args()

    # Treat --config as an alias for --params
    if args.config and not args.params:
        args.params = args.config

    # Handle --print-requirements early exit
    if args.print_requirements:
        print_requirements()           # exits via sys.exit(0)

    # Load JSON parameters if provided
    if args.params:
        logger.info(f"Loading parameters from {args.params}")
        try:
            with open(args.params, "r") as f:
                params = json.load(f)
            for k, v in params.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                    logger.debug(f"Set {k} = {v} from JSON")
        except Exception as e:
            logger.error(f"Failed to load JSON parameters: {e}")
            sys.exit(1)

    # Validate and normalise dates
    try:
        args.start_date = date_parser.parse(args.start_date).date()
        args.end_date   = date_parser.parse(args.end_date).date()
        if args.start_date > args.end_date:
            logger.warning("Start date after end date; swapping.")
            args.start_date, args.end_date = args.end_date, args.start_date
    except Exception as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    # Auto-select language if requested
    if args.target_lang == "auto":
        args.target_lang = LANGUAGE_DEFAULTS.get(args.country, "en")
        if args.country == "Uganda":
            county_lower = args.county.lower()
            if "central" in county_lower:
                args.target_lang = "lg"      # Luganda
            elif any(r in county_lower for r in ("southwest", "ankole")):
                args.target_lang = "xog"     # Soga
        logger.info(f"Auto-selected language: {args.target_lang}")

    return args


def initialize_earth_engine():
    """
    Initialize Google Earth Engine with retry logic.
    
    Handles authentication and initialization with proper error messages.
    """
    try:
        # Try to initialize with default credentials
        ee.Initialize()
        logger.info("Google Earth Engine initialized successfully")
    except Exception as e:
        logger.warning(f"Default EE initialization failed: {e}")
        try:
            # Fallback to authentication flow
            ee.Authenticate()
            ee.Initialize()
            logger.info("Google Earth Engine authenticated and initialized")
        except Exception as auth_error:
            logger.error(f"Failed to initialize Earth Engine: {auth_error}")
            logger.error("Please run 'earthengine authenticate' in terminal")
            sys.exit(1)


def get_community_location(country: str, county: str, subcounty: str, 
                          community: str) -> Tuple[float, float]:
    """
    Get community centroid coordinates using Nominatim or offline fallback.
    
    Args:
        country: Country name
        county: County/District name
        subcounty: Subcounty/Division name
        community: Community/Village name
        
    Returns:
        Tuple of (latitude, longitude)
    """
    # Check offline cache first
    location_key = (country, county, subcounty, community)
    if location_key in OFFLINE_GEOCODES:
        lat, lon = OFFLINE_GEOCODES[location_key]
        logger.info(f"Using offline coordinates for {community}: {lat}, {lon}")
        return lat, lon
    
    # Try online geocoding
    try:
        geolocator = Nominatim(user_agent="lkv_flood_alerts/1.0")
        # Build hierarchical query
        query = f"{community}, {subcounty}, {county}, {country}"
        logger.info(f"Geocoding query: {query}")
        
        location = geolocator.geocode(query, timeout=10)
        if location:
            logger.info(f"Found coordinates: {location.latitude}, {location.longitude}")
            return location.latitude, location.longitude
        else:
            # Try broader search
            query = f"{county}, {country}"
            location = geolocator.geocode(query, timeout=10)
            if location:
                logger.warning(f"Using county centroid for {community}")
                return location.latitude, location.longitude
                
    except Exception as e:
        logger.warning(f"Geocoding failed: {e}")
    
    # Ultimate fallback - Lake Victoria approximate center
    logger.warning("Using Lake Victoria center as fallback location")
    return -1.0, 33.0


def create_aoi_buffer(lat: float, lon: float, buffer_km: float = 25) -> ee.Geometry:
    """
    Create area of interest with buffer around community point.
    
    Args:
        lat: Latitude
        lon: Longitude
        buffer_km: Buffer radius in kilometers
        
    Returns:
        ee.Geometry: Buffered polygon
    """
    point = ee.Geometry.Point([lon, lat])
    # Convert km to meters for buffer
    buffer_m = buffer_km * 1000
    aoi = point.buffer(buffer_m)
    
    logger.info(f"Created AOI with {buffer_km}km buffer around {lat:.4f}, {lon:.4f}")
    return aoi


def calculate_rainfall_percentiles(aoi: ee.Geometry, start_date: str, 
                                 end_date: str) -> Dict[str, ee.Image]:
    """
    Calculate rainfall statistics and anomalies using CHIRPS data.
    
    Computes 3-day and 7-day rainfall totals and identifies pixels above
    95th percentile as potential flood indicators.
    
    Args:
        aoi: Area of interest geometry
        start_date: Start date string
        end_date: End date string
        
    Returns:
        Dictionary with rainfall statistics images
    """
    logger.info("Calculating rainfall statistics from CHIRPS...")
    
    # Load CHIRPS daily rainfall
    chirps = ee.ImageCollection(CHIRPS_COLLECTION).filterBounds(aoi)
    
    # Historical baseline for percentiles (5 years)
    baseline_start = ee.Date(start_date).advance(-5, 'year')
    baseline_end = ee.Date(start_date).advance(-1, 'year')
    
    historical = chirps.filterDate(baseline_start, baseline_end)
    
    # Current period
    current = chirps.filterDate(start_date, end_date)
    
    # Helper function to calculate rolling sum
    def rolling_sum(collection, days):
        """Calculate rolling sum over specified days."""
        def accumulate(image):
            # Get previous N days
            end = image.date()
            start = end.advance(-days, 'day')
            
            window = collection.filterDate(start, end)
            # Check if window has any images
            total = ee.Image(ee.Algorithms.If(
                window.size().gt(0),
                window.sum(),
                ee.Image(0).rename('precipitation')
            ))
            
            return total.set('system:time_start', image.get('system:time_start')).clip(aoi)
        
        return collection.map(accumulate)
    
    # Calculate 3-day and 7-day totals
    rain_3day = rolling_sum(current, 3)
    rain_7day = rolling_sum(current, 7)
    
    # Calculate historical 95th percentile for 3-day rainfall
    hist_3day = rolling_sum(historical, 3)
    
    # Check if we have historical data
    hist_count = hist_3day.size()
    
    # Calculate percentile or use fallback
    percentile_95 = ee.Image(ee.Algorithms.If(
        hist_count.gt(10),  # Need at least 10 historical images
        hist_3day.reduce(ee.Reducer.percentile([95])),
        ee.Image.constant(50).rename('precipitation_p95')  # Fallback: 50mm
    ))
    
    # Flag pixels exceeding 95th percentile
    def flag_rainburst(img):
        # Safe comparison with null handling
        safe_percentile = ee.Image(percentile_95).unmask(50)  # Default to 50mm if masked
        return img.unmask(0).gt(safe_percentile).rename('rainburst') \
                  .set('system:time_start', img.get('system:time_start'))
    
    rainburst_flag = rain_3day.map(flag_rainburst)
    
    return {
        'rain_3day': rain_3day,
        'rain_7day': rain_7day,
        'percentile_95': percentile_95,
        'rainburst_flag': rainburst_flag
    }


def calculate_water_anomaly(aoi: ee.Geometry, year: int, month: int) -> ee.Image:
    """
    Calculate surface water extent anomaly using JRC Global Surface Water.
    
    Compares current month against 2015-2020 baseline mean + 1 standard deviation.
    
    Args:
        aoi: Area of interest
        year: Current year
        month: Current month
        
    Returns:
        ee.Image: Binary anomaly mask
    """
    logger.info(f"Calculating water extent anomaly for {year}-{month:02d}")
    
    # Load JRC monthly water history
    jrc = ee.ImageCollection(JRC_WATER_COLLECTION).filterBounds(aoi)
    
    # Baseline period (2015-2020)
    baseline = jrc.filter(ee.Filter.calendarRange(2015, 2020, 'year')) \
                  .filter(ee.Filter.calendarRange(month, month, 'month')) \
                  .select('water')
    
    # Check if baseline has data
    baseline_count = baseline.size()
    
    # Calculate baseline statistics with fallback
    baseline_mean = ee.Image(ee.Algorithms.If(
        baseline_count.gt(0),
        baseline.mean(),
        ee.Image(30)  # Default water percentage
    ))
    
    baseline_std = ee.Image(ee.Algorithms.If(
        baseline_count.gt(0),
        baseline.reduce(ee.Reducer.stdDev()),
        ee.Image(10)  # Default std deviation
    ))
    
    threshold = baseline_mean.add(baseline_std)  # mean + 1σ
    
    # Current month
    current_filtered = jrc.filter(ee.Filter.calendarRange(year, year, 'year')) \
                          .filter(ee.Filter.calendarRange(month, month, 'month')) \
                          .select('water')
    
    # Use most recent if current not available
    current = ee.Image(ee.Algorithms.If(
        current_filtered.size().gt(0),
        current_filtered.first(),
        jrc.select('water').sort('system:time_start', False).first()
    ))
    
    # Anomaly detection with null handling
    anomaly = current.unmask(0).gt(threshold.unmask(40)).rename('water_anomaly')
    
    return anomaly.clip(aoi)


def process_sentinel1_flood(aoi: ee.Geometry, start_date: str, 
                           end_date: str) -> ee.ImageCollection:
    """
    Process Sentinel-1 SAR data for flood detection.
    
    Uses VV < -17 dB and VH < -15 dB thresholds for water detection.
    SAR penetrates clouds, making it ideal for flood monitoring.
    
    Args:
        aoi: Area of interest
        start_date: Start date
        end_date: End date
        
    Returns:
        ee.ImageCollection: Flood masks
    """
    logger.info("Processing Sentinel-1 SAR flood detection...")
    
    # Load Sentinel-1 GRD data
    s1 = ee.ImageCollection(SENTINEL1_COLLECTION) \
           .filterBounds(aoi) \
           .filterDate(start_date, end_date) \
           .filter(ee.Filter.eq('instrumentMode', 'IW')) \
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    
    def detect_water(image):
        """Apply SAR thresholds for water detection."""
        # Get VV and VH bands in dB
        vv = image.select('VV')
        vh = image.select('VH')
        
        # Water detection thresholds
        # Low backscatter indicates smooth water surface
        water_mask = vv.lt(-17).And(vh.lt(-15))
        
        return water_mask.rename('sar_flood') \
                        .set('system:time_start', image.get('system:time_start')) \
                        .clip(aoi)
    
    flood_masks = s1.map(detect_water)
    
    # Count images (avoid getInfo in production)
    logger.info("Processing Sentinel-1 images...")
    
    return flood_masks


def process_optical_water(aoi: ee.Geometry, start_date: str, 
                         end_date: str) -> ee.ImageCollection:
    """
    Process optical imagery (Sentinel-2, Landsat) for water detection.
    
    Uses NDWI (Normalized Difference Water Index) with cloud masking.
    Falls back to Landsat if Sentinel-2 unavailable.
    
    Args:
        aoi: Area of interest
        start_date: Start date
        end_date: End date
        
    Returns:
        ee.ImageCollection: Water masks
    """
    logger.info("Processing optical water detection...")
    
    # Try Sentinel-2 first (10m resolution)
    s2 = ee.ImageCollection(SENTINEL2_COLLECTION) \
           .filterBounds(aoi) \
           .filterDate(start_date, end_date) \
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    
    def s2_water_mask(image):
        """Calculate NDWI and cloud mask for Sentinel-2."""
        # Cloud masking using QA60 band
        qa = image.select('QA60')
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
                     qa.bitwiseAnd(1 << 11).eq(0))
        
        # NDWI = (Green - NIR) / (Green + NIR)
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('ndwi')
        
        # Water threshold (NDWI > 0.3 typically indicates water)
        water = ndwi.gt(0.3).And(cloud_mask).rename('optical_water')
        
        return water.set('system:time_start', image.get('system:time_start')) \
                   .clip(aoi)
    
    s2_water = s2.map(s2_water_mask)
    
    # Check if we have enough Sentinel-2 data
    s2_count = s2.size()
    
    # Fallback to Landsat if insufficient Sentinel-2
    def get_landsat_water():
        """Process Landsat 8/9 as fallback."""
        logger.info("Using Landsat fallback for optical water detection")
        
        # Combine Landsat 8 and 9
        l8 = ee.ImageCollection(LANDSAT8_COLLECTION).filterBounds(aoi)
        l9 = ee.ImageCollection(LANDSAT9_COLLECTION).filterBounds(aoi)
        landsat = l8.merge(l9).filterDate(start_date, end_date)
        
        def landsat_water_mask(image):
            """Calculate NDWI for Landsat."""
            # Apply scaling factors
            optical = image.select(['SR_B.*']).multiply(0.0000275).add(-0.2)
            
            # Cloud mask using QA_PIXEL
            qa = image.select('QA_PIXEL')
            cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(
                        qa.bitwiseAnd(1 << 4).eq(0))
            
            # NDWI for Landsat
            ndwi = optical.normalizedDifference(['SR_B3', 'SR_B5']).rename('ndwi')
            water = ndwi.gt(0.3).And(cloud_mask).rename('optical_water')
            
            return water.set('system:time_start', image.get('system:time_start')) \
                       .clip(aoi)
        
        return landsat.map(landsat_water_mask)
    
    # Use Landsat if less than 3 Sentinel-2 images
    optical_water = ee.Algorithms.If(
        s2_count.lt(3),
        get_landsat_water(),
        s2_water
    )
    
    return ee.ImageCollection(optical_water)


def fuse_flood_indicators(rainfall, water_anomaly,
                          sar_floods, optical_water, aoi):
    """
    Combine rainfall bursts, monthly water-extent anomaly, Sentinel-1 SAR floods,
    and optical NDWI water masks into a daily binary flood-risk image.

    Returns
    -------
    ee.ImageCollection
        One image per day with band 'flood_risk' (1 = high risk, 0 = low/no risk)
        and properties:
            • system:time_start  – millis for that date
            • date_string        – 'YYYY-MM-dd'
    """
    logger.info("Fusing flood indicators into risk assessment...")
    
    # Helper: if Earth Engine returns null, replace with all-zero image
    def _safe(img, band_name='dummy'):
        return ee.Image(ee.Algorithms.If(
            img,
            img,
            ee.Image(0).rename(band_name).clip(aoi)
        )).unmask(0)  # Also unmask any masked pixels

    # Dates to iterate over come from rainfall['rainburst_flag']
    date_list = rainfall['rainburst_flag'].aggregate_array('system:time_start')

    def _daily(date_millis):
        date = ee.Date(date_millis)

        # 1 ─ Rainfall burst flag (within that exact day)
        rain_flag = rainfall['rainburst_flag'] \
            .filterDate(date, date.advance(1, 'day')).first()
        rain_flag = _safe(rain_flag, 'rainburst')

        # 2 ─ SAR flood mask (median of ±3 days)
        sar_window = sar_floods.filterDate(
            date.advance(-3, 'day'), date.advance(3, 'day'))
        sar_flood = ee.Image(ee.Algorithms.If(
            sar_window.size().gt(0),
            sar_window.median(),
            ee.Image(0).rename('sar_flood')
        ))
        sar_flood = _safe(sar_flood, 'sar_flood')

        # 3 ─ Optical water mask (median of ±5 days)
        optical_window = optical_water.filterDate(
            date.advance(-5, 'day'), date.advance(5, 'day'))
        optical = ee.Image(ee.Algorithms.If(
            optical_window.size().gt(0),
            optical_window.median(),
            ee.Image(0).rename('optical_water')
        ))
        optical = _safe(optical, 'optical_water')

        # 4 ─ Monthly water-extent anomaly
        water_anom = _safe(water_anomaly, 'water_anomaly')

        # 5 ─ Weighted sum → risk score
        # Create individual contribution layers
        rain_contrib = rain_flag.multiply(0.30)
        water_contrib = water_anom.multiply(0.20)
        sar_contrib = sar_flood.multiply(0.35)
        optical_contrib = optical.multiply(0.15)
        
        # Sum all contributions
        risk = rain_contrib.add(water_contrib).add(sar_contrib).add(optical_contrib)

        # Binary threshold at 0.5
        flood_risk = risk.gt(0.5).rename('flood_risk').set({
            'system:time_start': date_millis,
            'date_string': date.format('YYYY-MM-dd')
        }).clip(aoi)

        return flood_risk

    # Build the collection
    return ee.ImageCollection(date_list.map(_daily))



def check_alert_threshold(risk_image: ee.Image, aoi: ee.Geometry,
                         threshold: float = 0.15) -> bool:
    """
    Check if flood risk exceeds alert threshold.
    
    Args:
        risk_image: Binary flood risk image
        aoi: Area of interest
        threshold: Fraction of AOI that must be at risk (default 15%)
        
    Returns:
        bool: True if alert should be issued
    """
    try:
        # Ensure risk_image is valid
        if risk_image is None:
            logger.warning("Risk image is None")
            return False
            
        # Calculate fraction of AOI at risk
        # Use unmask to handle null values
        safe_risk = risk_image.select('flood_risk').unmask(0)
        
        stats = safe_risk.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=100,  # 100m resolution for efficiency
            maxPixels=1e9,
            bestEffort=True  # Use approximate computation if needed
        )
        
        # Get the risk fraction with error handling
        risk_fraction_ee = stats.get('flood_risk', 0)
        risk_fraction = risk_fraction_ee.getInfo()
        
        if risk_fraction is None:
            logger.warning("Could not compute risk fraction")
            return False
            
        logger.info(f"Flood risk fraction: {risk_fraction:.2%}")
        
        return risk_fraction >= threshold
        
    except Exception as e:
        logger.error(f"Error checking alert threshold: {e}")
        return False



def export_geotiff(image: ee.Image, filename: str, aoi: ee.Geometry,
                  scale: int = 30) -> str:
    """
    Export Earth Engine image to local GeoTIFF file.
    
    Uses geemap for local download instead of Drive export.
    
    Args:
        image: Image to export
        filename: Output filename (without extension)
        aoi: Area of interest
        scale: Pixel resolution in meters
        
    Returns:
        str: Path to exported file
    """
    logger.info(f"Exporting GeoTIFF: {filename}.tif")
    
    try:
        # Use geemap to download image
        output_path = f"{filename}.tif"
        geemap.ee_export_image(
            image, 
            filename=output_path,
            scale=scale,
            region=aoi,
            file_per_band=False
        )
        
        logger.info(f"Exported GeoTIFF to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to export GeoTIFF: {e}")
        return None


def convert_geotiff_to_images(geotiff_path: str, output_base: str, 
                             community_name: str, date_str: str) -> Dict[str, str]:
    """
    Convert GeoTIFF to JPEG and PDF with legends and annotations.
    
    Args:
        geotiff_path: Path to GeoTIFF file
        output_base: Base path for output files (without extension)
        community_name: Name of community for title
        date_str: Date string for title
        
    Returns:
        Dict with paths to created files
    """
    logger.info("Converting GeoTIFF to JPEG and PDF formats...")
    
    output_paths = {}
    
    try:
        # Read GeoTIFF
        with rasterio.open(geotiff_path) as src:
            data = src.read(1)
            transform = src.transform
            bounds = src.bounds
            
        # Create figure with map
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Display flood risk map
        im = ax.imshow(data, cmap='RdYlBu_r', alpha=0.8, 
                      extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
        
        # Add title and labels
        plt.title(f'Flood Risk Map - {community_name}\n{date_str}', 
                 fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Add colorbar with labels
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Flood Risk Level', fontsize=12)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Low Risk', 'High Risk'])
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#d73027', label='High Flood Risk'),
            mpatches.Patch(color='#4575b4', label='Low/No Risk')
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add scale bar (approximate)
        scalebar_length = (bounds.right - bounds.left) * 0.2
        scalebar_y = bounds.bottom + (bounds.top - bounds.bottom) * 0.05
        ax.plot([bounds.left + scalebar_length*0.1, 
                bounds.left + scalebar_length*1.1],
                [scalebar_y, scalebar_y], 'k-', linewidth=3)
        ax.text(bounds.left + scalebar_length*0.6, scalebar_y*1.02,
                f'~{int(scalebar_length*111)}km', ha='center', fontsize=10)
        
        # Add north arrow
        arrow_x = bounds.right - (bounds.right - bounds.left) * 0.1
        arrow_y = bounds.top - (bounds.top - bounds.bottom) * 0.1
        ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y-0.02),
                   ha='center', va='bottom', fontsize=14, fontweight='bold',
                   arrowprops=dict(arrowstyle='^', lw=2))
        
        # Save as JPEG
        jpeg_path = f"{output_base}.jpg"
        plt.savefig(jpeg_path, dpi=300, bbox_inches='tight', 
                   format='jpg', quality=95)
        output_paths['jpeg'] = jpeg_path
        logger.info(f"Saved JPEG: {jpeg_path}")
        
        # Save as PDF
        pdf_path = f"{output_base}.pdf"
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = f'Flood Risk Map - {community_name}'
            d['Author'] = 'Lake Victoria Basin Flood Alert System'
            d['Subject'] = f'Flood risk assessment for {date_str}'
            d['Keywords'] = 'Flood, Risk, Lake Victoria, Satellite'
            d['CreationDate'] = datetime.now()
            
        output_paths['pdf'] = pdf_path
        logger.info(f"Saved PDF: {pdf_path}")
        
        plt.close(fig)
        
        # Create a simplified version for WhatsApp (smaller size)
        fig_wa, ax_wa = plt.subplots(figsize=(6, 5))
        im_wa = ax_wa.imshow(data, cmap='RdYlBu_r', alpha=0.8)
        ax_wa.set_title(f'{community_name} Flood Risk\n{date_str}', fontsize=12)
        ax_wa.axis('off')
        
        # Add simple legend
        red_patch = mpatches.Patch(color='#d73027', label='High Risk')
        blue_patch = mpatches.Patch(color='#4575b4', label='Safe')
        ax_wa.legend(handles=[red_patch, blue_patch], loc='lower center', 
                    bbox_to_anchor=(0.5, -0.1), ncol=2)
        
        whatsapp_path = f"{output_base}_whatsapp.jpg"
        plt.savefig(whatsapp_path, dpi=150, bbox_inches='tight', 
                   format='jpg', quality=85)
        output_paths['whatsapp'] = whatsapp_path
        plt.close(fig_wa)
        
    except Exception as e:
        logger.error(f"Failed to convert GeoTIFF: {e}")
        
    return output_paths


def generate_audio_alert(text: str, language: str, output_path: str) -> bool:
    """
    Generate audio file from text in specified language.
    
    Uses gTTS for languages with good support, pyttsx3 as fallback.
    
    Args:
        text: Text to convert to speech
        language: Language code
        output_path: Path for output audio file
        
    Returns:
        bool: Success status
    """
    logger.info(f"Generating audio alert in {language}...")
    
    try:
        # Map language codes to gTTS codes
        gtts_langs = {
            'sw': 'sw',      # Swahili
            'en': 'en',      # English
            'fr': 'fr',      # French (for Rwanda as alternative)
        }
        
        if language in gtts_langs:
            # Use gTTS for supported languages
            tts = gTTS(text=text, lang=gtts_langs.get(language, 'en'), slow=False)
            tts.save(output_path)
            logger.info(f"Generated audio using gTTS: {output_path}")
            
        else:
            # Use pyttsx3 as fallback
            engine = pyttsx3.init()
            
            # Adjust speech rate for clarity
            engine.setProperty('rate', 150)  # Slower for better understanding
            
            # Try to find appropriate voice
            voices = engine.getProperty('voices')
            for voice in voices:
                if language.lower() in voice.id.lower():
                    engine.setProperty('voice', voice.id)
                    break
            
            # Save to file
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            logger.info(f"Generated audio using pyttsx3: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        return False


def send_whatsapp_message(phone_number: str, message: str, 
                         media_url: Optional[str] = None) -> bool:
    """
    Send WhatsApp message using Twilio API.
    
    Args:
        phone_number: Recipient phone number with country code
        message: Text message
        media_url: Optional URL to media file (image/audio)
        
    Returns:
        bool: Success status
    """
    try:
        # Get Twilio credentials from environment
        account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
        from_whatsapp = os.environ.get('TWILIO_WHATSAPP_FROM', 'whatsapp:+14787806356')
        
        if not account_sid or not auth_token:
            logger.warning("Twilio credentials not found in environment")
            return False
        
        # Initialize Twilio client
        client = Client(account_sid, auth_token)
        
        # Ensure phone number has whatsapp: prefix
        if not phone_number.startswith('whatsapp:'):
            phone_number = f'whatsapp:{phone_number}'
        
        # Create message
        message_params = {
            'from_': from_whatsapp,
            'to': phone_number,
            'body': message
        }
        
        # Add media if provided
        if media_url:
            message_params['media_url'] = [media_url]
        
        # Send message
        message = client.messages.create(**message_params)
        
        logger.info(f"WhatsApp message sent: {message.sid}")
        return True
        
    except TwilioRestException as e:
        logger.error(f"Twilio error: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to send WhatsApp: {e}")
        return False


def send_sms_message(phone_number: str, message: str) -> bool:
    """
    Send SMS message using Twilio API.
    
    Args:
        phone_number: Recipient phone number with country code
        message: Text message (will be truncated to 160 chars if needed)
        
    Returns:
        bool: Success status
    """
    try:
        # Get Twilio credentials
        account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
        from_phone = os.environ.get('TWILIO_PHONE_FROM')
        
        if not all([account_sid, auth_token, from_phone]):
            logger.warning("Twilio SMS credentials not complete")
            return False
        
        # Initialize client
        client = Client(account_sid, auth_token)
        
        # Truncate message if too long
        if len(message) > 160:
            message = message[:157] + "..."
        
        # Send SMS
        message = client.messages.create(
            from_=from_phone,
            to=phone_number,
            body=message
        )
        
        logger.info(f"SMS sent: {message.sid}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send SMS: {e}")
        return False


def load_llama_model(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    Load LLaMA model with automatic GPU/CPU selection.
    
    Tries MPS (Metal) GPU first via transformers, falls back to 
    quantized llama-cpp-python for CPU/low memory.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Model instance and tokenizer/config
    """
    logger.info(f"Loading LLaMA model: {model_name}")
    
    # Check available memory
    import psutil
    available_ram = psutil.virtual_memory().available / (1024**3)  # GB
    logger.info(f"Available RAM: {available_ram:.1f} GB")
    
    # Check for MPS (Metal Performance Shaders) availability
    has_mps = torch.backends.mps.is_available()
    logger.info(f"Apple Metal GPU (MPS) available: {has_mps}")
    
    try:
        if has_mps and available_ram > 16:
            # Use transformers with MPS acceleration
            logger.info("Loading model with Metal GPU acceleration...")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="mps",
                low_cpu_mem_usage=True
            )
            
            return model, tokenizer, "transformers"
            
        else:
            # Use quantized llama-cpp-python for efficiency
            logger.info("Loading 4-bit quantized model for CPU...")
            
            from llama_cpp import Llama
            
            # Download quantized model (example with common 4-bit quantization)
            # In production, you'd host this file somewhere accessible
            model_path = f"./{model_name.split('/')[-1]}-q4_k_m.gguf"
            
            if not os.path.exists(model_path):
                logger.warning(f"Quantized model not found at {model_path}")
                logger.info("Please download a 4-bit quantized version of the model")
                logger.info("Example: https://huggingface.co/TheBloke")
                # For demo, create a mock model
                return None, None, "mock"
            
            # Load with Metal acceleration if available
            n_gpu_layers = -1 if has_mps else 0  # -1 means all layers on GPU
            
            model = Llama(
                model_path=model_path,
                n_ctx=2048,  # Context window
                n_threads=8,  # CPU threads
                n_gpu_layers=n_gpu_layers,
                seed=42
            )
            
            return model, None, "llama_cpp"
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Falling back to mock model for demo")
        return None, None, "mock"


@lru_cache(maxsize=128)
def translate_alert(text: str, target_lang: str, model_info: tuple) -> str:
    """
    Translate alert text to target language with caching.
    
    Uses LRU cache to avoid re-translating identical messages.
    
    Args:
        text: English text to translate
        target_lang: Target language code
        model_info: Tuple of (model, tokenizer, model_type)
        
    Returns:
        Translated text
    """
    model, tokenizer, model_type = model_info
    
    # If English requested or model unavailable, return original
    if target_lang == "en" or model_type == "mock":
        return text
    
    logger.info(f"Translating to {target_lang}...")
    
    try:
        if model_type == "transformers":
            # Transformers approach
            prompt = f"Translate the following emergency alert to {target_lang}. Maintain a calm, reassuring tone:\n\n{text}\n\nTranslation:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to("mps")
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )
            
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the translation part
            translation = translation.split("Translation:")[-1].strip()
            
        elif model_type == "llama_cpp":
            # llama-cpp-python approach
            prompt = f"Translate to {target_lang}: {text}"
            
            response = model(
                prompt,
                max_tokens=200,
                temperature=0.7,
                stop=["</s>", "\n\n"]
            )
            
            translation = response['choices'][0]['text'].strip()
            
        else:
            # Mock translation for demo
            lang_names = {
                "sw": "Swahili",
                "lg": "Luganda", 
                "rw": "Kinyarwanda",
                "xog": "Soga"
            }
            translation = f"[{lang_names.get(target_lang, target_lang)} translation]: {text}"
            
        return translation
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text  # Return original on error


def check_sentiment(text: str) -> str:
    """
    Check sentiment of generated alert using Cardiff NLP model.
    
    Args:
        text: Alert text to analyze
        
    Returns:
        Sentiment label: 'positive', 'negative', or 'neutral'
    """
    try:
        # For production, use: cardiffnlp/twitter-xlm-roberta-base-sentiment
        # Here we'll use a simple heuristic for demo
        negative_words = ['danger', 'emergency', 'urgent', 'severe', 'crisis']
        positive_words = ['safe', 'calm', 'prepared', 'together', 'help']
        
        text_lower = text.lower()
        neg_count = sum(1 for word in negative_words if word in text_lower)
        pos_count = sum(1 for word in positive_words if word in text_lower)
        
        if neg_count > pos_count + 1:
            return "negative"
        elif pos_count > neg_count:
            return "positive"
        else:
            return "neutral"
            
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return "neutral"


def generate_flood_alert(args: argparse.Namespace, event_date: str,
                        analytics_summary: str, model_info: tuple) -> Dict[str, str]:
    """
    Generate culturally-sensitive flood alert with translation.
    
    Args:
        args: Command line arguments
        event_date: Date of flood event
        analytics_summary: Technical summary of flood indicators
        model_info: LLaMA model information tuple
        
    Returns:
        Dictionary with translated and English alerts
    """
    logger.info(f"Generating flood alert for {event_date}")
    
    # Create prompt for alert generation
    prompt = f"""Assume the role of a trusted community-radio reporter in {args.country}.
Craft a concise, reassuring flood warning in {args.target_lang} for residents of
{args.community}, {args.subcounty}, {args.county}. ≤ 60 s airtime.
Include one actionable safety tip and mention a familiar landmark near Lake Victoria.
Event date: {event_date}. Technical summary: {analytics_summary}."""
    
    # Generate English version first
    english_alert = f"""Good {('morning' if datetime.now().hour < 12 else 'afternoon')}, 
dear residents of {args.community}. This is a friendly flood preparedness message for {event_date}.

Recent rainfall and satellite observations suggest increased water levels in our area. 
As a precaution, please ensure your important documents are in waterproof containers 
and identify the safest route to higher ground. Remember, the community center near 
the old fish market by Lake Victoria serves as our designated safe gathering point.

Stay tuned to this station for updates. Together, we keep {args.community} safe."""
    
    # Check sentiment
    sentiment = check_sentiment(english_alert)
    if sentiment == "negative":
        # Regenerate with calmer tone
        english_alert = english_alert.replace("flood preparedness", "weather awareness")
        english_alert = english_alert.replace("increased water levels", "changing water conditions")
    
    # Translate if needed
    translated = translate_alert(english_alert, args.target_lang, model_info)
    
    return {
        "english": english_alert,
        "translated": translated,
        "language": args.target_lang
    }


def save_alert_outputs(args: argparse.Namespace, alerts: List[Dict],
                      risk_images: ee.ImageCollection):
    """
    Save alert texts, images, audio, and send communications.
    
    Args:
        args: Command line arguments
        alerts: List of alert dictionaries
        risk_images: Collection of flood risk images
    """
    # Create output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving outputs to {output_dir}")
    
    # Save each alert
    for i, alert in enumerate(alerts):
        date_str = alert['date'].replace('-', '')
        filename_base = f"{args.country}_{args.community}_{date_str}"
        
        # Save text alert
        text_path = output_dir / f"{filename_base}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"Radio Flood Alert – {args.county}, {args.country} – {alert['date']}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"🔊 {alert['content']['translated']}\n\n")
            f.write("(English backup:)\n")
            f.write(f"{alert['content']['english']}\n")
            
        logger.info(f"Saved alert text: {text_path}")
        
        # Export GeoTIFF
        risk_image = risk_images.filter(
            ee.Filter.eq('date_string', alert['date'])
        ).first()
        
        if risk_image:
            geotiff_path = export_geotiff(
                risk_image, 
                str(output_dir / filename_base),
                alert['aoi']
            )
            
            if geotiff_path:
                # Convert to JPEG and PDF
                image_paths = convert_geotiff_to_images(
                    geotiff_path,
                    str(output_dir / filename_base),
                    args.community,
                    alert['date']
                )
                
                # Generate audio alerts
                audio_path_local = output_dir / f"{filename_base}_{args.target_lang}.mp3"
                audio_success = generate_audio_alert(
                    alert['content']['translated'],
                    args.target_lang,
                    str(audio_path_local)
                )
                
                # Also generate English audio
                audio_path_en = output_dir / f"{filename_base}_en.mp3"
                generate_audio_alert(
                    alert['content']['english'],
                    'en',
                    str(audio_path_en)
                )
                
                # Send WhatsApp to community leader
                if args.leader_phone:
                    logger.info(f"Sending WhatsApp to community leader: {args.leader_phone}")
                    
                    # Send text message
                    leader_message = f"""🚨 Flood Alert - {args.community} 🚨

{alert['content']['translated']}

Date: {alert['date']}
Please share with community members.

English: {alert['content']['english'][:200]}..."""
                    
                    send_whatsapp_message(args.leader_phone, leader_message)
                    
                    # Send image if available
                    if 'whatsapp' in image_paths:
                        # Note: In production, you'd upload to a public URL first
                        logger.info("WhatsApp image sending requires public URL hosting")
                
                # Send WhatsApp to radio station
                if args.radio_phone:
                    logger.info(f"Sending WhatsApp to radio station: {args.radio_phone}")
                    
                    radio_message = f"""📻 BROADCAST ALERT - {args.community} 📻

{alert['content']['translated']}

Duration: ~60 seconds
Date: {alert['date']}
Language: {args.target_lang}

Please broadcast during next bulletin."""
                    
                    send_whatsapp_message(args.radio_phone, radio_message)
                
                # Send SMS to radio station
                if args.radio_sms:
                    logger.info(f"Sending SMS to radio station: {args.radio_sms}")
                    
                    sms_message = f"FLOOD ALERT {args.community} {alert['date']}: Check WhatsApp/Email for full broadcast text in {args.target_lang}"
                    send_sms_message(args.radio_sms, sms_message)
    
    logger.info(f"Completed processing {len(alerts)} alerts")


def main():
    """
    Main execution flow with comprehensive error handling.
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging to file
        log_file = Path(args.outdir) / "flood_alerts.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        logger.info("=" * 70)
        logger.info("Lake Victoria Basin Flood Alert System")
        logger.info(f"Location: {args.community}, {args.subcounty}, {args.county}, {args.country}")
        logger.info(f"Period: {args.start_date} to {args.end_date}")
        logger.info("=" * 70)
        
        # Initialize Earth Engine
        initialize_earth_engine()
        
        # Get community location
        lat, lon = get_community_location(
            args.country, args.county, args.subcounty, args.community
        )
        
        # Create area of interest
        aoi = create_aoi_buffer(lat, lon, buffer_km=25)
        
        # Process flood indicators
        logger.info("Processing flood indicators...")
        
        # 1. Rainfall analysis
        rainfall_data = calculate_rainfall_percentiles(
            aoi, 
            args.start_date.isoformat(),
            args.end_date.isoformat()
        )
        
        # 2. Water extent anomaly
        water_anomaly = calculate_water_anomaly(
            aoi,
            args.start_date.year,
            args.start_date.month
        )
        
        # 3. SAR flood detection
        sar_floods = process_sentinel1_flood(
            aoi,
            args.start_date.isoformat(),
            args.end_date.isoformat()
        )
        
        # 4. Optical water detection
        optical_water = process_optical_water(
            aoi,
            args.start_date.isoformat(),
            args.end_date.isoformat()
        )
        
        # Fuse indicators
        flood_risk = fuse_flood_indicators(
            rainfall_data, water_anomaly, sar_floods, optical_water, aoi
        )
        
        # Load translation model once
        model_info = load_llama_model()
        
        # Check each date for alerts
        alerts = []
        dates = pd.date_range(args.start_date, args.end_date, freq='D')
        
        for date in tqdm(dates, desc="Checking flood risk"):
            date_str = date.strftime('%Y-%m-%d')
            
            # Get risk image for this date
            risk_image = flood_risk.filter(
                ee.Filter.eq('date_string', date_str)
            ).first()
            
            if risk_image and check_alert_threshold(risk_image, aoi):
                # Generate alert
                analytics = f"Elevated flood risk detected on {date_str}. Multiple indicators suggest potential flooding."
                
                alert_content = generate_flood_alert(
                    args, date_str, analytics, model_info
                )
                
                alerts.append({
                    'date': date_str,
                    'content': alert_content,
                    'aoi': aoi
                })
                
                # Print to stdout
                print(f"\n{'='*70}")
                print(f"Radio Flood Alert – {args.county}, {args.country} – {date_str}")
                print(f"{'='*70}")
                print(f"🔊 {alert_content['translated']}\n")
                print("(English backup:)")
                print(alert_content['english'])
                print(f"{'='*70}\n")

        # Check each date for flood risk
        for date in tqdm(dates, desc="Checking flood risk"):
            date_str = date.strftime('%Y-%m-%d')
            
            # Get risk image for this date
            risk_image_filtered = flood_risk.filter(
                ee.Filter.eq('date_string', date_str)
            )
            
            # First check if collection has any images
            try:
                has_image = risk_image_filtered.size().gt(0).getInfo()
                
                if not has_image:
                    logger.debug(f"No risk image available for {date_str}")
                    continue
                    
                risk_image = risk_image_filtered.first()
                
                # Additional check to ensure image is valid
                if risk_image is None:
                    logger.debug(f"Risk image is null for {date_str}")
                    continue
                    
                # Now check threshold
                if check_alert_threshold(risk_image, aoi):
                    # Generate alert
                    analytics = f"Elevated flood risk detected on {date_str}. Multiple indicators suggest potential flooding."
                    
                    alert_content = generate_flood_alert(
                        args, date_str, analytics, model_info
                    )
                    
                    alerts.append({
                        'date': date_str,
                        'content': alert_content,
                        'aoi': aoi
                    })
                    
                    # Print to stdout
                    print(f"\n{'='*70}")
                    print(f"Radio Flood Alert – {args.county}, {args.country} – {date_str}")
                    print(f"{'='*70}")
                    print(f"🔊 {alert_content['translated']}\n")
                    print("(English backup:)")
                    print(alert_content['english'])
                    print(f"{'='*70}\n")
                    
            except Exception as e:
                logger.debug(f"Error processing date {date_str}: {e}")
                continue
        
        # Save all outputs and send communications
        if alerts:
            save_alert_outputs(args, alerts, flood_risk)
            logger.info(f"Generated {len(alerts)} flood alerts")
            
            # Summary message
            print(f"\n✅ Successfully generated {len(alerts)} flood alerts")
            print(f"📁 Outputs saved to: {args.outdir}")
            if args.leader_phone or args.radio_phone:
                print("📱 Messages sent to configured contacts")
        else:
            logger.info("No flood alerts needed for this period")
            print("\n✅ No flood conditions detected during the specified period.")
        
        logger.info("Flood alert processing completed successfully")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check flood_alerts.log for details", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()