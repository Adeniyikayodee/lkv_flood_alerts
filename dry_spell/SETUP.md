# Nigerian LGA Dry Spell Alert System - Setup Guide

## Overview

A lightweight system for predicting and alerting communities about dry spells/drought conditions in Nigerian Local Government Areas (LGAs). Designed to run on small devices like Raspberry Pi, with automatic translation to major Nigerian languages.

## Features

- **Multi-source drought detection**: CHIRPS rainfall, MODIS NDVI/temperature, soil moisture
- **Nigerian language support**: Hausa, Yoruba, Igbo, English + regional languages
- **LGA-specific alerts**: Covers all 774 Nigerian LGAs
- **Low-resource optimized**: Runs on Raspberry Pi 4 (2GB RAM minimum)
- **Multiple alert channels**: SMS, WhatsApp, audio files for radio broadcast

## Quick Setup (Raspberry Pi / Small Device)

### 1. System Requirements

**Minimum Hardware:**
- Raspberry Pi 4 (2GB RAM) or equivalent
- 16GB SD card
- Internet connection
- Optional: USB speaker for audio testing

**Software:**
- Raspberry Pi OS (64-bit recommended) or Ubuntu Server
- Python 3.9+

### 2. Installation

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y python3-pip python3-venv git
sudo apt-get install -y libgdal-dev libspatialindex-dev
sudo apt-get install -y libatlas-base-dev libopenblas-dev

# Clone or copy the script
mkdir ~/dryspell_alerts
cd ~/dryspell_alerts

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install minimal dependencies
pip install --upgrade pip wheel

# Core requirements (lightweight)
pip install earthengine-api==0.1.386
pip install pandas numpy matplotlib
pip install geopy tqdm python-dateutil
pip install gTTS pygame requests

# Optional: For maps (adds ~200MB)
pip install geemap rasterio Pillow
```

### 3. Google Earth Engine Setup

```bash
# Authenticate (one-time)
earthengine authenticate --force

# Follow the browser prompts to authenticate

# Set project (get from Google Cloud Console)
earthengine set_project YOUR_PROJECT_ID
```

### 4. Create Configuration

Create `config.json`:

```json
{
  "state": "Kano",
  "lga": "Kano Municipal",
  "start_date": "2025-03-01",
  "end_date": "2025-05-31",
  "rain_threshold": 5.0,
  "dry_days": 10,
  "language": "auto"
}
```

### 5. Run the System

```bash
# Activate environment
source .venv/bin/activate

# Run with config file
python nigeria_dryspell_alerts.py --config config.json

# Or with command line arguments
python nigeria_dryspell_alerts.py \
  --state "Lagos" \
  --lga "Ikeja" \
  --start-date "2025-01-01" \
  --end-date "2025-03-31"
```

## Language Configuration

The system automatically selects languages based on geopolitical zones:

| Zone | Primary Language | States |
|------|-----------------|---------|
| North-West | Hausa (ha) | Sokoto, Kebbi, Zamfara, Katsina, Kaduna, Kano, Jigawa |
| North-East | Hausa (ha) | Borno, Yobe, Bauchi, Gombe, Adamawa, Taraba |
| North-Central | Hausa (ha) | Niger, Kogi, Benue, Plateau, Nasarawa, Kwara, FCT |
| South-West | Yoruba (yo) | Lagos, Ogun, Oyo, Osun, Ondo, Ekiti |
| South-East | Igbo (ig) | Anambra, Enugu, Imo, Abia, Ebonyi |
| South-South | English (en) | Edo, Delta, Rivers, Bayelsa, Cross River, Akwa Ibom |

Override with `--language` parameter if needed.

## Threshold Configuration

Adjust thresholds based on your region's climate:

| Region Type | Rain Threshold | Dry Days | Notes |
|------------|---------------|----------|-------|
| Arid (North) | 2-3mm | 20-30 | Sahel region |
| Semi-Arid | 5-8mm | 15-20 | Middle belt |
| Sub-humid | 10-15mm | 10-15 | Southern regions |
| Coastal | 15-20mm | 7-10 | Niger Delta |

## SMS/WhatsApp Setup (Optional)

For Twilio integration:

```bash
# Set environment variables
export TWILIO_ACCOUNT_SID="your_sid"
export TWILIO_AUTH_TOKEN="your_token"
export TWILIO_PHONE_FROM="+234xxxxxxxxx"

# Install Twilio
pip install twilio
```

## Output Files

The system generates:

```
dry_spell_alerts/
├── Kano_KanoMunicipal_20250315_alert.txt    # Text alert
├── Kano_KanoMunicipal_20250315_audio.mp3    # Audio for radio
├── Kano_KanoMunicipal_20250315_analysis.png # Risk maps
└── Kano_KanoMunicipal_20250315_risk.tif     # GeoTIFF data
```

## Automated Monitoring

Set up a cron job for daily checks:

```bash
# Edit crontab
crontab -e

# Add daily check at 6 AM
0 6 * * * cd /home/pi/dryspell_alerts && .venv/bin/python nigeria_dryspell_alerts.py --config config.json >> logs.txt 2>&1
```

## Memory Optimization for Small Devices

For devices with <2GB RAM:

```python
# In the script, reduce these values:
scale=2000  # Instead of 1000 (reduces resolution)
maxPixels=1e8  # Instead of 1e9
buffer_km=20  # Instead of 30
```

## Testing

Test with a known dry period:

```bash
# Test Maiduguri during dry season
python nigeria_dryspell_alerts.py \
  --state "Borno" \
  --lga "Maiduguri" \
  --start-date "2024-11-01" \
  --end-date "2024-12-31" \
  --rain-threshold 2.0 \
  --dry-days 20
```

## Troubleshooting

### "Earth Engine not initialized"
```bash
earthengine authenticate --force
earthengine set_project YOUR_PROJECT
```

### "Memory Error" on Raspberry Pi
- Reduce area buffer size
- Process shorter date ranges
- Increase swap space:
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### "No module named 'ee'"
```bash
source .venv/bin/activate
pip install earthengine-api
```

## API Limits

Google Earth Engine free tier limits:
- 40,000 requests/day
- 40 concurrent requests
- 1000 assets

For production use, consider:
- Caching results locally
- Processing in batches
- Using Earth Engine Apps for visualization

## Support & Resources

- **Agricultural Extension Services**: Contact your state's agricultural ministry
- **NIMET**: Nigerian Meteorological Agency (www.nimet.gov.ng)
- **Google Earth Engine**: developers.google.com/earth-engine
- **Community Forum**: Join Nigerian AgTech WhatsApp groups

## Sample Use Cases

### 1. Early Warning for Farmers
```bash
# Check next 3 months for planting season
python nigeria_dryspell_alerts.py \
  --state "Kaduna" \
  --lga "Zaria" \
  --start-date "2025-04-01" \
  --end-date "2025-06-30" \
  --phone "+2348012345678"
```

### 2. Government Monitoring
```bash
# Monitor multiple LGAs
for lga in "Kano Municipal" "Dala" "Fagge"; do
  python nigeria_dryspell_alerts.py \
    --state "Kano" \
    --lga "$lga" \
    --config state_config.json
done
```

### 3. Radio Station Alerts
Generate audio alerts for broadcast:
```bash
python nigeria_dryspell_alerts.py \
  --state "Oyo" \
  --lga "Ibadan North" \
  --language "yo" \
  --outdir "./radio_alerts"
```

---

**Version**: 1.0  
**License**: MIT  