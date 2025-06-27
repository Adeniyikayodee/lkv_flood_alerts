# Lake Victoria Basin Flood Alert System - Setup Guide

## Overview

This system generates culturally-sensitive flood alerts for community radio stations in the Lake Victoria Basin (Uganda, Kenya, Tanzania, Rwanda) using Google Earth Engine satellite data and AI-powered local language translation.

**Key Features:**
- Multi-satellite flood detection (CHIRPS, Sentinel-1/2, Landsat, JRC Water)
- AI translation to local languages (Swahili, Luganda, Kinyarwanda)
- WhatsApp/SMS integration via Twilio
- Audio alert generation with Text-to-Speech
- PDF/JPEG map generation
- Optimized for Apple Silicon (M1/M2/M3/M4) MacBooks

## Quick Start (Apple Silicon Mac)

### 1. Prerequisites
- macOS 12+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 (recommended) or 3.10+
- Git
- Google Earth Engine account
- Twilio account (optional, for WhatsApp/SMS)

### 2. Setup Environment

```bash
# Clone or download the script files
cd /path/to/your/project

# Create Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Set environment variables for Metal GPU acceleration
export CMAKE_ARGS="-DLLAMA_METAL=on"

# Upgrade pip and install build tools
pip install --upgrade pip wheel cmake ninja

# Install Python dependencies (this may take 10-15 minutes)
pip install -r requirements.txt

# Install llama-cpp-python with Metal GPU support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

### 3. Google Earth Engine Setup

```bash
# Install and authenticate Google Earth Engine
pip install --upgrade earthengine-api
earthengine authenticate --force

# Set up a GEE project (replace YOUR_PROJECT_ID)
# 1. Go to https://console.cloud.google.com/apis/library/earthengine.googleapis.com
# 2. Select or create a project → Enable the Earth Engine API
# 3. Run:
earthengine set_project YOUR_PROJECT_ID
```

### 4. Optional: WhatsApp/SMS Setup (Twilio)

```bash
# Set Twilio environment variables
export TWILIO_ACCOUNT_SID="your_account_sid"
export TWILIO_AUTH_TOKEN="your_auth_token"
export TWILIO_WHATSAPP_FROM="whatsapp:+14155238886"  # Twilio sandbox
```

### 5. Verify Installation

```bash
# Test the setup
python - <<'PY'
import torch, numpy, platform, sys
print("✅ Torch version :", torch.__version__)
print("✅ NumPy version :", numpy.__version__)  
print("✅ MPS available :", torch.backends.mps.is_available())
print("✅ Python exe    :", sys.executable)
print("✅ Platform      :", platform.platform())
PY

# Test Google Earth Engine
python - <<'PY'
import ee; ee.Initialize()
print("✅ EE ready for:", ee.AccountInfo()['email'])
PY
```

## Usage

### Command Line Interface

```bash
# Basic usage with CLI arguments
python lkv_flood_alerts.py \
    --country "Kenya" \
    --county "Kisumu" \
    --subcounty "Nyando" \
    --community "Obunga" \
    --start-date "2023-06-01" \
    --end-date "2023-06-30" \
    --target-lang "auto" \
    --outdir "./alerts"

# With communication options
python lkv_flood_alerts.py \
    --config params.json \
    --leader-phone "+254700123456" \
    --radio-phone "+254700789012" \
    --radio-sms "+254700789012"
```

### JSON Configuration

```bash
# Use JSON config file (recommended)
python lkv_flood_alerts.py --config params.json
```

The `params.json` file should contain:
```json
{
  "country": "Kenya",
  "county": "Kisumu", 
  "subcounty": "Nyando",
  "community": "Obunga",
  "start_date": "2025-06-01",
  "end_date": "2025-06-30",
  "target_lang": "auto",
  "outdir": "./alerts"
}
```

### Language Options

- `auto`: Automatically detect language by country
- `sw`: Swahili (Kenya, Tanzania)
- `lg`: Luganda (Uganda - Central)
- `xog`: Soga (Uganda - Southwest)  
- `rw`: Kinyarwanda (Rwanda)
- `en`: English (fallback)

## Output Files

For each flood alert, the system generates:

```
alerts/
├── Kenya_Obunga_20250615.txt          # Text alert (translated + English)
├── Kenya_Obunga_20250615.tif          # GeoTIFF flood risk map
├── Kenya_Obunga_20250615.jpg          # High-res JPEG map
├── Kenya_Obunga_20250615.pdf          # PDF report with metadata
├── Kenya_Obunga_20250615_whatsapp.jpg # WhatsApp-optimized image
├── Kenya_Obunga_20250615_sw.mp3       # Audio alert (local language)
└── Kenya_Obunga_20250615_en.mp3       # Audio alert (English backup)
```

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'ee'"**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall earthengine-api
pip install --upgrade earthengine-api
```

**2. "Failed to initialize Earth Engine"**
```bash
# Re-authenticate
earthengine authenticate --force

# Check project ID
earthengine set_project YOUR_PROJECT_ID
```

**3. "Metal GPU not available"**
```bash
# Check MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Reinstall llama-cpp-python with Metal
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall llama-cpp-python
```

**4. "Memory issues with AI models"**
- The system automatically uses quantized models for systems with <16GB RAM
- Close other applications to free memory
- Consider using CPU-only mode for very memory-constrained systems

### Performance Tips

**Apple Silicon Optimization:**
- The system automatically detects and uses Metal Performance Shaders (MPS)
- AI translation uses GPU acceleration when available
- Earth Engine processing is optimized for parallel execution

**Memory Management:**
- Models are loaded lazily (only when needed)
- Translation uses LRU caching to avoid re-processing
- Large rasters use streaming processing to avoid memory limits

## Advanced Configuration

### Custom Model Configuration

To use different AI models, modify the `load_llama_model()` function in `lkv_flood_alerts.py`:

```python
# Use a different model
model_info = load_llama_model("microsoft/DialoGPT-medium")
```

### Custom Thresholds

Flood detection thresholds can be adjusted in the script:

```python
# In check_alert_threshold() function
threshold = 0.15  # 15% of area must show flood risk (default)

# In fuse_flood_indicators() function  
# Adjust component weights:
rain_contrib = rain_flag.multiply(0.30)    # 30% rainfall weight
water_contrib = water_anom.multiply(0.20)  # 20% water anomaly
sar_contrib = sar_flood.multiply(0.50)     # 50% SAR flood detection
```

### Adding New Communities

Add coordinates to `OFFLINE_GEOCODES` in the script:

```python
OFFLINE_GEOCODES = {
    ("Kenya", "Kisumu", "Nyando", "Obunga"): (-0.1036, 34.7617),
    ("Your_Country", "Your_County", "Your_Subcounty", "Your_Community"): (lat, lon),
}
```

## Info

For technical issues:
1. Check the generated `flood_alerts.log` file
2. Ensure all environment variables are set correctly
3. Verify Google Earth Engine quota and permissions
4. Test with smaller date ranges if processing is slow


---

**Author:** Kayode Adeniyi - Lake Victoria Basin Flood Alert System
**Optimized for:** Apple Silicon MacBooks with Python 3.11 