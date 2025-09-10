# requirements_minimal.txt - For Raspberry Pi and small devices
# Total install size: ~500MB (without optional packages)

# ===== Core Dependencies (Required) =====
# Google Earth Engine
earthengine-api==0.1.386
google-auth==2.26.1
google-auth-oauthlib==1.2.0

# Data processing (minimal versions)
pandas==2.0.3
numpy==1.24.3
python-dateutil==2.8.2

# Utilities
tqdm==4.66.1
geopy==2.4.1
requests==2.31.0

# Audio generation
gTTS==2.5.0
pygame==2.5.2

# ===== Optional - Add if RAM > 2GB =====
# Mapping (adds ~200MB)
# geemap==0.30.2
# matplotlib==3.7.2
# Pillow==10.2.0
# rasterio==1.3.9

# ===== Optional - SMS/WhatsApp =====
# twilio==8.11.0

# ===== Optional - Advanced ML (needs 4GB+ RAM) =====
# For local language models (not recommended for Pi)
# torch==2.1.0+cpu  # CPU-only version
# transformers==4.36.0
# sentencepiece==0.1.99

# ===== Build tools (if needed) =====
# wheel>=0.40.0
# setuptools>=65.0.0