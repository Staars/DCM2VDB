"""Constants for DICOM visualization"""

# Volume settings
MIN_SLICES_REQUIRED = 2
EXTREME_NEGATIVE_THRESHOLD = -2000
EXTREME_NEGATIVE_CLAMP = -1024

# Fixed HU range for consistent normalization across all volumes
HU_MIN_FIXED = -1024.0  # Standard CT minimum (air)
HU_MAX_FIXED = 3071.0   # Standard CT maximum (12-bit range)

# Preview settings
PREVIEW_ICON_COUNT = 5
PREVIEW_THUMBNAIL_SIZE = 128
MAX_PREVIEW_IMAGES = 100  # Maximum preview images in popup grid (10x10)
PREVIEW_POPUP_WIDTH = 600  # Width of preview popup window in pixels
PREVIEW_ICON_SIZE = 32  # Size of preview icons in pixels

# Material settings - Mesh material color ramp
MESH_COLOR_RAMP_POINTINESS_MIN = 0.414  # Color ramp position for dark pointiness
MESH_COLOR_RAMP_POINTINESS_MAX = 0.527  # Color ramp position for bright pointiness
MESH_COLOR_A_RGB = (0.7, 0.301, 0.117)  # Base color A (brownish)
MESH_COLOR_B_RGB = (0.95, 0.92, 0.85)  # Base color B (light beige)
MESH_ROUGHNESS = 0.4  # Default roughness for mesh materials
MESH_SPECULAR_MULTIPLIER = 0.5  # Multiplier for specular and sheen

# Material settings - Volume rendering
VOLUME_DENSITY_DISPLAY = 1.0  # Viewport display density for normalized data
VOLUME_PRINCIPLED_LOCATION = (600, 0)  # Node location for Volume Principled shader

# Geometry nodes - Mesh extraction
MESH_THRESHOLD_MAX = 10000  # Maximum threshold value for mesh extraction
MIN_ISLAND_VERTICES = 100  # Minimum vertices for island separation

# Denoising settings
DENOISING_PROGRESS_LOG_INTERVAL = 10  # Log progress every N percent
DENOISING_PERCENTILE_BLEND_MULTIPLIER = 2.0  # Multiplier for percentile filter blending
DENOISING_WIENER_SIZE_MULTIPLIER = 40  # Multiplier for Wiener filter window size
DENOISING_MEDIAN_KERNEL_SIZE = 3  # Kernel size for median filter

# Unit conversion
MM_TO_METERS = 0.001  # Millimeters to meters conversion
MM3_TO_ML = 1000.0  # Cubic millimeters to milliliters conversion

# File size thresholds
MIN_DICOM_FILE_SIZE = 1000  # Minimum file size in bytes to consider as DICOM

# DICOM SOP Class UIDs
SOP_CLASS_SECONDARY_CAPTURE = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage

# Percentile settings for auto window/level
PERCENTILE_MIN = 1  # Lower percentile for auto contrast
PERCENTILE_MAX = 99  # Upper percentile for auto contrast
