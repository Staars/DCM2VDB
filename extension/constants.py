"""Constants for DICOM visualization"""

# Hounsfield Unit thresholds
HU_AIR_THRESHOLD = -200.0
HU_FAT = -100.0
HU_SOFT_TISSUE = 40.0
HU_BONE_START = 200.0
HU_BONE_DENSE = 400.0

# Tissue colors (RGBA)
COLOR_AIR = (0.01, 0.01, 0.02, 1.0)
COLOR_FAT = (0.9, 0.8, 0.4, 1.0)
COLOR_SOFT_TISSUE_RED = (0.6, 0.4, 0.3, 1.0)
COLOR_SOFT_TISSUE_PINK = (0.9, 0.7, 0.6, 1.0)
COLOR_MUSCLE = (0.7, 0.3, 0.3, 1.0)
COLOR_BONE_LIGHT = (0.85, 0.82, 0.7, 1.0)
COLOR_BONE_DENSE = (0.95, 0.92, 0.8, 1.0)
COLOR_BONE_WHITE = (1.0, 1.0, 1.0, 1.0)

# Material settings
DENSITY_SCALE_DEFAULT = 0.01
VIEWPORT_DENSITY_DEFAULT = 0.005
SUBSURFACE_WEIGHT = 0.1
SUBSURFACE_RADIUS = (0.01, 0.005, 0.003)

# Volume settings
MIN_SLICES_REQUIRED = 2
EXTREME_NEGATIVE_THRESHOLD = -2000
EXTREME_NEGATIVE_CLAMP = -1024

# Fixed HU range for consistent normalization across all volumes
HU_MIN_FIXED = -1024.0  # Standard CT minimum (air)
HU_MAX_FIXED = 3071.0   # Standard CT maximum (12-bit range)

# Tissue HU ranges (medical standard values)
HU_FAT_MIN = -120.0
HU_FAT_MAX = -90.0

HU_SOFT_MIN = 20.0
HU_SOFT_MAX = 70.0

HU_BONE_MIN = 400.0
HU_BONE_MAX = 1000.0

# Tissue colors (RGB only, alpha controlled by UI sliders)
COLOR_FAT_RGB = (0.776, 0.565, 0.018)      # Yellow
COLOR_SOFT_RGB = (0.906, 0.071, 0.029)     # Red
COLOR_BONE_RGB = (1.0, 1.0, 1.0)           # White
COLOR_AIR_RGB = (0.0, 0.0, 0.0)            # Black

# Default tissue opacity values
ALPHA_FAT_DEFAULT = 0.059
ALPHA_SOFT_DEFAULT = 1.0
ALPHA_BONE_DEFAULT = 1.0

# Preview settings
PREVIEW_ICON_COUNT = 5
PREVIEW_THUMBNAIL_SIZE = 128

# Geometry nodes defaults
GEONODES_DEFAULT_THRESHOLD = -200.0
GEONODES_THRESHOLD_OFFSET = 1.0
