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
