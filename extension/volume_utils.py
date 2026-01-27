"""Utility functions for volume handling"""

import os
import glob
import tempfile
import numpy as np
from typing import Optional, Literal
from numpy.typing import NDArray
from .utils import SimpleLogger
from .constants import (
    DENOISING_PERCENTILE_BLEND_MULTIPLIER,
    DENOISING_WIENER_SIZE_MULTIPLIER,
    DENOISING_MEDIAN_KERNEL_SIZE
)

# Get logger for this extension
log = SimpleLogger()

def clean_old_volumes(name_prefix: str = "CT_Volume") -> None:
    """
    Remove old volume objects, materials, and volume data.
    
    Args:
        name_prefix: Prefix of objects to remove (e.g., 'CT_Volume', 'MR_Volume')
    """
    import bpy
    
    for o in list(bpy.data.objects):
        if o.name.startswith(name_prefix): 
            bpy.data.objects.remove(o, do_unlink=True)
    
    for m in list(bpy.data.materials):
        if m.name.startswith(name_prefix): 
            bpy.data.materials.remove(m, do_unlink=True)
    
    for v in list(bpy.data.volumes):
        if v.name.startswith(name_prefix): 
            bpy.data.volumes.remove(v, do_unlink=True)

def hu_to_normalized(hu_value: float) -> float:
    """
    Convert Hounsfield Unit to normalized 0-1 value using fixed range.
    
    Args:
        hu_value: Hounsfield Unit value
        
    Returns:
        Normalized value between 0 and 1
    """
    from .constants import HU_MIN_FIXED, HU_MAX_FIXED
    return (hu_value - HU_MIN_FIXED) / (HU_MAX_FIXED - HU_MIN_FIXED)

def save_debug_slice(volume_array: NDArray[np.float32], 
                     output_name: str = "dicom_middle_slice_debug.png") -> Optional[str]:
    """
    Save middle slice of volume as PNG for debugging.
    
    Args:
        volume_array: 3D numpy array (depth, height, width)
        output_name: Output filename
        
    Returns:
        Path to saved image, or None if failed
    """
    try:
        from PIL import Image
        
        depth = volume_array.shape[0]
        middle_slice = volume_array[depth//2, :, :]
        
        # Normalize for viewing
        slice_min, slice_max = middle_slice.min(), middle_slice.max()
        if slice_max > slice_min:
            normalized = ((middle_slice - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(middle_slice, dtype=np.uint8)
        
        debug_path = os.path.join(tempfile.gettempdir(), output_name)
        img = Image.fromarray(normalized, mode='L')
        img.save(debug_path)
        log.debug(f"Saved middle slice to {debug_path}")
        return debug_path
    except Exception as e:
        log.warning(f"Could not save debug slice: {e}")
        return None


def denoise_slice_scipy(
    slice_array: NDArray[np.float32], 
    method: Literal['GAUSSIAN', 'PERCENTILE_25', 'PERCENTILE_75', 'MEDIAN', 'GAUSSIAN_3D'] = 'GAUSSIAN',
    strength: float = 1.0
) -> NDArray[np.float32]:
    """
    Denoise a single 2D slice using scipy.ndimage - FAST and high quality!
    
    Args:
        slice_array: 2D numpy array (single slice)
        method: Denoising method to use
        strength: Filter strength (0.01-1.0, where 0.1=subtle, 0.5=strong)
    
    Returns:
        Denoised 2D numpy array with same dtype as input
    """
    try:
        from scipy import ndimage
    except ImportError:
        log.error("scipy not available, skipping denoising")
        return slice_array
    
    # Apply filter based on method
    if method == 'GAUSSIAN':
        # Gaussian filter - sigma scales directly with strength (0.01-0.5)
        result = ndimage.gaussian_filter(slice_array, sigma=strength)
    
    elif method == 'PERCENTILE_25':
        # 25th percentile filter - darkens slightly, less aggressive than median
        size = DENOISING_MEDIAN_KERNEL_SIZE  # Always use smallest kernel
        filtered = ndimage.percentile_filter(slice_array, percentile=25, size=size)
        
        # Blend with original
        blend_factor = strength * DENOISING_PERCENTILE_BLEND_MULTIPLIER
        blend_factor = min(blend_factor, 1.0)
        result = (1.0 - blend_factor) * slice_array + blend_factor * filtered
        log.info(f"  Percentile 25% filter with {blend_factor*100:.1f}% blend")
    
    elif method == 'PERCENTILE_75':
        # 75th percentile filter - brightens slightly, less aggressive than median
        size = DENOISING_MEDIAN_KERNEL_SIZE  # Always use smallest kernel
        filtered = ndimage.percentile_filter(slice_array, percentile=75, size=size)
        
        # Blend with original
        blend_factor = strength * DENOISING_PERCENTILE_BLEND_MULTIPLIER
        blend_factor = min(blend_factor, 1.0)
        result = (1.0 - blend_factor) * slice_array + blend_factor * filtered
        log.info(f"  Percentile 75% filter with {blend_factor*100:.1f}% blend")
    
    elif method == 'MEDIAN':
        # Median filter with blending for subtle control
        # Always use DENOISING_MEDIAN_KERNEL_SIZE kernel (minimal), but blend with original based on strength
        size = DENOISING_MEDIAN_KERNEL_SIZE  # Always use smallest kernel
        filtered = ndimage.median_filter(slice_array, size=size)
        
        # Blend: result = (1-strength)*original + strength*filtered
        blend_factor = strength * DENOISING_PERCENTILE_BLEND_MULTIPLIER  # Scale 0.01-0.5 to 0.02-1.0 range
        blend_factor = min(blend_factor, 1.0)  # Cap at 1.0
        
        result = (1.0 - blend_factor) * slice_array + blend_factor * filtered
        
        log.debug(f"  Median filter ({DENOISING_MEDIAN_KERNEL_SIZE}x{DENOISING_MEDIAN_KERNEL_SIZE} kernel) with {blend_factor*100:.1f}% blend")
    
    else:
        result = slice_array
    
    return result.astype(slice_array.dtype)
