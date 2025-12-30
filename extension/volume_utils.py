"""Utility functions for volume handling"""

import os
import glob
import tempfile
import numpy as np
from .utils import SimpleLogger

# Get logger for this extension
log = SimpleLogger()

def clean_old_volumes(name_prefix="CT_Volume"):
    """Remove old volume objects, materials, and volume data"""
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

def hu_to_normalized(hu_value):
    """Convert Hounsfield Unit to normalized 0-1 value using fixed range"""
    from .constants import HU_MIN_FIXED, HU_MAX_FIXED
    return (hu_value - HU_MIN_FIXED) / (HU_MAX_FIXED - HU_MIN_FIXED)

def save_debug_slice(volume_array, output_name="dicom_middle_slice_debug.png"):
    """Save middle slice of volume as PNG for debugging"""
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


def denoise_slice_scipy(slice_array, method='GAUSSIAN', strength=1.0):
    """
    Denoise a single 2D slice using scipy.ndimage - FAST and high quality!
    
    Args:
        slice_array: 2D numpy array (single slice)
        method: 'GAUSSIAN' or 'MEDIAN'
        strength: filter strength (1.0 = mild, 2.0 = moderate, 3.0 = strong)
    
    Returns:
        Denoised 2D numpy array
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
        size = 3  # Always use smallest kernel
        filtered = ndimage.percentile_filter(slice_array, percentile=25, size=size)
        
        # Blend with original
        blend_factor = strength * 2.0
        blend_factor = min(blend_factor, 1.0)
        result = (1.0 - blend_factor) * slice_array + blend_factor * filtered
        log.info(f"  Percentile 25% filter with {blend_factor*100:.1f}% blend")
    
    elif method == 'PERCENTILE_75':
        # 75th percentile filter - brightens slightly, less aggressive than median
        size = 3  # Always use smallest kernel
        filtered = ndimage.percentile_filter(slice_array, percentile=75, size=size)
        
        # Blend with original
        blend_factor = strength * 2.0
        blend_factor = min(blend_factor, 1.0)
        result = (1.0 - blend_factor) * slice_array + blend_factor * filtered
        log.info(f"  Percentile 75% filter with {blend_factor*100:.1f}% blend")
    
    elif method == 'WIENER':
        # Wiener filter - adaptive noise reduction
        try:
            from scipy.signal import wiener
            # Map strength to window size with doubled range: 0.01-1.0 -> 3-21 pixels
            # This allows stronger denoising than before
            mysize = max(3, int(strength * 40 + 1))
            if mysize % 2 == 0:
                mysize += 1
            result = wiener(slice_array, mysize=mysize)
            log.info(f"  Wiener filter with {mysize}x{mysize} window (strength={strength:.2f})")
        except ImportError:
            log.info("  WARNING: scipy.signal not available, falling back to Gaussian")
            result = ndimage.gaussian_filter(slice_array, sigma=strength)
    
    elif method == 'MEDIAN':
        # Median filter with blending for subtle control
        # Always use 3x3 kernel (minimal), but blend with original based on strength
        size = 3  # Always use smallest kernel
        filtered = ndimage.median_filter(slice_array, size=size)
        
        # Blend: result = (1-strength)*original + strength*filtered
        blend_factor = strength * 2.0  # Scale 0.01-0.5 to 0.02-1.0 range
        blend_factor = min(blend_factor, 1.0)  # Cap at 1.0
        
        result = (1.0 - blend_factor) * slice_array + blend_factor * filtered
        
        log.debug(f"  Median filter (3x3 kernel) with {blend_factor*100:.1f}% blend")
    
    else:
        result = slice_array
    
    return result.astype(slice_array.dtype)
