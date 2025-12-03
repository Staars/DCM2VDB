"""Utility functions for volume handling"""

import os
import glob
import tempfile
from .dicom_io import log

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
        import numpy as np
        
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
        log(f"DEBUG: Saved middle slice to {debug_path}")
        return debug_path
    except Exception as e:
        log(f"Could not save debug slice: {e}")
        return None
