"""DICOM image preview functionality"""

import bpy
import numpy as np
from .dicom_io import load_slice

def load_and_display_slice(context, filepath, series):
    """Load a DICOM slice into Blender image for preview"""
    slice_data = load_slice(filepath)
    pixels = slice_data["pixels"]
    
    # Apply window/level if available
    wc = series.get('window_center') or slice_data.get('window_center')
    ww = series.get('window_width') or slice_data.get('window_width')
    
    if wc is not None and ww is not None and ww > 0:
        low = wc - ww / 2
        high = wc + ww / 2
        pixels_windowed = np.clip(pixels, low, high)
        normalized = ((pixels_windowed - low) / ww).astype(np.float32)
    else:
        # Auto window/level using percentiles
        pmin, pmax = np.percentile(pixels, [1, 99])
        if pmax > pmin:
            normalized = np.clip((pixels - pmin) / (pmax - pmin), 0, 1).astype(np.float32)
        else:
            normalized = np.zeros_like(pixels, dtype=np.float32)
    
    height, width = normalized.shape
    
    # Create or update Blender image
    img_name = "DICOM_Preview"
    if img_name in bpy.data.images:
        img = bpy.data.images[img_name]
        if img.size[0] != width or img.size[1] != height:
            bpy.data.images.remove(img)
            img = bpy.data.images.new(img_name, width, height, alpha=False, float_buffer=True)
    else:
        img = bpy.data.images.new(img_name, width, height, alpha=False, float_buffer=True)
    
    # Convert to RGBA for Blender (grayscale -> RGB)
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[:, :, 0] = normalized
    rgba[:, :, 1] = normalized
    rgba[:, :, 2] = normalized
    rgba[:, :, 3] = 1.0
    
    # Flip vertically (Blender expects bottom-to-top)
    rgba = np.flipud(rgba)
    
    # Flatten and assign to image
    img.pixels[:] = rgba.ravel()
    img.update()
    
    # Force preview icon to regenerate
    if img.preview:
        # Clear the old preview to force regeneration
        img.preview.image_size = (0, 0)
    img.preview_ensure()
    
    # Mark areas for redraw
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()