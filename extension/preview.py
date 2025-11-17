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

def generate_series_preview_icons(series, dicom_root_path, preview_collection):
    """
    Generate 5 preview icons for a series (evenly distributed slices).
    
    Args:
        series: SeriesInfo object
        dicom_root_path: Root path to DICOM files
        preview_collection: bpy.utils.previews collection
    
    Returns:
        List of icon_ids
    """
    import os
    import tempfile
    from PIL import Image
    
    icon_ids = []
    
    print(f"[DICOM Preview] Generating icons for series: {series.series_description}")
    print(f"[DICOM Preview] Root path: {dicom_root_path}")
    print(f"[DICOM Preview] File paths count: {len(series.file_paths)}")
    
    # Select 5 evenly distributed slices
    slice_count = len(series.file_paths)
    if slice_count == 0:
        print("[DICOM Preview] No file paths in series")
        return icon_ids
    
    # Calculate indices for 5 slices
    if slice_count <= 5:
        indices = list(range(slice_count))
    else:
        step = (slice_count - 1) / 4  # Distribute across first to last
        indices = [int(i * step) for i in range(5)]
    
    print(f"[DICOM Preview] Selected indices: {indices}")
    
    # Generate preview for each selected slice
    for i, idx in enumerate(indices):
        try:
            # Build absolute path
            rel_path = series.file_paths[idx]
            abs_path = os.path.join(dicom_root_path, rel_path)
            
            print(f"[DICOM Preview] Loading slice {i}: {abs_path}")
            
            # Load slice
            slice_data = load_slice(abs_path)
            pixels = slice_data["pixels"]
            
            # Apply window/level
            wc = series.window_center or slice_data.get('window_center')
            ww = series.window_width or slice_data.get('window_width')
            
            if wc is not None and ww is not None and ww > 0:
                low = wc - ww / 2
                high = wc + ww / 2
                pixels_windowed = np.clip(pixels, low, high)
                normalized = ((pixels_windowed - low) / ww * 255).astype(np.uint8)
            else:
                pmin, pmax = np.percentile(pixels, [1, 99])
                if pmax > pmin:
                    normalized = np.clip((pixels - pmin) / (pmax - pmin) * 255, 0, 255).astype(np.uint8)
                else:
                    normalized = np.zeros_like(pixels, dtype=np.uint8)
            
            # Save as temporary PNG
            temp_path = os.path.join(tempfile.gettempdir(), f"dicom_preview_{series.series_instance_uid}_{i}.png")
            img = Image.fromarray(normalized, mode='L')
            
            # Resize to small icon (32x32)
            img = img.resize((32, 32), Image.Resampling.LANCZOS)
            img.save(temp_path)
            
            # Load into preview collection
            icon_key = f"{series.series_instance_uid}_{i}"
            
            # Check if key already exists, if so use existing preview
            if icon_key in preview_collection:
                preview = preview_collection[icon_key]
                print(f"[DICOM Preview] Using cached icon {i}: icon_id={preview.icon_id}")
            else:
                preview = preview_collection.load(icon_key, temp_path, 'IMAGE')
                print(f"[DICOM Preview] Generated new icon {i}: icon_id={preview.icon_id}")
            
            icon_ids.append(preview.icon_id)
            
        except Exception as e:
            print(f"[DICOM Preview] Failed to generate icon {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"[DICOM Preview] Total icons generated: {len(icon_ids)}")
    return icon_ids
