"""Volume creation and OpenVDB handling"""

import bpy
import os
import tempfile
import numpy as np
import uuid
import math

from .dicom_io import log
from .constants import *
from .volume_utils import clean_temp_dir, clean_old_volumes, save_debug_slice
from .materials import create_volume_material, create_mesh_material
from .geometry_nodes import create_volume_to_mesh_geonodes

def create_volume(slices):
    """Create a volume object from DICOM slices with proper Hounsfield units."""
    clean_temp_dir()

    # Parse ImageOrientationPatient and ImagePositionPatient
    for slice_data in slices:
        orientation = slice_data["ds"].ImageOrientationPatient
        position = slice_data["ds"].ImagePositionPatient
        
        # Extract row and column direction cosines
        row_cosines = np.array(orientation[:3], dtype=np.float32)
        col_cosines = np.array(orientation[3:], dtype=np.float32)
        
        # Calculate the normal vector of the slice plane
        normal_vector = np.cross(row_cosines, col_cosines)
        slice_data["normal_vector"] = normal_vector
        slice_data["position"] = np.array(position, dtype=np.float32)
    
    # Sort slices based on their position along the normal vector
    slices = sorted(slices, key=lambda s: np.dot(s["position"], s["normal_vector"]))
    
    # Check if all slices have the same dimensions
    first_shape = slices[0]["pixels"].shape
    slices = [s for s in slices if s["pixels"].shape == first_shape]
    
    if len(slices) < MIN_SLICES_REQUIRED:
        raise ValueError(f"Need at least {MIN_SLICES_REQUIRED} slices with matching dimensions")
     
    vol = np.stack([s["pixels"] for s in slices])
    depth, height, width = vol.shape
    
    # PixelSpacing is [row spacing, column spacing] = [Y spacing, X spacing]
    pixel_spacing = slices[0]["pixel_spacing"]
    slice_thickness = slices[0]["slice_thickness"]
    
    # Validate slice thickness
    if not slice_thickness or slice_thickness <= 0:
        log(f"WARNING: Invalid slice_thickness ({slice_thickness}), using pixel spacing as fallback")
        slice_thickness = max(pixel_spacing)
    
    # spacing order: [X, Y, Z]
    spacing = [pixel_spacing[1], pixel_spacing[0], slice_thickness]
    
    # Log spacing values
    log(f"Pixel spacing (X, Y): {pixel_spacing}")
    log(f"Slice thickness (Z): {slice_thickness}")
    log(f"Spacing (X, Y, Z) in mm: {spacing}")
    
    # Calculate and log physical dimensions
    phys_x = width * spacing[0]
    phys_y = height * spacing[1]
    phys_z = depth * spacing[2]
    log(f"Physical dimensions (mm): {phys_x:.1f} x {phys_y:.1f} x {phys_z:.1f}")
    log(f"Physical dimensions (cm): {phys_x/10:.1f} x {phys_y/10:.1f} x {phys_z/10:.1f}")
    
    # Get value range for info
    vol_min, vol_max = vol.min(), vol.max()
    log(f"Creating volume: {width}x{height}x{depth}, spacing: {spacing}")
    log(f"RAW value range: {vol_min:.1f} to {vol_max:.1f} HU")
    
    # Clean up invalid values (padding, errors)
    if vol_min < EXTREME_NEGATIVE_THRESHOLD:
        log(f"WARNING: Found extreme negative values (min: {vol_min:.1f}), likely padding. Clamping to {EXTREME_NEGATIVE_CLAMP}")
        vol = np.clip(vol, EXTREME_NEGATIVE_CLAMP, vol_max)
        vol_min = vol.min()
    
    # Recalculate stats after cleaning
    vol_mean = vol.mean()
    vol_std = vol.std()
    log(f"CLEANED value range: {vol_min:.1f} to {vol_max:.1f} HU")
    log(f"Mean: {vol_mean:.1f}, Std: {vol_std:.1f}")
    log(f"Data type: {vol.dtype}, Shape: {vol.shape}")
    
    # Check for data issues
    unique_values = len(np.unique(vol))
    log(f"Unique values in volume: {unique_values}")
    if unique_values < 10:
        log("WARNING: Very few unique values - data might be corrupted or improperly scaled")
    
    # Sample some values to verify data looks reasonable
    sample_indices = [
        (depth//4, height//2, width//2),
        (depth//2, height//2, width//2),
        (3*depth//4, height//2, width//2)
    ]
    log("Sample voxel values (z=25%, 50%, 75% center):")
    for idx in sample_indices:
        log(f"  {idx}: {vol[idx]:.1f} HU")
    
    # Debug: Save middle slice as PNG
    save_debug_slice(vol)
    
    # Clean up old volumes
    clean_old_volumes("CT_Volume")

    # Save volume data to temporary VDB file with UNIQUE name
    unique_id = str(uuid.uuid4())[:8]
    temp_vdb = os.path.join(tempfile.gettempdir(), f"ct_volume_{unique_id}.vdb")
    
    log(f"Creating VDB file: {temp_vdb}")
    
    try:
        import openvdb as vdb
        
        # Keep as float32 with real Hounsfield units
        vol_float = vol.astype(np.float32)
        
        log(f"Original numpy array shape (Z,Y,X): {vol_float.shape}")
        log(f"This means: {depth} slices of {height}x{width} images")
        
        vol_for_vdb = vol_float
        
        log(f"VDB input shape: {vol_for_vdb.shape}")
        
        # Create grid from array
        grid = vdb.FloatGrid()
        grid.copyFromArray(vol_for_vdb)
        grid.name = "density"
        
        # Convert spacing to meters
        spacing_meters = [s * 0.001 for s in spacing]  # [X, Y, Z] in meters
        log(f"Spacing in meters (X, Y, Z): {spacing_meters}")
        
        # Fix the OpenVDB transform matrix
        # Our array is [Z, Y, X] order (slices, rows, columns)
        # So the transform must match: [Z-spacing, Y-spacing, X-spacing]
        transform_matrix = [
            [spacing_meters[2], 0, 0, 0],  # First array axis = Z (slice thickness)
            [0, spacing_meters[1], 0, 0],  # Second array axis = Y (row spacing)
            [0, 0, spacing_meters[0], 0],  # Third array axis = X (column spacing)
            [0, 0, 0, 1]
        ]
        grid.transform = vdb.createLinearTransform(transform_matrix)
        
        log(f"Transform matrix diagonal (Z, Y, X) in meters: {[spacing_meters[2], spacing_meters[1], spacing_meters[0]]}")
        
        # Write VDB file
        vdb.write(temp_vdb, grids=[grid])
        log(f"Wrote VDB file: {temp_vdb}")
        
    except Exception as e:
        log(f"OpenVDB error: {e}")
        raise Exception(f"Failed to create OpenVDB file: {e}")

    # Load VDB file into Blender
    bpy.ops.object.volume_import(filepath=temp_vdb, files=[{"name": os.path.basename(temp_vdb)}])
    
    # Get the imported volume object
    vol_obj = bpy.context.active_object
    vol_obj.name = "CT_Volume"
    
    # Rotate volume to correct orientation
    # Patient Z-axis (head-to-feet) should align with Blender Z-axis (up-down)
    # Rotate 270° around Y-axis for proper anatomical orientation
    vol_obj.rotation_euler = (0, math.radians(270), 0)
    
    vol_obj.scale = (1.0, 1.0, 1.0)
    log(f"Imported volume scale: {vol_obj.scale}")
    log(f"Imported volume rotation: {vol_obj.rotation_euler}")
    log(f"Imported volume dimensions: {vol_obj.dimensions}")
    
    expected_dims = (width * spacing_meters[0], height * spacing_meters[1], depth * spacing_meters[2])
    log(f"Expected dimensions (meters): {expected_dims}")
    
    # Create materials
    create_volume_material(vol_obj, vol_min, vol_max)
    
    log("Creating mesh material...")
    create_mesh_material(vol_obj, vol_min, vol_max)
    
    log("Creating geometry nodes setup...")
    create_volume_to_mesh_geonodes(vol_obj)
    
    log(f"Volume created with Hounsfield units preserved ({vol_min:.1f} to {vol_max:.1f})")
    log("=" * 60)
    
    return vol_obj
