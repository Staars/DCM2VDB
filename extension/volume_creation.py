"""Volume creation and OpenVDB handling"""

import bpy
import os
import tempfile
import numpy as np
import uuid
import math
from .utils import SimpleLogger

from .constants import *
from .volume_utils import clean_old_volumes, save_debug_slice
from .materials import create_volume_material
from .geometry_nodes import create_tissue_mesh_geonodes
from .dicom_io import load_slice

# Get logger for this extension
log = SimpleLogger()

def _create_4d_volume_sequence(time_points_data, series_number):
    """Create a 4D volume sequence with numbered VDB files"""
    import openvdb as vdb
    
    log.info(f"Creating 4D volume sequence with {len(time_points_data)} time points...")
    
    temp_dir = tempfile.gettempdir()
    vdb_paths = []
    first_slices = None  # Store first time point slices for positioning
    first_vol_raw = None  # Store first time point raw volume for measurements
    first_spacing = None  # Store spacing for measurements
    
    # Create VDB file for each time point
    for frame_idx, tp_data in enumerate(time_points_data, start=1):
        log.info(f"  Processing time point {frame_idx}/{len(time_points_data)}...")
        
        # 1. Load slices for this time point (SAME as regular volume)
        slices = []
        for file_path in tp_data['files']:
            slice_data = load_slice(file_path)
            if slice_data:
                slices.append(slice_data)
        
        if len(slices) < MIN_SLICES_REQUIRED:
            log.warning(f"  Skipping time point {frame_idx}: not enough slices")
            continue
        
        # 2. Parse ImageOrientationPatient and ImagePositionPatient (SAME as regular volume)
        for slice_data in slices:
            orientation = slice_data["ds"].ImageOrientationPatient
            position = slice_data["ds"].ImagePositionPatient
            row_cosines = np.array(orientation[:3], dtype=np.float32)
            col_cosines = np.array(orientation[3:], dtype=np.float32)
            normal_vector = np.cross(row_cosines, col_cosines)
            slice_data["normal_vector"] = normal_vector
            slice_data["position"] = np.array(position, dtype=np.float32)
        
        # 3. Sort slices (SAME as regular volume)
        slices = sorted(slices, key=lambda s: np.dot(s["position"], s["normal_vector"]))
        
        # Check dimensions match
        first_shape = slices[0]["pixels"].shape
        slices = [s for s in slices if s["pixels"].shape == first_shape]
        
        # 4. Stack into 3D volume (SAME as regular volume)
        vol = np.stack([s["pixels"] for s in slices])
        depth, height, width = vol.shape
        
        # 5. Get spacing (SAME as regular volume)
        pixel_spacing = slices[0]["pixel_spacing"]
        slice_thickness = slices[0]["slice_thickness"]
        
        # Validate slice thickness
        if not slice_thickness or slice_thickness <= 0:
            slice_thickness = max(pixel_spacing)
        
        spacing = [pixel_spacing[1], pixel_spacing[0], slice_thickness]  # [X, Y, Z] in mm
        
        # Store first time point data for measurements
        if frame_idx == 1:
            first_vol_raw = vol.copy()  # Keep raw HU values
            first_spacing = spacing
            first_slices = slices
        
        # 6. Normalize to 0-1 range (SAME as regular volume)
        vol_float = vol.astype(np.float32)
        vol_normalized = (vol_float - HU_MIN_FIXED) / (HU_MAX_FIXED - HU_MIN_FIXED)
        vol_normalized = np.clip(vol_normalized, 0.0, 1.0)
        
        log.debug(f"  Frame {frame_idx}: {width}x{height}x{depth}, spacing: {spacing} mm")
        
        # 7. Create VDB grid with proper transform (SAME as regular volume)
        grid = vdb.FloatGrid()
        grid.copyFromArray(vol_normalized)
        grid.name = "density"
        
        # Convert spacing to meters
        spacing_meters = [s * MM_TO_METERS for s in spacing]  # [X, Y, Z] in meters
        
        # Transform matrix (SAME as regular volume)
        transform_matrix = [
            [spacing_meters[2], 0, 0, 0],  # First array axis = Z (slice thickness)
            [0, spacing_meters[1], 0, 0],  # Second array axis = Y (row spacing)
            [0, 0, spacing_meters[0], 0],  # Third array axis = X (column spacing)
            [0, 0, 0, 1]
        ]
        grid.transform = vdb.createLinearTransform(transform_matrix)
        
        # 8. Save numbered VDB file
        vdb_path = os.path.join(temp_dir, f"ct_s{series_number}_{frame_idx:03d}.vdb")
        vdb.write(vdb_path, grids=[grid])
        vdb_paths.append(vdb_path)
        
        log.info(f"  Created {os.path.basename(vdb_path)}")
        
        # Store first time point slices for positioning
        if frame_idx == 1:
            first_slices = slices
    
    if not vdb_paths:
        raise ValueError("No valid time points created")
    
    # 9. Load first VDB into Blender as sequence
    bpy.ops.object.volume_import(filepath=vdb_paths[0], files=[{"name": os.path.basename(vdb_paths[0])}])
    
    # Get the imported volume object
    vol_obj = bpy.context.active_object
    modality = first_slices[0]["ds"].Modality if hasattr(first_slices[0]["ds"], "Modality") else "CT"
    vol_obj.name = f"{modality}_Volume_S{series_number}_4D"
    
    # 10. Configure as sequence
    vol_obj.data.is_sequence = True
    vol_obj.data.frame_duration = len(vdb_paths)
    vol_obj.data.frame_start = 1
    vol_obj.data.sequence_mode = 'REPEAT'
    
    # 11. Apply positioning (SAME as regular volume)
    first_position = first_slices[0]["position"]  # Already in mm
    
    blender_location = (
        -first_position[0] * MM_TO_METERS,  # X: negate for 270° rotation
        -first_position[1] * MM_TO_METERS,  # Y: anterior → back
        first_position[2] * MM_TO_METERS    # Z: superior
    )
    
    vol_obj.location = blender_location
    
    # 12. Apply rotation (SAME as regular volume)
    vol_obj.rotation_euler = (0, math.radians(270), 0)
    vol_obj.scale = (1.0, 1.0, 1.0)
    
    log.debug(f"4D Volume positioned at: {vol_obj.location}")
    log.debug(f"4D Volume rotation: {vol_obj.rotation_euler}")
    
    # 13. Apply material with user-selected preset
    modality = first_slices[0]["ds"].Modality if hasattr(first_slices[0]["ds"], "Modality") else "CT"
    series_desc = first_slices[0]["ds"].SeriesDescription if hasattr(first_slices[0]["ds"], "SeriesDescription") else ""
    vol_min = first_vol_raw.min()
    vol_max = first_vol_raw.max()
    
    # Use the user-selected preset from the dropdown
    preset_name = bpy.context.scene.dicom_material_preset
    create_volume_material(vol_obj, vol_min, vol_max, preset_name=preset_name, modality=modality, series_description=series_desc)
    
    # Update tissue alphas to match the detected preset (SAME as regular volume)
    from .material_presets import get_preset_for_modality
    from .properties import initialize_tissue_alphas
    
    detected_preset = get_preset_for_modality(modality, series_desc)
    
    # Only auto-set preset if user hasn't manually selected one
    # Check if current preset matches the modality (if it's a CT preset for CT data, user chose it)
    current_preset = bpy.context.scene.dicom_material_preset
    current_is_ct = current_preset.startswith('ct_')
    current_is_mri = current_preset.startswith('mri_')
    modality_matches = (modality == 'CT' and current_is_ct) or (modality == 'MR' and current_is_mri)
    
    if not modality_matches:
        # Modality doesn't match, so auto-detect
        log.info(f"Auto-detected preset: {detected_preset} (was: {current_preset})")
        bpy.context.scene.dicom_material_preset = detected_preset
        bpy.context.scene.dicom_active_material_preset = detected_preset
    else:
        # User has selected a preset for this modality, keep it
        log.info(f"Using user-selected preset: {current_preset}")
        detected_preset = current_preset
        bpy.context.scene.dicom_active_material_preset = detected_preset
    
    # Initialize tissue alphas for the preset
    if len(bpy.context.scene.dicom_tissue_alphas) == 0:
        initialize_tissue_alphas(bpy.context, detected_preset, silent=True)
    
    # 14. Save first time point data for measurements (SAME as regular volume)
    unique_id = str(uuid.uuid4())[:8]
    numpy_path = os.path.join(temp_dir, f"ct_volume_{unique_id}.npy")
    np.save(numpy_path, first_vol_raw)
    log.info(f"Saved first time point data for measurements: {numpy_path}")
    
    import json
    
    # Store in scene for measurements
    bpy.context.scene.dicom_volume_data_path = numpy_path
    bpy.context.scene.dicom_volume_spacing = json.dumps(first_spacing)  # [X, Y, Z] in mm
    bpy.context.scene.dicom_volume_unique_id = unique_id
    bpy.context.scene.dicom_volume_hu_min = first_vol_raw.min()
    bpy.context.scene.dicom_volume_hu_max = first_vol_raw.max()
    
    log.info(f"4D volume sequence created: {len(vdb_paths)} frames")
    log.info("=" * 60)
    
    return vol_obj

def create_volume(slices, series_number=1, time_points_data=None):
    """Create a volume object from DICOM slices with proper Hounsfield units.
    
    Args:
        slices: List of DICOM slice data (for single volume or first time point)
        series_number: Series number for unique object naming
        time_points_data: List of dicts with 'files' for each time point (for 4D sequences)
    """
    # Handle 4D sequence
    if time_points_data is not None:
        return _create_4d_volume_sequence(time_points_data, series_number)
    
    # DON'T clean temp dir here - it would delete VDB files from other series!
    # Cleanup only happens on explicit reload
    
    # Get modality for naming
    modality = slices[0]["ds"].Modality if hasattr(slices[0]["ds"], "Modality") else "CT"
    modality_prefix = modality.upper()  # CT, MR, etc.

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
    
    # Stack slices into 3D volume first
    vol = np.stack([s["pixels"] for s in slices])
    depth, height, width = vol.shape
    
    # Apply denoising if enabled
    if bpy.context.scene.denoise_enabled:
        log.info("=" * 60)
        log.info(f"DENOISING ENABLED: {bpy.context.scene.denoise_method}, Strength: {bpy.context.scene.denoise_strength}")
        log.info(f"Processing volume for Series {series_number} ({depth}x{height}x{width})...")
        
        method = bpy.context.scene.denoise_method
        strength = bpy.context.scene.denoise_strength
        
        if method == 'GAUSSIAN_3D':
            # 3D Gaussian filter - processes entire volume at once
            from scipy import ndimage
            log.info("Applying 3D Gaussian filter...")
            vol = ndimage.gaussian_filter(vol, sigma=strength)
            log.info("3D Gaussian filter complete!")
        else:
            # 2D slice-by-slice filtering (existing methods)
            from .volume_utils import denoise_slice_scipy
            
            from .constants import DENOISING_PROGRESS_LOG_INTERVAL
            
            log.info(f"Applying 2D slice-by-slice filtering ({method})...")
            for i in range(depth):
                vol[i] = denoise_slice_scipy(vol[i], method=method, strength=strength)
                
                # Log progress every DENOISING_PROGRESS_LOG_INTERVAL%
                if (i + 1) % max(1, depth // DENOISING_PROGRESS_LOG_INTERVAL) == 0:
                    progress = (i + 1) / depth
                    log.info(f"  Series {series_number} denoising: {progress*100:.0f}% ({i+1}/{depth} slices)")
            
            log.info(f"2D slice-by-slice filtering complete!")
        
        log.info("=" * 60)
    
    # PixelSpacing is [row spacing, column spacing] = [Y spacing, X spacing]
    pixel_spacing = slices[0]["pixel_spacing"]
    slice_thickness = slices[0]["slice_thickness"]
    
    # Validate slice thickness
    if not slice_thickness or slice_thickness <= 0:
        log.warning(f"Invalid slice_thickness ({slice_thickness}), using pixel spacing as fallback")
        slice_thickness = max(pixel_spacing)
    
    # spacing order: [X, Y, Z]
    spacing = [pixel_spacing[1], pixel_spacing[0], slice_thickness]
    
    # Log spacing values
    log.debug(f"Pixel spacing (X, Y): {pixel_spacing}")
    log.debug(f"Slice thickness (Z): {slice_thickness}")
    log.debug(f"Spacing (X, Y, Z) in mm: {spacing}")
    
    # Calculate and log physical dimensions
    phys_x = width * spacing[0]
    phys_y = height * spacing[1]
    phys_z = depth * spacing[2]
    log.debug(f"Physical dimensions (mm): {phys_x:.1f} x {phys_y:.1f} x {phys_z:.1f}")
    log.debug(f"Physical dimensions (cm): {phys_x/10:.1f} x {phys_y/10:.1f} x {phys_z/10:.1f}")
    
    # Get value range for info
    vol_min, vol_max = vol.min(), vol.max()
    log.info(f"Creating volume: {width}x{height}x{depth}, spacing: {spacing}")
    log.info(f"RAW value range: {vol_min:.1f} to {vol_max:.1f} HU")
    
    # Clean up invalid values (padding, errors)
    if vol_min < EXTREME_NEGATIVE_THRESHOLD:
        log.warning(f"Found extreme negative values (min: {vol_min:.1f}), likely padding. Clamping to {EXTREME_NEGATIVE_CLAMP}")
        vol = np.clip(vol, EXTREME_NEGATIVE_CLAMP, vol_max)
        vol_min = vol.min()
    
    # Recalculate stats after cleaning
    vol_mean = vol.mean()
    vol_std = vol.std()
    log.info(f"CLEANED value range: {vol_min:.1f} to {vol_max:.1f} HU")
    log.debug(f"Mean: {vol_mean:.1f}, Std: {vol_std:.1f}")
    log.debug(f"Data type: {vol.dtype}, Shape: {vol.shape}")
    
    # Check for data issues
    unique_values = len(np.unique(vol))
    log.debug(f"Unique values in volume: {unique_values}")
    if unique_values < 10:
        log.warning("Very few unique values - data might be corrupted or improperly scaled")
    
    # Sample some values to verify data looks reasonable
    sample_indices = [
        (depth//4, height//2, width//2),
        (depth//2, height//2, width//2),
        (3*depth//4, height//2, width//2)
    ]
    log.debug("Sample voxel values (z=25%, 50%, 75% center):")
    for idx in sample_indices:
        log.debug(f"  {idx}: {vol[idx]:.1f} HU")
    
    # Debug: Save middle slice as PNG
    save_debug_slice(vol)
    
    # Clean up old volumes for this series
    volume_name = f"{modality_prefix}_Volume_S{series_number}"
    clean_old_volumes(volume_name)

    # Generate unique ID for this volume session
    unique_id = str(uuid.uuid4())[:8]
    
    # Save numpy array to temp file for recomputation
    numpy_path = os.path.join(tempfile.gettempdir(), f"ct_volume_{unique_id}.npy")
    np.save(numpy_path, vol)
    log.info(f"Saved volume data to: {numpy_path}")
    
    import json
    
    # Store in scene for recomputation
    bpy.context.scene.dicom_volume_data_path = numpy_path
    bpy.context.scene.dicom_volume_spacing = json.dumps(spacing)  # [X, Y, Z] in mm
    bpy.context.scene.dicom_volume_unique_id = unique_id
    # Store HU range for threshold conversion
    bpy.context.scene.dicom_volume_hu_min = vol_min
    bpy.context.scene.dicom_volume_hu_max = vol_max
    
    # Save volume data to temporary VDB file
    temp_vdb = os.path.join(tempfile.gettempdir(), f"ct_volume_{unique_id}.vdb")
    
    log.info(f"Creating VDB file: {temp_vdb}")
    
    try:
        import openvdb as vdb
        
        # Normalize to 0-1 range using FIXED HU range for consistency across all volumes
        # This ensures same tissue types always map to same normalized values
        vol_float = vol.astype(np.float32)
        vol_normalized = (vol_float - HU_MIN_FIXED) / (HU_MAX_FIXED - HU_MIN_FIXED)
        
        # Clamp to 0-1 range (in case data exceeds standard range)
        vol_normalized = np.clip(vol_normalized, 0.0, 1.0)
        
        log.debug(f"Original numpy array shape (Z,Y,X): {vol_float.shape}")
        log.debug(f"This means: {depth} slices of {height}x{width} images")
        log.debug(f"Normalized range: {vol_normalized.min():.6f} to {vol_normalized.max():.6f}")
        
        vol_for_vdb = vol_normalized
        
        log.debug(f"VDB input shape: {vol_for_vdb.shape}")
        
        # Create grid from array
        grid = vdb.FloatGrid()
        grid.copyFromArray(vol_for_vdb)
        grid.name = "density"
        
        from .constants import MM_TO_METERS
        
        # Convert spacing to meters
        spacing_meters = [s * MM_TO_METERS for s in spacing]  # [X, Y, Z] in meters
        log.debug(f"Spacing in meters (X, Y, Z): {spacing_meters}")
        
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
        
        log.debug(f"Transform matrix diagonal (Z, Y, X) in meters: {[spacing_meters[2], spacing_meters[1], spacing_meters[0]]}")
        
        # Write VDB file
        vdb.write(temp_vdb, grids=[grid])
        log.info(f"Wrote VDB file: {temp_vdb}")
        
    except Exception as e:
        log.error(f"OpenVDB error: {e}")
        raise Exception(f"Failed to create OpenVDB file: {e}")

    # Load VDB file into Blender
    bpy.ops.object.volume_import(filepath=temp_vdb, files=[{"name": os.path.basename(temp_vdb)}])
    
    # Get the imported volume object
    vol_obj = bpy.context.active_object
    vol_obj.name = volume_name
    
    # Get the first slice's ImagePositionPatient (position of first voxel)
    first_position = slices[0]["position"]  # Already in mm
    
    from .constants import MM_TO_METERS
    
    # Convert DICOM patient coordinates to Blender world coordinates
    # DICOM: X=right, Y=anterior, Z=superior (head)
    # Blender: X=right, Y=back, Z=up
    # With 270° Y-axis rotation applied, negate X
    blender_location = (
        -first_position[0] * MM_TO_METERS,  # X: negate for 270° rotation (mm → m)
        -first_position[1] * MM_TO_METERS,  # Y: anterior → back (mm → m, flip)
        first_position[2] * MM_TO_METERS    # Z: superior (mm → m)
    )
    
    # Get FrameOfReferenceUID and ImageOrientationPatient from first slice
    frame_of_ref = slices[0]["ds"].FrameOfReferenceUID if hasattr(slices[0]["ds"], "FrameOfReferenceUID") else "NOT SET"
    orientation = slices[0].get("orientation", [1, 0, 0, 0, 1, 0])
    
    log.debug(f"===== VOLUME POSITIONING DEBUG =====")
    log.debug(f"Series number: {series_number}")
    log.debug(f"Number of slices: {len(slices)}")
    log.debug(f"Volume dimensions (voxels): {vol.shape}")
    log.debug(f"FrameOfReferenceUID: {frame_of_ref}")
    log.debug(f"ImageOrientationPatient: {orientation}")
    log.debug(f"First slice ImagePositionPatient: {first_position} mm")
    log.debug(f"Last slice ImagePositionPatient: {slices[-1]['position']} mm")
    log.debug(f"Pixel spacing: {slices[0]['pixel_spacing']} mm")
    log.debug(f"Slice thickness: {slices[0]['slice_thickness']} mm")
    log.debug(f"Calculated Blender location: {blender_location} m")
    
    # Set position
    vol_obj.location = blender_location
    log.debug(f"Volume object location set to: {vol_obj.location}")
    
    # Rotate volume to correct orientation
    # Patient Z-axis (head-to-feet) should align with Blender Z-axis (up-down)
    # Rotate 270° around Y-axis for proper anatomical orientation
    vol_obj.rotation_euler = (0, math.radians(270), 0)
    
    vol_obj.scale = (1.0, 1.0, 1.0)
    log.debug(f"Imported volume scale: {vol_obj.scale}")
    log.debug(f"Imported volume rotation: {vol_obj.rotation_euler}")
    log.debug(f"Imported volume dimensions: {vol_obj.dimensions}")
    
    expected_dims = (width * spacing_meters[0], height * spacing_meters[1], depth * spacing_meters[2])
    log.debug(f"Expected dimensions (meters): {expected_dims}")
    
    # Create volume material with user-selected preset
    series_desc = slices[0]["ds"].SeriesDescription if hasattr(slices[0]["ds"], "SeriesDescription") else ""
    
    # Use the user-selected preset from the dropdown
    preset_name = bpy.context.scene.dicom_material_preset
    create_volume_material(vol_obj, vol_min, vol_max, preset_name=preset_name, modality=modality, series_description=series_desc)
    
    # Update tissue alphas to match the detected preset
    from .material_presets import get_preset_for_modality
    from .properties import initialize_tissue_alphas
    
    detected_preset = get_preset_for_modality(modality, series_desc)
    
    # Only auto-set preset if user hasn't manually selected one
    # Check if current preset matches the modality (if it's a CT preset for CT data, user chose it)
    current_preset = bpy.context.scene.dicom_material_preset
    current_is_ct = current_preset.startswith('ct_')
    current_is_mri = current_preset.startswith('mri_')
    modality_matches = (modality == 'CT' and current_is_ct) or (modality == 'MR' and current_is_mri)
    
    if not modality_matches:
        # Modality doesn't match, so auto-detect
        log.info(f"Auto-detected preset: {detected_preset} (was: {current_preset})")
        bpy.context.scene.dicom_material_preset = detected_preset
        bpy.context.scene.dicom_active_material_preset = detected_preset
    else:
        # User has selected a preset for this modality, keep it
        log.info(f"Using user-selected preset: {current_preset}")
        detected_preset = current_preset
        bpy.context.scene.dicom_active_material_preset = detected_preset
    
    # Initialize tissue alphas for the preset
    if len(bpy.context.scene.dicom_tissue_alphas) == 0:
        initialize_tissue_alphas(bpy.context, detected_preset, silent=True)
    
    # Create meshes based on preset definitions
    from .material_presets import load_preset
    preset = load_preset(detected_preset)
    
    if preset and preset.meshes:
        log.info("=" * 60)
        log.info(f"Creating {len(preset.meshes)} mesh(es) from preset...")
        log.info("=" * 60)
        
        for mesh_def in preset.meshes:
            from .constants import MIN_ISLAND_VERTICES, MESH_THRESHOLD_MAX
            
            mesh_name = mesh_def.get("name", "mesh")
            mesh_label = mesh_def.get("label", mesh_name.title())
            mesh_threshold = mesh_def.get("threshold", 400)
            separate_islands = mesh_def.get("separate_islands", False)
            min_island_verts = mesh_def.get("min_island_verts", MIN_ISLAND_VERTICES)
            
            # Object and material names
            mesh_obj_name = f"{modality_prefix}_{mesh_label.replace(' ', '_')}_S{series_number}"
            mesh_mat_name = f"{modality_prefix}_{mesh_label.replace(' ', '_')}_Material"
            
            # Clean up old mesh object if exists
            old_mesh = bpy.data.objects.get(mesh_obj_name)
            if old_mesh:
                bpy.data.objects.remove(old_mesh, do_unlink=True)
                log.debug(f"Removed old {mesh_obj_name} object")
            
            # Get or create shared mesh material
            mat_mesh = bpy.data.materials.get(mesh_mat_name)
            
            if mat_mesh:
                log.debug(f"Reusing existing material: {mesh_mat_name}")
            else:
                log.info(f"Creating new shared material: {mesh_mat_name}")
                mat_mesh = bpy.data.materials.new(mesh_mat_name)
                mat_mesh.use_nodes = True
                nodes = mat_mesh.node_tree.nodes
                links = mat_mesh.node_tree.links
                nodes.clear()
                
                # Simple material with pointiness-based coloring
                from .constants import (
                    MESH_COLOR_RAMP_POINTINESS_MIN,
                    MESH_COLOR_RAMP_POINTINESS_MAX,
                    MESH_COLOR_A_RGB,
                    MESH_COLOR_B_RGB,
                    MESH_ROUGHNESS,
                    MESH_SPECULAR_MULTIPLIER
                )
                
                geom = nodes.new('ShaderNodeNewGeometry')
                geom.location = (-565, 330)
                
                color_ramp = nodes.new('ShaderNodeValToRGB')
                color_ramp.location = (-238, 418)
                color_ramp.color_ramp.elements[0].position = MESH_COLOR_RAMP_POINTINESS_MIN
                color_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
                color_ramp.color_ramp.elements[1].position = MESH_COLOR_RAMP_POINTINESS_MAX
                color_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
                
                math_node = nodes.new('ShaderNodeMath')
                math_node.location = (144, 301)
                math_node.operation = 'MULTIPLY'
                math_node.inputs[1].default_value = MESH_SPECULAR_MULTIPLIER
                
                mix = nodes.new('ShaderNodeMix')
                mix.location = (74, 570)
                mix.data_type = 'RGBA'
                mix.inputs['A'].default_value = (*MESH_COLOR_A_RGB, 1.0)
                mix.inputs['B'].default_value = (*MESH_COLOR_B_RGB, 1.0)
                
                bsdf = nodes.new('ShaderNodeBsdfPrincipled')
                bsdf.location = (555, 449)
                bsdf.inputs['Roughness'].default_value = MESH_ROUGHNESS
                
                out = nodes.new('ShaderNodeOutputMaterial')
                out.location = (976, 418)
                
                links.new(geom.outputs['Pointiness'], color_ramp.inputs['Fac'])
                links.new(color_ramp.outputs['Color'], mix.inputs['Factor'])
                links.new(color_ramp.outputs['Color'], math_node.inputs[0])
                links.new(mix.outputs['Result'], bsdf.inputs['Base Color'])
                links.new(math_node.outputs['Value'], bsdf.inputs['Specular IOR Level'])
                links.new(math_node.outputs['Value'], bsdf.inputs['Sheen Weight'])
                links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
            
            # Create mesh object
            log.info(f"Creating {mesh_label} mesh object...")
            mesh_obj = vol_obj.copy()
            mesh_obj.name = mesh_obj_name
            mesh_obj.data = vol_obj.data  # Share VDB data
            bpy.context.collection.objects.link(mesh_obj)
            mesh_obj.location = vol_obj.location.copy()
            mesh_obj.rotation_euler = vol_obj.rotation_euler.copy()
            
            from .constants import MESH_THRESHOLD_MAX
            
            # Create geometry nodes for mesh
            mesh_mod = create_tissue_mesh_geonodes(
                mesh_obj, mesh_label, mesh_threshold, MESH_THRESHOLD_MAX, mat_mesh
            )
            if mesh_mod:
                mesh_mod.show_viewport = True  # Visible by default
                mesh_mod.show_render = True
                log.info(f"{mesh_label} mesh created (visible)")
            else:
                log.error(f"{mesh_label} modifier creation failed!")
    else:
        log.debug("No mesh definitions in preset - skipping mesh creation")
    
    log.info(f"Volume created with Hounsfield units preserved ({vol_min:.1f} to {vol_max:.1f})")
    log.info("=" * 60)
    
    return vol_obj
