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
from .geometry_nodes import create_tissue_mesh_geonodes


def create_masked_vdb(vol_array, mask, tissue_name, spacing_meters, unique_id):
    """Create a VDB file with masked volume data"""
    import openvdb as vdb
    
    # Apply mask: set non-tissue voxels to very low value (will be below any threshold)
    masked_vol = np.where(mask, vol_array, -10000.0).astype(np.float32)
    
    # Create VDB grid
    grid = vdb.FloatGrid()
    grid.copyFromArray(masked_vol)
    grid.name = "density"
    
    # Set transform (same as original volume)
    transform_matrix = [
        [spacing_meters[2], 0, 0, 0],  # Z
        [0, spacing_meters[1], 0, 0],  # Y
        [0, 0, spacing_meters[0], 0],  # X
        [0, 0, 0, 1]
    ]
    grid.transform = vdb.createLinearTransform(transform_matrix)
    
    # Save to temp file
    temp_vdb = os.path.join(tempfile.gettempdir(), f"ct_{tissue_name}_{unique_id}.vdb")
    vdb.write(temp_vdb, grids=[grid])
    
    log(f"Created masked VDB for {tissue_name}: {temp_vdb}")
    return temp_vdb


def create_tissue_volumes(vol_array, spacing_meters, unique_id, fat_min, fat_max, fluid_min, fluid_max, soft_min, soft_max, bone_min):
    """Create 4 masked VDB volumes for different tissue types"""
    log("Creating tissue-specific VDB volumes...")
    
    # Create masks for each tissue type
    fat_mask = (vol_array >= fat_min) & (vol_array <= fat_max)
    fluid_mask = (vol_array >= fluid_min) & (vol_array <= fluid_max)
    soft_mask = (vol_array >= soft_min) & (vol_array <= soft_max)
    bone_mask = (vol_array >= bone_min)
    
    # Create VDB files
    fat_vdb = create_masked_vdb(vol_array, fat_mask, "fat", spacing_meters, unique_id)
    fluid_vdb = create_masked_vdb(vol_array, fluid_mask, "fluid", spacing_meters, unique_id)
    soft_vdb = create_masked_vdb(vol_array, soft_mask, "soft", spacing_meters, unique_id)
    bone_vdb = create_masked_vdb(vol_array, bone_mask, "bone", spacing_meters, unique_id)
    
    return {
        'fat': fat_vdb,
        'fluid': fluid_vdb,
        'soft': soft_vdb,
        'bone': bone_vdb
    }


def create_volume(slices, series_number=1):
    """Create a volume object from DICOM slices with proper Hounsfield units.
    
    Args:
        slices: List of DICOM slice data
        series_number: Series number for unique object naming
    """
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
    
    # Clean up old volumes for this series
    volume_name = f"{modality_prefix}_Volume_S{series_number}"
    clean_old_volumes(volume_name)

    # Generate unique ID for this volume session
    unique_id = str(uuid.uuid4())[:8]
    
    # Save numpy array to temp file for recomputation
    numpy_path = os.path.join(tempfile.gettempdir(), f"ct_volume_{unique_id}.npy")
    np.save(numpy_path, vol)
    log(f"Saved volume data to: {numpy_path}")
    
    # Store in scene for recomputation
    bpy.context.scene.dicom_volume_data_path = numpy_path
    bpy.context.scene.dicom_volume_spacing = str(spacing)  # [X, Y, Z] in mm
    bpy.context.scene.dicom_volume_unique_id = unique_id
    # Store HU range for threshold conversion
    bpy.context.scene.dicom_volume_hu_min = vol_min
    bpy.context.scene.dicom_volume_hu_max = vol_max
    
    # Save volume data to temporary VDB file
    temp_vdb = os.path.join(tempfile.gettempdir(), f"ct_volume_{unique_id}.vdb")
    
    log(f"Creating VDB file: {temp_vdb}")
    
    try:
        import openvdb as vdb
        
        # Normalize to 0-1 range using FIXED HU range for consistency across all volumes
        # This ensures same tissue types always map to same normalized values
        vol_float = vol.astype(np.float32)
        vol_normalized = (vol_float - HU_MIN_FIXED) / (HU_MAX_FIXED - HU_MIN_FIXED)
        
        # Clamp to 0-1 range (in case data exceeds standard range)
        vol_normalized = np.clip(vol_normalized, 0.0, 1.0)
        
        log(f"Original numpy array shape (Z,Y,X): {vol_float.shape}")
        log(f"This means: {depth} slices of {height}x{width} images")
        log(f"Normalized range: {vol_normalized.min():.6f} to {vol_normalized.max():.6f}")
        
        vol_for_vdb = vol_normalized
        
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
    vol_obj.name = volume_name
    
    # Get the first slice's ImagePositionPatient (position of first voxel)
    first_position = slices[0]["position"]  # Already in mm
    
    # Convert DICOM patient coordinates to Blender world coordinates
    # DICOM: X=right, Y=anterior, Z=superior (head)
    # Blender: X=right, Y=back, Z=up
    # We need to transform: DICOM (x,y,z) → Blender (x, -y, z) and scale to meters
    blender_location = (
        first_position[0] * 0.001,   # X: right (mm → m)
        -first_position[1] * 0.001,  # Y: anterior → back (mm → m, flip)
        first_position[2] * 0.001    # Z: superior (mm → m)
    )
    
    log(f"DICOM ImagePositionPatient: {first_position} mm")
    log(f"Blender world location: {blender_location} m")
    
    # Set position
    vol_obj.location = blender_location
    
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
    
    # Create volume material with modality detection
    series_desc = slices[0]["ds"].SeriesDescription if hasattr(slices[0]["ds"], "SeriesDescription") else ""
    create_volume_material(vol_obj, vol_min, vol_max, modality=modality, series_description=series_desc)
    
    # Update tissue alphas to match the detected preset
    from .material_presets import get_preset_for_modality
    from .properties import initialize_tissue_alphas
    
    detected_preset = get_preset_for_modality(modality, series_desc)
    current_preset = bpy.context.scene.dicom_active_material_preset
    
    if detected_preset != current_preset or len(bpy.context.scene.dicom_tissue_alphas) == 0:
        log(f"Updating tissue alphas from '{current_preset}' to '{detected_preset}'")
        initialize_tissue_alphas(bpy.context, detected_preset, silent=True)
    else:
        log(f"Tissue alphas already match preset '{detected_preset}'")
    
    # Create meshes based on preset definitions
    from .material_presets import load_preset, get_preset_for_modality
    preset_name = get_preset_for_modality(modality, series_desc)
    preset = load_preset(preset_name)
    
    if preset and preset.meshes:
        log("=" * 60)
        log(f"Creating {len(preset.meshes)} mesh(es) from preset...")
        log("=" * 60)
        
        for mesh_def in preset.meshes:
            mesh_name = mesh_def.get("name", "mesh")
            mesh_label = mesh_def.get("label", mesh_name.title())
            mesh_threshold = mesh_def.get("threshold", 400)
            separate_islands = mesh_def.get("separate_islands", False)
            min_island_verts = mesh_def.get("min_island_verts", 100)
            
            # Object and material names
            mesh_obj_name = f"{modality_prefix}_{mesh_label.replace(' ', '_')}_S{series_number}"
            mesh_mat_name = f"{modality_prefix}_{mesh_label.replace(' ', '_')}_Material"
            
            # Clean up old mesh object if exists
            old_mesh = bpy.data.objects.get(mesh_obj_name)
            if old_mesh:
                bpy.data.objects.remove(old_mesh, do_unlink=True)
                log(f"Removed old {mesh_obj_name} object")
            
            # Get or create shared mesh material
            mat_mesh = bpy.data.materials.get(mesh_mat_name)
            
            if mat_mesh:
                log(f"Reusing existing material: {mesh_mat_name}")
            else:
                log(f"Creating new shared material: {mesh_mat_name}")
                mat_mesh = bpy.data.materials.new(mesh_mat_name)
                mat_mesh.use_nodes = True
                nodes = mat_mesh.node_tree.nodes
                links = mat_mesh.node_tree.links
                nodes.clear()
                
                # Simple material with pointiness-based coloring
                geom = nodes.new('ShaderNodeNewGeometry')
                geom.location = (-565, 330)
                
                color_ramp = nodes.new('ShaderNodeValToRGB')
                color_ramp.location = (-238, 418)
                color_ramp.color_ramp.elements[0].position = 0.414
                color_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
                color_ramp.color_ramp.elements[1].position = 0.527
                color_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
                
                math_node = nodes.new('ShaderNodeMath')
                math_node.location = (144, 301)
                math_node.operation = 'MULTIPLY'
                math_node.inputs[1].default_value = 0.5
                
                mix = nodes.new('ShaderNodeMix')
                mix.location = (74, 570)
                mix.data_type = 'RGBA'
                mix.inputs['A'].default_value = (0.7, 0.301, 0.117, 1.0)
                mix.inputs['B'].default_value = (0.95, 0.92, 0.85, 1.0)
                
                bsdf = nodes.new('ShaderNodeBsdfPrincipled')
                bsdf.location = (555, 449)
                bsdf.inputs['Roughness'].default_value = 0.4
                
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
            log(f"Creating {mesh_label} mesh object...")
            mesh_obj = vol_obj.copy()
            mesh_obj.name = mesh_obj_name
            mesh_obj.data = vol_obj.data  # Share VDB data
            bpy.context.collection.objects.link(mesh_obj)
            mesh_obj.location = vol_obj.location.copy()
            mesh_obj.rotation_euler = vol_obj.rotation_euler.copy()
            
            # Create geometry nodes for mesh
            mesh_mod = create_tissue_mesh_geonodes(
                mesh_obj, mesh_label, mesh_threshold, 10000, mat_mesh
            )
            if mesh_mod:
                mesh_mod.show_viewport = True  # Visible by default
                mesh_mod.show_render = True
                log(f"{mesh_label} mesh created (visible)")
            else:
                log(f"ERROR: {mesh_label} modifier creation failed!")
    else:
        log("No mesh definitions in preset - skipping mesh creation")
    
    log(f"Volume created with Hounsfield units preserved ({vol_min:.1f} to {vol_max:.1f})")
    log("=" * 60)
    
    return vol_obj
