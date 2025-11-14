"""Volume creation and OpenVDB handling"""

import bpy
import os
import tempfile
import numpy as np
from .dicom_io import log
import uuid

def clean_temp_dir():
    # Clean up old temporary VDB files
    try:
        import glob
        old_vdbs = glob.glob(os.path.join(tempfile.gettempdir(), "ct_volume_*.vdb"))
        for old_vdb in old_vdbs:
            try:
                os.remove(old_vdb)
                log(f"Cleaned up old VDB: {old_vdb}")
            except:
                pass
    except:
        pass

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
    
    if len(slices) < 2:
        raise ValueError("Need at least 2 slices with matching dimensions")
     
    vol = np.stack([s["pixels"] for s in slices])
    depth, height, width = vol.shape
    
    # PixelSpacing is [row spacing, column spacing] = [Y spacing, X spacing]
    pixel_spacing = slices[0]["pixel_spacing"]
    slice_thickness = slices[0]["slice_thickness"]
    
    # STEP 1: Validate slice thickness
    if not slice_thickness or slice_thickness <= 0:
        log(f"WARNING: Invalid slice_thickness ({slice_thickness}), using pixel spacing as fallback")
        slice_thickness = max(pixel_spacing)
    
    # spacing order: [X, Y, Z]
    spacing = [pixel_spacing[1], pixel_spacing[0], slice_thickness]
    
    # STEP 2: Log spacing values for debugging
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
    vol_mean = vol.mean()
    vol_std = vol.std()
    log(f"Creating volume: {width}x{height}x{depth}, spacing: {spacing}")
    log(f"RAW value range: {vol_min:.1f} to {vol_max:.1f} HU")
    
    # Clean up invalid values (padding, errors)
    if vol_min < -2000:
        log(f"WARNING: Found extreme negative values (min: {vol_min:.1f}), likely padding. Clamping to -1024")
        vol = np.clip(vol, -1024, vol_max)
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
    
    # Debug: Save middle slice as PNG to verify data
    try:
        from PIL import Image
        middle_slice = vol[depth//2, :, :]
        
        # Normalize for viewing
        slice_min, slice_max = middle_slice.min(), middle_slice.max()
        if slice_max > slice_min:
            normalized = ((middle_slice - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(middle_slice, dtype=np.uint8)
        
        import tempfile
        debug_path = os.path.join(tempfile.gettempdir(), "dicom_middle_slice_debug.png")
        img = Image.fromarray(normalized, mode='L')
        img.save(debug_path)
        log(f"DEBUG: Saved middle slice to {debug_path}")
    except Exception as e:
        log(f"Could not save debug slice: {e}")
    
    # Clean up old volumes
    for o in list(bpy.data.objects):
        if o.name.startswith("CT_Volume"): 
            bpy.data.objects.remove(o, do_unlink=True)
    for m in list(bpy.data.materials):
        if m.name.startswith("CT_Volume"): 
            bpy.data.materials.remove(m, do_unlink=True)
    for v in list(bpy.data.volumes):
        if v.name.startswith("CT_Volume"): 
            bpy.data.volumes.remove(v, do_unlink=True)

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
        
        # STEP 3: Convert spacing to meters
        spacing_meters = [s * 0.001 for s in spacing]  # [X, Y, Z] in meters
        log(f"Spacing in meters (X, Y, Z): {spacing_meters}")
        
        # STEP 4: Fix the OpenVDB transform matrix
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
    
    # STEP 5: Rotate volume to correct orientation
    # Patient Z-axis (head-to-feet) should align with Blender Z-axis (up-down)
    # Rotate 90° + 180° = 270° around Y-axis for proper anatomical orientation
    import math
    vol_obj.rotation_euler = (0, math.radians(270), 0)  # Rotate 270° around Y-axis (or -90°)
    
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

def create_volume_material(vol_obj, vol_min, vol_max):
    """Create material with proper CT visualization"""
    mat = bpy.data.materials.new("CT_Volume_Material")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    
    # Output
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (1200, 0)
    
    # Volume Principled
    prin = nodes.new("ShaderNodeVolumePrincipled")
    prin.location = (900, 0)
    prin.inputs["Anisotropy"].default_value = 0.0
    
    # Volume Info node to read density
    vol_info = nodes.new("ShaderNodeVolumeInfo")
    vol_info.location = (-600, 0)
    
    # Map Range: Convert Hounsfield units to 0-1 range
    map_range = nodes.new("ShaderNodeMapRange")
    map_range.location = (-400, 0)
    map_range.inputs["From Min"].default_value = vol_min
    map_range.inputs["From Max"].default_value = vol_max
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0
    map_range.clamp = True
    
    # THRESHOLD: Make air completely transparent
    # Math Greater Than to create hard cutoff
    threshold_hu = -200.0  # Everything below this is transparent
    
    math_threshold = nodes.new("ShaderNodeMath")
    math_threshold.location = (-200, 200)
    math_threshold.operation = 'GREATER_THAN'
    math_threshold.inputs[1].default_value = threshold_hu
    math_threshold.label = "Air_Threshold"
    math_threshold.use_clamp = True
    
    # Color Ramp for density/absorption (only for visible parts)
    ramp_density = nodes.new("ShaderNodeValToRGB")
    ramp_density.location = (0, 100)
    ramp_density.color_ramp.interpolation = 'LINEAR'
    
    # Setup density ramp based on CT values
    ramp_density.color_ramp.elements[0].position = 0.0
    ramp_density.color_ramp.elements[0].color = (0, 0, 0, 1)
    
    # Add stops for soft tissue and bone
    if len(ramp_density.color_ramp.elements) < 4:
        ramp_density.color_ramp.elements.new(0.3)
        ramp_density.color_ramp.elements.new(0.5)
        ramp_density.color_ramp.elements.new(0.7)
    
    # Calculate positions based on actual HU values
    def hu_to_pos(hu):
        return max(0.0, min(1.0, (hu - vol_min) / (vol_max - vol_min)))
    
    # Soft tissue start (around -100 HU)
    ramp_density.color_ramp.elements[1].position = hu_to_pos(-100)
    ramp_density.color_ramp.elements[1].color = (0.1, 0.1, 0.1, 1)
    
    # Soft tissue peak (around 40 HU)
    ramp_density.color_ramp.elements[2].position = hu_to_pos(40)
    ramp_density.color_ramp.elements[2].color = (0.5, 0.5, 0.5, 1)
    
    # Bone (around 400 HU)
    ramp_density.color_ramp.elements[3].position = hu_to_pos(400)
    ramp_density.color_ramp.elements[3].color = (1, 1, 1, 1)
    
    # Color Ramp for visual appearance
    ramp_color = nodes.new("ShaderNodeValToRGB")
    ramp_color.location = (0, -100)
    ramp_color.color_ramp.interpolation = 'LINEAR'
    
    # Add color stops
    if len(ramp_color.color_ramp.elements) < 4:
        ramp_color.color_ramp.elements.new(0.4)
        ramp_color.color_ramp.elements.new(0.7)
    
    # Air: dark
    ramp_color.color_ramp.elements[0].position = 0.0
    ramp_color.color_ramp.elements[0].color = (0.01, 0.01, 0.02, 1)
    
    # Fat/Soft tissue: reddish
    ramp_color.color_ramp.elements[1].position = hu_to_pos(-50)
    ramp_color.color_ramp.elements[1].color = (0.6, 0.4, 0.3, 1)
    
    # Muscle/organs: pinkish
    ramp_color.color_ramp.elements[2].position = hu_to_pos(50)
    ramp_color.color_ramp.elements[2].color = (0.9, 0.7, 0.6, 1)
    
    # Bone: white
    ramp_color.color_ramp.elements[3].position = hu_to_pos(400)
    ramp_color.color_ramp.elements[3].color = (1, 1, 1, 1)
    
    # Multiply threshold mask with density ramp
    math_mask = nodes.new("ShaderNodeMath")
    math_mask.location = (300, 100)
    math_mask.operation = 'MULTIPLY'
    math_mask.label = "Apply_Threshold"
    
    # Math multiply to scale density - make it adjustable
    math_scale = nodes.new("ShaderNodeMath")
    math_scale.location = (500, 100)
    math_scale.operation = 'MULTIPLY'
    math_scale.inputs[1].default_value = 0.01
    math_scale.label = "Density_Scale"
    
    # Connections
    links.new(vol_info.outputs["Density"], map_range.inputs["Value"])
    links.new(vol_info.outputs["Density"], math_threshold.inputs[0])  # Threshold based on raw HU
    links.new(map_range.outputs["Result"], ramp_density.inputs["Fac"])
    links.new(map_range.outputs["Result"], ramp_color.inputs["Fac"])
    
    # Apply threshold mask
    links.new(ramp_density.outputs["Color"], math_mask.inputs[0])
    links.new(math_threshold.outputs["Value"], math_mask.inputs[1])
    
    # Scale and output
    links.new(math_mask.outputs["Value"], math_scale.inputs[0])
    links.new(math_scale.outputs["Value"], prin.inputs["Density"])
    links.new(ramp_color.outputs["Color"], prin.inputs["Color"])
    links.new(prin.outputs["Volume"], out.inputs["Volume"])
    
    # Assign material
    if len(vol_obj.data.materials) == 0:
        vol_obj.data.materials.append(mat)
    else:
        vol_obj.data.materials[0] = mat
    
    # Set viewport display settings for better visualization
    vol_obj.data.display.density = 0.005
    
    log(f"Material configured for HU range {vol_min:.0f} to {vol_max:.0f}")
    log(f"Air threshold: {threshold_hu:.0f} HU (everything below is transparent)")
    log(f"Soft tissue at ~{hu_to_pos(40):.2f}, Bone at ~{hu_to_pos(400):.2f}")

def create_volume_to_mesh_geonodes(vol_obj):
    """Create a Geometry Nodes modifier with Volume to Mesh setup"""
    
    try:
        log("Creating Geometry Nodes modifier...")
        
        # Add geometry nodes modifier
        mod = vol_obj.modifiers.new(name="VolumeToMesh", type='NODES')
        
        # Create new node group
        node_group = bpy.data.node_groups.new(name="CT_VolumeToMesh", type='GeometryNodeTree')
        mod.node_group = node_group
        
        # Create input/output nodes
        nodes = node_group.nodes
        links = node_group.links
        
        group_input = nodes.new('NodeGroupInput')
        group_input.location = (-600, 0)
        
        group_output = nodes.new('NodeGroupOutput')
        group_output.location = (600, 0)
        
        # Add Math node (Add) for threshold adjustment
        math_add = nodes.new('ShaderNodeMath')
        math_add.operation = 'ADD'
        math_add.location = (-300, 0)
        math_add.inputs[1].default_value = 1.0  # Add 1 by default
        
        # Add Volume to Mesh node
        vol_to_mesh = nodes.new('GeometryNodeVolumeToMesh')
        vol_to_mesh.location = (0, 0)
        vol_to_mesh.resolution_mode = 'GRID'
        
        # Add Set Material node
        set_material = nodes.new('GeometryNodeSetMaterial')
        set_material.location = (300, 0)
        
        # Get the CT_Mesh_Material
        mesh_material = bpy.data.materials.get("CT_Mesh_Material")
        if mesh_material:
            set_material.inputs['Material'].default_value = mesh_material
        
        # Create input sockets
        node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name="Threshold", in_out='INPUT', socket_type='NodeSocketFloat')
        node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        
        # Set default threshold value
        mod["Input_2_attribute_name"] = "density"
        mod["Input_2_use_attribute"] = 0
        mod["Input_2"] = -200.0  # Default threshold value (HU)
        
        # Connect nodes
        links.new(group_input.outputs[0], vol_to_mesh.inputs['Volume'])
        links.new(group_input.outputs[1], math_add.inputs[0])
        links.new(math_add.outputs[0], vol_to_mesh.inputs['Threshold'])
        links.new(vol_to_mesh.outputs['Mesh'], set_material.inputs['Geometry'])
        links.new(set_material.outputs['Geometry'], group_output.inputs[0])
        
        log("Created Geometry Nodes: Volume to Mesh with material (Threshold: -200 HU + 1)")
        
    except Exception as e:
        log(f"ERROR creating Geometry Nodes: {e}")
        import traceback
        traceback.print_exc()


def create_mesh_material(vol_obj, vol_min, vol_max):
    """Create a surface material for the mesh with tissue-specific colors"""
    mat = bpy.data.materials.new("CT_Mesh_Material")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    
    # Output
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (800, 0)
    
    # Principled BSDF
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (500, 0)
    
    # Attribute node to read density
    attr = nodes.new("ShaderNodeAttribute")
    attr.location = (-600, 0)
    attr.attribute_name = "density"
    
    # Map Range: Convert HU to 0-1
    map_range = nodes.new("ShaderNodeMapRange")
    map_range.location = (-400, 0)
    map_range.inputs["From Min"].default_value = vol_min
    map_range.inputs["From Max"].default_value = vol_max
    map_range.clamp = True
    
    # Color Ramp for tissue colors
    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.location = (-100, 100)
    color_ramp.color_ramp.interpolation = 'LINEAR'
    
    # Setup color stops for different tissues
    def hu_to_pos(hu):
        return max(0.0, min(1.0, (hu - vol_min) / (vol_max - vol_min)))
    
    # Clear default stops and add tissue-specific colors
    while len(color_ramp.color_ramp.elements) > 2:
        color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[0])
    
    # Fat (-100 HU): yellowish
    color_ramp.color_ramp.elements[0].position = hu_to_pos(-100)
    color_ramp.color_ramp.elements[0].color = (0.9, 0.8, 0.4, 1.0)
    
    # Add more stops
    color_ramp.color_ramp.elements.new(hu_to_pos(40))  # Muscle
    color_ramp.color_ramp.elements.new(hu_to_pos(200))  # Bone start
    color_ramp.color_ramp.elements.new(hu_to_pos(400))  # Dense bone
    
    # Muscle (40 HU): reddish
    color_ramp.color_ramp.elements[1].color = (0.7, 0.3, 0.3, 1.0)
    
    # Bone start (200 HU): light grey-yellow
    color_ramp.color_ramp.elements[2].color = (0.85, 0.82, 0.7, 1.0)
    
    # Dense bone (400+ HU): yellowish-grey
    color_ramp.color_ramp.elements[3].color = (0.95, 0.92, 0.8, 1.0)
    
    # Alpha ramp for transparency based on density
    alpha_ramp = nodes.new("ShaderNodeValToRGB")
    alpha_ramp.location = (-100, -100)
    alpha_ramp.color_ramp.interpolation = 'LINEAR'
    
    # Low density = transparent, high density = opaque
    alpha_ramp.color_ramp.elements[0].position = 0.0
    alpha_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    alpha_ramp.color_ramp.elements[1].position = 0.3
    alpha_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    
    # Separate RGB to get alpha
    sep_rgb = nodes.new("ShaderNodeSeparateColor")
    sep_rgb.location = (200, -100)
    
    # Connect nodes
    links.new(attr.outputs["Fac"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
    links.new(map_range.outputs["Result"], alpha_ramp.inputs["Fac"])
    
    links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(alpha_ramp.outputs["Color"], sep_rgb.inputs["Color"])
    links.new(sep_rgb.outputs["Red"], bsdf.inputs["Alpha"])
    
    # Subsurface scattering for skin
    bsdf.inputs["Subsurface Weight"].default_value = 0.1
    bsdf.inputs["Subsurface Radius"].default_value = (0.01, 0.005, 0.003)
    
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    
    # Set blend mode to alpha blend
    mat.blend_method = 'BLEND'
    
    log(f"Created mesh material with tissue colors (Fat: yellow, Muscle: red, Bone: grey-yellow)")
    
    return mat
