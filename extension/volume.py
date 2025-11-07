"""Volume creation and OpenVDB handling"""

import bpy
import os
import tempfile
import numpy as np
from .dicom_io import log

def create_volume(slices):
    """Create a volume object from DICOM slices with proper Hounsfield units"""
    # Sort by slice location or instance number
    slices = sorted(slices, key=lambda s: (s["slice_location"], s["instance_number"]))
    
    # Check if all slices have same dimensions
    first_shape = slices[0]["pixels"].shape
    slices = [s for s in slices if s["pixels"].shape == first_shape]
    
    if len(slices) < 2:
        raise ValueError("Need at least 2 slices with matching dimensions")
    
    vol = np.stack([s["pixels"] for s in slices])
    depth, height, width = vol.shape
    spacing = slices[0]["pixel_spacing"] + [slices[0]["slice_thickness"]]
    
    # Get value range for info
    vol_min, vol_max = vol.min(), vol.max()
    log(f"Creating volume: {width}x{height}x{depth}, spacing: {spacing}")
    log(f"Value range: {vol_min:.1f} to {vol_max:.1f} (Hounsfield units for CT)")
    
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

    # Save volume data to temporary VDB file
    temp_vdb = os.path.join(tempfile.gettempdir(), "ct_volume_temp.vdb")
    
    try:
        import openvdb as vdb
        
        # Keep as float32 with real Hounsfield units
        vol_float = vol.astype(np.float32)
        
        # OpenVDB expects data in (x, y, z) order but numpy is (z, y, x)
        vol_transposed = np.transpose(vol_float, (2, 1, 0))
        
        # Create grid from array
        grid = vdb.FloatGrid()
        grid.copyFromArray(vol_transposed)
        grid.name = "density"
        
        # Set voxel size (transform) using matrix
        transform_matrix = [
            [spacing[1], 0, 0, 0],
            [0, spacing[0], 0, 0],
            [0, 0, spacing[2], 0],
            [0, 0, 0, 1]
        ]
        grid.transform = vdb.createLinearTransform(transform_matrix)
        
        # Write VDB file
        vdb.write(temp_vdb, grids=[grid])
        log(f"Wrote VDB file: {temp_vdb}")
        
    except Exception as e:
        log(f"OpenVDB error: {e}")
        raise Exception(f"Failed to create OpenVDB file: {e}")

    # Load VDB file into Blender
    bpy.ops.object.volume_import(filepath=temp_vdb, files=[{"name": "ct_volume_temp.vdb"}])
    
    # Get the imported volume object
    vol_obj = bpy.context.active_object
    vol_obj.name = "CT_Volume"
    vol_obj.scale = (1.0, 1.0, 1.0)

    # Create material with proper CT visualization
    create_volume_material(vol_obj, vol_min, vol_max)
    
    log(f"Volume created with Hounsfield units preserved ({vol_min:.1f} to {vol_max:.1f})")
    log("Tip: Adjust the 'Multiply' node value (currently 0.05) to control overall opacity")
    
    return vol_obj

def create_volume_material(vol_obj, vol_min, vol_max):
    """Create material with proper CT visualization"""
    mat = bpy.data.materials.new("CT_Volume_Material")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    
    # Output
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    
    # Volume Principled
    prin = nodes.new("ShaderNodeVolumePrincipled")
    prin.location = (600, 0)
    prin.inputs["Anisotropy"].default_value = 0.0
    
    # Volume Info node to read density
    vol_info = nodes.new("ShaderNodeVolumeInfo")
    vol_info.location = (-400, 0)
    
    # Map Hounsfield units to density (absorption)
    ramp_density = nodes.new("ShaderNodeValToRGB")
    ramp_density.location = (200, 100)
    ramp_density.color_ramp.interpolation = 'LINEAR'
    
    # Normalize HU to 0-1 range for color ramp input
    map_range = nodes.new("ShaderNodeMapRange")
    map_range.location = (0, 100)
    map_range.inputs["From Min"].default_value = vol_min
    map_range.inputs["From Max"].default_value = vol_max
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0
    map_range.clamp = True
    
    # Configure density ramp for CT
    ramp_density.color_ramp.elements[0].position = 0.0
    ramp_density.color_ramp.elements[0].color = (0, 0, 0, 1)
    
    # Add middle stop for soft tissue threshold
    if len(ramp_density.color_ramp.elements) < 3:
        ramp_density.color_ramp.elements.new(0.5)
    
    # Soft tissue threshold (around 0 HU)
    soft_tissue_pos = (0 - vol_min) / (vol_max - vol_min) if vol_max > vol_min else 0.5
    ramp_density.color_ramp.elements[1].position = max(0.1, min(0.9, soft_tissue_pos))
    ramp_density.color_ramp.elements[1].color = (0.3, 0.3, 0.3, 1)
    
    # Bone (high HU): full density
    ramp_density.color_ramp.elements[2].position = 1.0
    ramp_density.color_ramp.elements[2].color = (1, 1, 1, 1)
    
    # Color ramp for visual appearance
    ramp_color = nodes.new("ShaderNodeValToRGB")
    ramp_color.location = (200, -100)
    ramp_color.color_ramp.interpolation = 'LINEAR'
    
    # Air/background: black
    ramp_color.color_ramp.elements[0].position = 0.0
    ramp_color.color_ramp.elements[0].color = (0.02, 0.02, 0.05, 1)
    
    # Add soft tissue color
    if len(ramp_color.color_ramp.elements) < 4:
        ramp_color.color_ramp.elements.new(0.4)
        ramp_color.color_ramp.elements.new(0.7)
    
    # Soft tissue: pinkish
    ramp_color.color_ramp.elements[1].position = max(0.2, min(0.5, soft_tissue_pos))
    ramp_color.color_ramp.elements[1].color = (0.8, 0.6, 0.5, 1)
    
    # Dense tissue: lighter
    ramp_color.color_ramp.elements[2].position = max(0.5, min(0.8, soft_tissue_pos + 0.2))
    ramp_color.color_ramp.elements[2].color = (0.9, 0.85, 0.8, 1)
    
    # Bone: white
    ramp_color.color_ramp.elements[3].position = 1.0
    ramp_color.color_ramp.elements[3].color = (1, 1, 1, 1)
    
    # Math multiply to scale density for visibility
    math_scale = nodes.new("ShaderNodeMath")
    math_scale.location = (400, 100)
    math_scale.operation = 'MULTIPLY'
    math_scale.inputs[1].default_value = 0.05
    
    # Connections
    links.new(vol_info.outputs["Density"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], ramp_density.inputs["Fac"])
    links.new(map_range.outputs["Result"], ramp_color.inputs["Fac"])
    links.new(ramp_density.outputs["Color"], math_scale.inputs[0])
    links.new(math_scale.outputs["Value"], prin.inputs["Density"])
    links.new(ramp_color.outputs["Color"], prin.inputs["Color"])
    links.new(prin.outputs["Volume"], out.inputs["Volume"])
    
    # Assign material
    if len(vol_obj.data.materials) == 0:
        vol_obj.data.materials.append(mat)
    else:
        vol_obj.data.materials[0] = mat
    
    # Set viewport display settings for better visualization
    vol_obj.data.display.density = 0.01