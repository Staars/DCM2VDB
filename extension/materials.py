"""Material creation for volume and mesh visualization"""

import bpy
from .dicom_io import log
from .constants import *
from .node_builders import *

def create_volume_material(vol_obj, vol_min, vol_max):
    """Create material for normalized (0-1) VDB data"""
    # Use series-specific material name based on object name
    mat_name = vol_obj.name.replace("CT_Volume", "CT_Volume_Material")
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    
    log("Creating volume material for NORMALIZED data (0-1 range)")
    
    # Output
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (800, 0)
    
    # Volume Principled
    prin = nodes.new("ShaderNodeVolumePrincipled")
    prin.location = (600, 0)
    prin.inputs["Anisotropy"].default_value = 0.0
    
    # Volume Info node to read density (already 0-1)
    vol_info = nodes.new("ShaderNodeVolumeInfo")
    vol_info.location = (-400, 0)
    
    # Calculate normalized air threshold
    # Air is around -1000 HU, normalize it
    air_threshold_normalized = (HU_AIR_THRESHOLD - vol_min) / (vol_max - vol_min)
    log(f"Air threshold: {HU_AIR_THRESHOLD} HU = {air_threshold_normalized:.6f} normalized")
    
    # Threshold mask: hide air (values below threshold)
    math_threshold = nodes.new("ShaderNodeMath")
    math_threshold.location = (-200, 200)
    math_threshold.operation = 'GREATER_THAN'
    math_threshold.inputs[1].default_value = air_threshold_normalized
    math_threshold.label = "Air_Threshold"
    
    # Color ramp for tissue colors (works directly with 0-1 data)
    ramp_color = nodes.new("ShaderNodeValToRGB")
    ramp_color.location = (0, -100)
    ramp_color.label = "Tissue_Colors"
    
    # Set up color stops for different tissues
    ramp_color.color_ramp.elements[0].position = 0.0
    ramp_color.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)  # Black (air/low)
    ramp_color.color_ramp.elements[1].position = 1.0
    ramp_color.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)  # White (bone/high)
    
    # Add intermediate stops
    elem_soft = ramp_color.color_ramp.elements.new(0.5)
    elem_soft.color = (0.8, 0.4, 0.3, 1.0)  # Reddish for soft tissue
    
    # Density ramp (controls opacity)
    ramp_density = nodes.new("ShaderNodeValToRGB")
    ramp_density.location = (0, 100)
    ramp_density.label = "Density_Ramp"
    
    # Density curve: low values = transparent, high values = opaque
    ramp_density.color_ramp.elements[0].position = 0.0
    ramp_density.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)  # Transparent
    ramp_density.color_ramp.elements[1].position = 1.0
    ramp_density.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)  # Opaque
    
    # Multiply threshold mask with density
    math_mask = nodes.new("ShaderNodeMath")
    math_mask.location = (300, 100)
    math_mask.operation = 'MULTIPLY'
    math_mask.label = "Apply_Threshold"
    
    # Scale density for viewport - MUCH HIGHER for visibility
    math_scale = nodes.new("ShaderNodeMath")
    math_scale.location = (450, 100)
    math_scale.operation = 'MULTIPLY'
    math_scale.inputs[1].default_value = 10.0  # High value for normalized data
    math_scale.label = "Density_Scale"
    
    # Connections
    links.new(vol_info.outputs["Density"], math_threshold.inputs[0])
    links.new(vol_info.outputs["Density"], ramp_color.inputs["Fac"])
    links.new(vol_info.outputs["Density"], ramp_density.inputs["Fac"])
    
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
    
    # Set viewport display settings - MUCH HIGHER for visibility
    vol_obj.data.display.density = 1.0  # High value for normalized data
    
    log(f"Volume material created for normalized data")
    log(f"Original HU range: {vol_min:.0f} to {vol_max:.0f}")
    log(f"VDB contains normalized 0-1 values")

def create_mesh_material(vol_obj, vol_min, vol_max):
    """TODO: Reimplement mesh material for tissue-specific visualization"""
    log("Mesh material creation - placeholder (to be reimplemented)")
    pass
