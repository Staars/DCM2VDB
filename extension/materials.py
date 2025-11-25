"""Material creation for volume and mesh visualization"""

import bpy
from .dicom_io import log
from .constants import *
from .node_builders import *

def create_volume_material(vol_obj, vol_min, vol_max):
    """Create or reuse shared CT volume material for normalized (0-1) VDB data"""
    mat_name = "CT_Volume_Material"
    
    # Check if material already exists, reuse it
    mat = bpy.data.materials.get(mat_name)
    if mat:
        log(f"Reusing existing material: {mat_name}")
        if len(vol_obj.data.materials) == 0:
            vol_obj.data.materials.append(mat)
        else:
            vol_obj.data.materials[0] = mat
        return
    
    # Create new shared material
    log(f"Creating new shared material: {mat_name}")
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
    
    # Math: Threshold mask (air removal)
    math_threshold = nodes.new("ShaderNodeMath")
    math_threshold.location = (-392.4792, 359.1090)
    math_threshold.operation = 'GREATER_THAN'
    math_threshold.inputs[1].default_value = air_threshold_normalized
    math_threshold.label = "Air_Threshold"
    
    # Color Ramp: Tissue colors with multiple stops
    ramp_color = nodes.new("ShaderNodeValToRGB")
    ramp_color.location = (-97.6240, -54.9656)
    ramp_color.label = "Tissue_Colors"
    
    # Configure color stops (5 stops for detailed tissue coloring)
    ramp_color.color_ramp.elements[0].position = 0.105
    ramp_color.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)  # Black (air)
    
    ramp_color.color_ramp.elements[1].position = 0.218
    ramp_color.color_ramp.elements[1].color = (0.776, 0.565, 0.018, 0.405)  # Yellow/fat
    
    # Add more stops
    elem_2 = ramp_color.color_ramp.elements.new(0.249)
    elem_2.color = (0.68, 0.008, 0.0, 0.868)  # Dark red
    
    elem_3 = ramp_color.color_ramp.elements.new(0.305)
    elem_3.color = (0.906, 0.071, 0.029, 0.666)  # Bright red/soft tissue
    
    elem_4 = ramp_color.color_ramp.elements.new(0.411)
    elem_4.color = (1.0, 1.0, 1.0, 1.0)  # White (bone)
    
    # Color Ramp.001: Density ramp
    ramp_density = nodes.new("ShaderNodeValToRGB")
    ramp_density.location = (-193.3591, 220.9367)
    ramp_density.label = "Density_Ramp"
    
    # Simple 0-1 gradient
    ramp_density.color_ramp.elements[0].position = 0.0
    ramp_density.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp_density.color_ramp.elements[1].position = 1.0
    ramp_density.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    # Math.001: Multiply threshold with density color
    math_001 = nodes.new("ShaderNodeMath")
    math_001.location = (159.4760, 394.2954)
    math_001.operation = 'MULTIPLY'
    math_001.label = "Apply_Threshold"
    
    # Math.002: Scale alpha by 300
    math_002 = nodes.new("ShaderNodeMath")
    math_002.location = (131.4298, 120.4075)
    math_002.operation = 'MULTIPLY'
    math_002.inputs[1].default_value = 300.0
    math_002.label = "Alpha_Scale"
    
    # Math.003: Final multiply
    math_003 = nodes.new("ShaderNodeMath")
    math_003.location = (349.4558, 181.9858)
    math_003.operation = 'MULTIPLY'
    math_003.label = "Final_Density"
    
    # Connections
    links.new(vol_info.outputs["Density"], math_threshold.inputs[0])
    links.new(vol_info.outputs["Density"], ramp_color.inputs["Fac"])
    links.new(vol_info.outputs["Density"], ramp_density.inputs["Fac"])
    
    # Density path
    links.new(ramp_density.outputs["Color"], math_001.inputs[0])
    links.new(math_threshold.outputs["Value"], math_001.inputs[1])
    links.new(ramp_density.outputs["Alpha"], math_002.inputs[0])
    
    # Combine
    links.new(math_001.outputs["Value"], math_003.inputs[0])
    links.new(math_002.outputs["Value"], math_003.inputs[1])
    
    # Output
    links.new(math_003.outputs["Value"], prin.inputs["Density"])
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
