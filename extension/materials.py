"""Material creation for volume and mesh visualization"""

import bpy
from .dicom_io import log
from .constants import *
from .node_builders import *

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
    map_range = build_hu_mapper(nodes, links, vol_min, vol_max, location=(-400, 0))
    
    # Threshold mask
    math_threshold = build_threshold_mask(nodes, HU_AIR_THRESHOLD, location=(-200, 200))
    
    # Density ramp
    ramp_density = build_density_ramp(nodes, vol_min, vol_max, location=(0, 100))
    
    # Color ramp
    ramp_color = build_tissue_color_ramp(nodes, vol_min, vol_max, location=(0, -100))
    
    # Multiply threshold mask with density ramp
    math_mask = nodes.new("ShaderNodeMath")
    math_mask.location = (300, 100)
    math_mask.operation = 'MULTIPLY'
    math_mask.label = "Apply_Threshold"
    
    # Math multiply to scale density
    math_scale = nodes.new("ShaderNodeMath")
    math_scale.location = (500, 100)
    math_scale.operation = 'MULTIPLY'
    math_scale.inputs[1].default_value = DENSITY_SCALE_DEFAULT
    math_scale.label = "Density_Scale"
    
    # Connections
    links.new(vol_info.outputs["Density"], map_range.inputs["Value"])
    links.new(vol_info.outputs["Density"], math_threshold.inputs[0])
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
    
    # Set viewport display settings
    vol_obj.data.display.density = VIEWPORT_DENSITY_DEFAULT
    
    log(f"Material configured for HU range {vol_min:.0f} to {vol_max:.0f}")
    log(f"Air threshold: {HU_AIR_THRESHOLD:.0f} HU (everything below is transparent)")

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
    map_range = build_hu_mapper(nodes, links, vol_min, vol_max, location=(-400, 0))
    
    # Color ramp for tissue colors
    color_ramp = build_mesh_color_ramp(nodes, vol_min, vol_max, location=(-100, 100))
    
    # Alpha ramp for transparency
    alpha_ramp = build_alpha_ramp(nodes, location=(-100, -100))
    
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
    bsdf.inputs["Subsurface Weight"].default_value = SUBSURFACE_WEIGHT
    bsdf.inputs["Subsurface Radius"].default_value = SUBSURFACE_RADIUS
    
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    
    # Set blend mode to alpha blend
    mat.blend_method = 'BLEND'
    
    log(f"Created mesh material with tissue colors")
    
    return mat
