"""Material creation for volume and mesh visualization"""

import bpy
from .dicom_io import log
from .constants import *
from .node_builders import *
from .volume_utils import hu_to_normalized

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
    
    # Calculate normalized air threshold using FIXED range
    # This ensures consistent threshold across all volumes
    air_threshold_normalized = (HU_AIR_THRESHOLD - HU_MIN_FIXED) / (HU_MAX_FIXED - HU_MIN_FIXED)
    log(f"Using FIXED HU range: {HU_MIN_FIXED} to {HU_MAX_FIXED}")
    log(f"Actual volume HU range: {vol_min:.1f} to {vol_max:.1f}")
    log(f"Air threshold: {HU_AIR_THRESHOLD} HU = {air_threshold_normalized:.6f} normalized (fixed range)")
    
    # Math: Threshold mask (air removal)
    math_threshold = nodes.new("ShaderNodeMath")
    math_threshold.location = (-87.0482, 246.0720)
    math_threshold.operation = 'GREATER_THAN'
    math_threshold.inputs[1].default_value = air_threshold_normalized
    math_threshold.label = "Air_Threshold"
    
    # Color Ramp: Tissue colors with sharp transitions (2 stops per tissue)
    ramp_color = nodes.new("ShaderNodeValToRGB")
    ramp_color.location = (-95.5784, -31.2797)
    ramp_color.label = "Tissue_Colors"
    
    # Calculate normalized positions from HU values
    fat_start_pos = hu_to_normalized(HU_FAT_MIN)
    fat_end_pos = hu_to_normalized(HU_FAT_MAX)
    soft_start_pos = hu_to_normalized(HU_SOFT_MIN)
    soft_end_pos = hu_to_normalized(HU_SOFT_MAX)
    bone_start_pos = hu_to_normalized(HU_BONE_MIN)
    bone_end_pos = hu_to_normalized(HU_BONE_MAX)
    
    log(f"Tissue positions (normalized):")
    log(f"  Fat: {fat_start_pos:.4f} - {fat_end_pos:.4f} (HU {HU_FAT_MIN} - {HU_FAT_MAX})")
    log(f"  Soft: {soft_start_pos:.4f} - {soft_end_pos:.4f} (HU {HU_SOFT_MIN} - {HU_SOFT_MAX})")
    log(f"  Bone: {bone_start_pos:.4f} - {bone_end_pos:.4f} (HU {HU_BONE_MIN} - {HU_BONE_MAX})")
    
    # Configure color stops for sharp tissue boundaries
    # Air/Background (before fat)
    air_pos = hu_to_normalized(HU_AIR_THRESHOLD)
    ramp_color.color_ramp.elements[0].position = air_pos
    ramp_color.color_ramp.elements[0].color = (*COLOR_AIR_RGB, 0.0)  # Black transparent
    
    # Fat tissue (sharp transition)
    ramp_color.color_ramp.elements[1].position = fat_start_pos
    ramp_color.color_ramp.elements[1].color = (*COLOR_AIR_RGB, 0.0)  # Fat START - transparent
    
    elem_fat_end = ramp_color.color_ramp.elements.new(fat_end_pos)
    elem_fat_end.color = (*COLOR_FAT_RGB, ALPHA_FAT_DEFAULT)  # Fat END - yellow opaque
    
    # Soft tissue (sharp transition)
    elem_soft_start = ramp_color.color_ramp.elements.new(soft_start_pos)
    elem_soft_start.color = (*COLOR_FAT_RGB, 0.0)  # Soft START - transparent (blend from fat color)
    
    elem_soft_end = ramp_color.color_ramp.elements.new(soft_end_pos)
    elem_soft_end.color = (*COLOR_SOFT_RGB, ALPHA_SOFT_DEFAULT)  # Soft END - red opaque
    
    # Bone (sharp transition)
    elem_bone_start = ramp_color.color_ramp.elements.new(bone_start_pos)
    elem_bone_start.color = (*COLOR_SOFT_RGB, 0.0)  # Bone START - transparent (blend from soft color)
    
    elem_bone_end = ramp_color.color_ramp.elements.new(bone_end_pos)
    elem_bone_end.color = (*COLOR_BONE_RGB, ALPHA_BONE_DEFAULT)  # Bone END - white opaque
    
    # Math.002: Scale alpha by 600
    math_002 = nodes.new("ShaderNodeMath")
    math_002.location = (243.4801, 83.9188)
    math_002.operation = 'MULTIPLY'
    math_002.inputs[1].default_value = 600.0
    math_002.label = "Alpha_Scale"
    
    # Math.003: Final multiply (threshold × alpha)
    math_003 = nodes.new("ShaderNodeMath")
    math_003.location = (482.8159, 169.0526)
    math_003.operation = 'MULTIPLY'
    math_003.label = "Final_Density"
    
    # Connections
    links.new(vol_info.outputs["Density"], math_threshold.inputs[0])
    links.new(vol_info.outputs["Density"], ramp_color.inputs["Fac"])
    
    # Density path: Color Ramp Alpha → Scale → Combine with Threshold
    links.new(ramp_color.outputs["Alpha"], math_002.inputs[0])
    links.new(math_002.outputs["Value"], math_003.inputs[0])
    links.new(math_threshold.outputs["Value"], math_003.inputs[1])
    
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
