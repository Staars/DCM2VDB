"""Material creation for volume and mesh visualization"""

import bpy
from .dicom_io import log
from .constants import *
from .volume_utils import hu_to_normalized
from .material_presets import load_preset

def create_volume_material(vol_obj, vol_min, vol_max, preset_name="ct_standard"):
    """Create or reuse shared CT volume material from preset"""
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
    
    # Load preset
    preset = load_preset(preset_name)
    if not preset:
        log(f"Failed to load preset '{preset_name}', using defaults")
        preset = load_preset("ct_standard")  # Fallback
        if not preset:
            log("ERROR: No presets available!")
            return
    
    # Create new shared material
    log(f"Creating new shared material: {mat_name} from preset '{preset.name}'")
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
    
    # Calculate normalized air threshold from preset
    air_threshold_normalized = hu_to_normalized(preset.air_threshold)
    log(f"Using preset HU range: {preset.hu_min} to {preset.hu_max}")
    log(f"Actual volume HU range: {vol_min:.1f} to {vol_max:.1f}")
    log(f"Air threshold: {preset.air_threshold} HU = {air_threshold_normalized:.6f} normalized")
    log(f"Density multiplier: {preset.density_multiplier}")
    
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
    
    # Calculate normalized positions from preset tissue data
    log(f"Tissue positions (normalized):")
    tissue_positions = []
    for tissue in preset.tissues:
        start_pos = hu_to_normalized(tissue["hu_min"])
        end_pos = hu_to_normalized(tissue["hu_max"])
        tissue_positions.append({
            "name": tissue["name"],
            "label": tissue["label"],
            "start_pos": start_pos,
            "end_pos": end_pos,
            "color_rgb": tuple(tissue["color_rgb"]),
            "alpha": tissue["alpha_default"]
        })
        log(f"  {tissue['label']}: {start_pos:.4f} - {end_pos:.4f} (HU {tissue['hu_min']} - {tissue['hu_max']})")
    
    # Configure color stops dynamically from preset
    # Remove extra default elements (keep minimum 2 required by Blender)
    while len(ramp_color.color_ramp.elements) > 2:
        ramp_color.color_ramp.elements.remove(ramp_color.color_ramp.elements[0])
    
    # Create tissue stops dynamically (2 stops per tissue: START and END)
    # Sharp transitions: both stops use the SAME color/alpha for each tissue
    # Blending happens in gaps between tissues, not within tissue ranges
    
    for i, tissue in enumerate(tissue_positions):
        # Both START and END use this tissue's color/alpha for sharp transitions
        tissue_color = tissue["color_rgb"]
        tissue_alpha = tissue["alpha"]
        
        if i == 0:
            # First tissue: use existing elements[0] and [1]
            ramp_color.color_ramp.elements[0].position = tissue["start_pos"]
            ramp_color.color_ramp.elements[0].color = (*tissue_color, tissue_alpha)
            log(f"  Stop {i*2}: {tissue['label']} START at {tissue['start_pos']:.4f} (color: {tissue_color}, alpha: {tissue_alpha})")
            
            ramp_color.color_ramp.elements[1].position = tissue["end_pos"]
            ramp_color.color_ramp.elements[1].color = (*tissue_color, tissue_alpha)
            log(f"  Stop {i*2+1}: {tissue['label']} END at {tissue['end_pos']:.4f} (color: {tissue_color}, alpha: {tissue_alpha})")
        else:
            # Subsequent tissues: create new elements
            elem_start = ramp_color.color_ramp.elements.new(tissue["start_pos"])
            elem_start.color = (*tissue_color, tissue_alpha)
            log(f"  Stop {i*2}: {tissue['label']} START at {tissue['start_pos']:.4f} (color: {tissue_color}, alpha: {tissue_alpha})")
            
            elem_end = ramp_color.color_ramp.elements.new(tissue["end_pos"])
            elem_end.color = (*tissue_color, tissue_alpha)
            log(f"  Stop {i*2+1}: {tissue['label']} END at {tissue['end_pos']:.4f} (color: {tissue_color}, alpha: {tissue_alpha})")
    
    # Math.002: Scale alpha by preset multiplier
    math_002 = nodes.new("ShaderNodeMath")
    math_002.location = (243.4801, 83.9188)
    math_002.operation = 'MULTIPLY'
    math_002.inputs[1].default_value = preset.density_multiplier
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
