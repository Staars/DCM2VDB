"""Reusable shader node builders for materials"""

from .constants import *
from .volume_utils import hu_to_normalized

def build_hu_mapper(nodes, links, vol_min, vol_max, location=(-400, 0)):
    """Build Map Range node to convert HU to 0-1"""
    map_range = nodes.new("ShaderNodeMapRange")
    map_range.location = location
    map_range.inputs["From Min"].default_value = vol_min
    map_range.inputs["From Max"].default_value = vol_max
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0
    map_range.clamp = True
    return map_range

def build_tissue_color_ramp(nodes, vol_min, vol_max, location=(0, 100)):
    """Build color ramp for tissue visualization"""
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = location
    ramp.color_ramp.interpolation = 'LINEAR'
    
    # Clear defaults
    while len(ramp.color_ramp.elements) > 2:
        ramp.color_ramp.elements.remove(ramp.color_ramp.elements[0])
    
    # Air
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = COLOR_AIR
    
    # Add tissue stops
    ramp.color_ramp.elements.new(hu_to_normalized(HU_FAT, vol_min, vol_max))
    ramp.color_ramp.elements.new(hu_to_normalized(HU_SOFT_TISSUE, vol_min, vol_max))
    ramp.color_ramp.elements.new(hu_to_normalized(HU_BONE_DENSE, vol_min, vol_max))
    
    # Fat
    ramp.color_ramp.elements[1].color = COLOR_SOFT_TISSUE_RED
    # Soft tissue
    ramp.color_ramp.elements[2].color = COLOR_SOFT_TISSUE_PINK
    # Bone
    ramp.color_ramp.elements[3].color = COLOR_BONE_WHITE
    
    return ramp

def build_density_ramp(nodes, vol_min, vol_max, location=(0, 100)):
    """Build density ramp for volume absorption"""
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = location
    ramp.color_ramp.interpolation = 'LINEAR'
    
    # Setup density ramp
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    
    # Add stops for soft tissue and bone
    if len(ramp.color_ramp.elements) < 4:
        ramp.color_ramp.elements.new(0.3)
        ramp.color_ramp.elements.new(0.5)
        ramp.color_ramp.elements.new(0.7)
    
    # Soft tissue start
    ramp.color_ramp.elements[1].position = hu_to_normalized(HU_FAT, vol_min, vol_max)
    ramp.color_ramp.elements[1].color = (0.1, 0.1, 0.1, 1)
    
    # Soft tissue peak
    ramp.color_ramp.elements[2].position = hu_to_normalized(HU_SOFT_TISSUE, vol_min, vol_max)
    ramp.color_ramp.elements[2].color = (0.5, 0.5, 0.5, 1)
    
    # Bone
    ramp.color_ramp.elements[3].position = hu_to_normalized(HU_BONE_DENSE, vol_min, vol_max)
    ramp.color_ramp.elements[3].color = (1, 1, 1, 1)
    
    return ramp

def build_threshold_mask(nodes, threshold_hu=HU_AIR_THRESHOLD, location=(-200, 200)):
    """Build threshold node to mask air"""
    math_threshold = nodes.new("ShaderNodeMath")
    math_threshold.location = location
    math_threshold.operation = 'GREATER_THAN'
    math_threshold.inputs[1].default_value = threshold_hu
    math_threshold.label = "Air_Threshold"
    math_threshold.use_clamp = True
    return math_threshold

def build_mesh_color_ramp(nodes, vol_min, vol_max, location=(-100, 100)):
    """Build color ramp for mesh materials with tissue-specific colors"""
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = location
    ramp.color_ramp.interpolation = 'LINEAR'
    
    # Clear defaults
    while len(ramp.color_ramp.elements) > 2:
        ramp.color_ramp.elements.remove(ramp.color_ramp.elements[0])
    
    # Fat
    ramp.color_ramp.elements[0].position = hu_to_normalized(HU_FAT, vol_min, vol_max)
    ramp.color_ramp.elements[0].color = COLOR_FAT
    
    # Add more stops
    ramp.color_ramp.elements.new(hu_to_normalized(HU_SOFT_TISSUE, vol_min, vol_max))
    ramp.color_ramp.elements.new(hu_to_normalized(HU_BONE_START, vol_min, vol_max))
    ramp.color_ramp.elements.new(hu_to_normalized(HU_BONE_DENSE, vol_min, vol_max))
    
    # Muscle
    ramp.color_ramp.elements[1].color = COLOR_MUSCLE
    # Bone start
    ramp.color_ramp.elements[2].color = COLOR_BONE_LIGHT
    # Dense bone
    ramp.color_ramp.elements[3].color = COLOR_BONE_DENSE
    
    return ramp

def build_alpha_ramp(nodes, location=(-100, -100)):
    """Build alpha ramp for transparency based on density"""
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = location
    ramp.color_ramp.interpolation = 'LINEAR'
    
    # Low density = transparent, high density = opaque
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = 0.3
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    
    return ramp
