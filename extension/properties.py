"""Property definitions for DICOM importer"""

import bpy
from bpy.props import StringProperty, FloatProperty, IntProperty, CollectionProperty, BoolProperty, EnumProperty
from bpy.types import PropertyGroup
from .constants import ALPHA_FAT_DEFAULT, ALPHA_SOFT_DEFAULT, ALPHA_BONE_DEFAULT

class DicomSeriesProperty(PropertyGroup):
    """Properties for a single DICOM series"""
    uid: StringProperty()
    description: StringProperty()
    modality: StringProperty()
    number: IntProperty()
    instance_count: IntProperty()
    rows: IntProperty()
    cols: IntProperty()
    window_center: FloatProperty()
    window_width: FloatProperty()

def update_volume_visibility(self, context):
    """Update volume object visibility for all series"""
    # Find all CT_Volume_S* objects
    for obj in bpy.data.objects:
        if obj.name.startswith("CT_Volume_S"):
            obj.hide_viewport = not context.scene.dicom_show_volume

def update_tissue_visibility(self, context, tissue_prefix, prop_name):
    """Update tissue mesh visibility for all series"""
    show = getattr(context.scene, prop_name)
    # Find all objects matching the tissue prefix (e.g., CT_Bone_S*)
    for obj in bpy.data.objects:
        if obj.name.startswith(tissue_prefix):
            # Toggle geometry nodes modifier
            for mod in obj.modifiers:
                if mod.type == 'NODES':
                    mod.show_viewport = show
                    break

def update_tissue_alpha(self, context):
    """Update alpha values in volume material color ramp (both START and END stops per tissue)"""
    mat = bpy.data.materials.get("CT_Volume_Material")
    if not mat or not mat.use_nodes:
        return
    
    # Find the color ramp node
    color_ramp = None
    for node in mat.node_tree.nodes:
        if node.type == 'VALTORGB' and node.label == "Tissue_Colors":
            color_ramp = node
            break
    
    if not color_ramp:
        return
    
    # Update alpha values for each tissue
    # Each tissue has START (transparent) and END (slider value) stops
    # Stop 0: Air threshold - transparent
    # Stop 1: Fat START - transparent
    # Stop 2: Fat END - controlled by fat slider
    # Stop 3: Soft START - controlled by fat slider (transition from fat)
    # Stop 4: Soft END - controlled by soft slider
    # Stop 5: Bone START - controlled by soft slider (transition from soft)
    # Stop 6: Bone END - controlled by bone slider
    
    elements = color_ramp.color_ramp.elements
    if len(elements) >= 7:
        fat_alpha = context.scene.dicom_tissue_alpha_fat
        soft_alpha = context.scene.dicom_tissue_alpha_soft
        bone_alpha = context.scene.dicom_tissue_alpha_bone
        
        # Fat END (stop 2) - uses fat slider
        elements[2].color = (elements[2].color[0], elements[2].color[1], elements[2].color[2], fat_alpha)
        
        # Soft START (stop 3) - uses fat slider (transition from fat color)
        elements[3].color = (elements[3].color[0], elements[3].color[1], elements[3].color[2], fat_alpha)
        
        # Soft END (stop 4) - uses soft slider
        elements[4].color = (elements[4].color[0], elements[4].color[1], elements[4].color[2], soft_alpha)
        
        # Bone START (stop 5) - uses soft slider (transition from soft color)
        elements[5].color = (elements[5].color[0], elements[5].color[1], elements[5].color[2], soft_alpha)
        
        # Bone END (stop 6) - uses bone slider
        elements[6].color = (elements[6].color[0], elements[6].color[1], elements[6].color[2], bone_alpha)

def register_scene_props():
    """Register scene properties"""
    bpy.types.Scene.dicom_import_folder = StringProperty(
        name="Folder", 
        subtype="DIR_PATH"
    )
    bpy.types.Scene.dicom_series_collection = CollectionProperty(
        type=DicomSeriesProperty
    )
    bpy.types.Scene.dicom_series_data = StringProperty()
    bpy.types.Scene.dicom_preview_series_index = IntProperty(default=-1)
    bpy.types.Scene.dicom_preview_slice_index = IntProperty(
        name="Slice",
        default=0,
        min=0
    )
    bpy.types.Scene.dicom_preview_slice_count = IntProperty(default=0)
    bpy.types.Scene.dicom_show_series_list = BoolProperty(
        name="Show Series List",
        default=True,
        description="Expand/collapse series list"
    )
    bpy.types.Scene.dicom_show_spatial_info = BoolProperty(
        name="Show Spatial Info",
        default=False,
        description="Expand/collapse spatial information in preview"
    )
    bpy.types.Scene.dicom_debug_pyramid = BoolProperty(
        name="Create Debug Pyramid",
        default=False,
        description="Create test pyramid with each import for comparison"
    )
    
    # Patient data (JSON serialized)
    bpy.types.Scene.dicom_patient_data = StringProperty(
        name="DICOM Patient Data",
        description="Serialized patient data (JSON)",
        default=""
    )
    
    # Active tool selection
    bpy.types.Scene.dicom_active_tool = EnumProperty(
        name="Active Tool",
        description="Select which tool to use with the loaded patient data",
        items=[
            ('NONE', "None", "No tool selected", 'QUESTION', 0),
            ('VISUALIZATION', "Visualization", "View and render volumes", 'SHADING_RENDERED', 1),
            ('MEASUREMENT', "Measurement", "Measure distances, angles, volumes (Coming Soon)", 'DRIVER_DISTANCE', 2),
            ('SEGMENTATION', "Segmentation", "Isolate anatomical structures (Coming Soon)", 'MOD_MASK', 3),
            ('REGISTRATION', "Registration", "Align multiple scans (Coming Soon)", 'ORIENTATION_GIMBAL', 4),
            ('EXPORT', "Export", "Export data and meshes (Coming Soon)", 'EXPORT', 5),
            ('ANALYSIS', "Analysis", "Density analysis and statistics (Coming Soon)", 'GRAPH', 6),
        ],
        default='NONE'
    )
    
    # Tissue threshold settings
    bpy.types.Scene.dicom_show_tissue_thresholds = BoolProperty(
        name="Show Tissue Thresholds",
        default=False,
        description="Expand/collapse tissue threshold settings"
    )
    
    # Fat thresholds
    bpy.types.Scene.dicom_fat_min = FloatProperty(
        name="Fat Min",
        default=-160.0,
        description="Minimum HU value for fat tissue"
    )
    bpy.types.Scene.dicom_fat_max = FloatProperty(
        name="Fat Max",
        default=0.0,
        description="Maximum HU value for fat tissue"
    )
    
    # Fluid thresholds
    bpy.types.Scene.dicom_fluid_min = FloatProperty(
        name="Fluid Min",
        default=0.0,
        description="Minimum HU value for fluid"
    )
    bpy.types.Scene.dicom_fluid_max = FloatProperty(
        name="Fluid Max",
        default=30.0,
        description="Maximum HU value for fluid"
    )
    
    # Soft tissue thresholds
    bpy.types.Scene.dicom_soft_min = FloatProperty(
        name="Soft Tissue Min",
        default=35.0,
        description="Minimum HU value for soft tissue"
    )
    bpy.types.Scene.dicom_soft_max = FloatProperty(
        name="Soft Tissue Max",
        default=100.0,
        description="Maximum HU value for soft tissue"
    )
    
    # Bone threshold
    bpy.types.Scene.dicom_bone_min = FloatProperty(
        name="Bone Min",
        default=400.0,
        description="Minimum HU value for bone"
    )
    
    # Volume data cache (for recomputation)
    bpy.types.Scene.dicom_volume_data_path = StringProperty(
        name="Volume Data Path",
        description="Path to cached numpy array for recomputation",
        default=""
    )
    bpy.types.Scene.dicom_volume_spacing = StringProperty(
        name="Volume Spacing",
        description="Cached spacing values (JSON)",
        default=""
    )
    bpy.types.Scene.dicom_volume_unique_id = StringProperty(
        name="Volume Unique ID",
        description="Unique ID for this volume session",
        default=""
    )
    bpy.types.Scene.dicom_volume_hu_min = FloatProperty(
        name="Volume HU Min",
        description="Minimum HU value in volume (for normalization)",
        default=-1024.0
    )
    bpy.types.Scene.dicom_volume_hu_max = FloatProperty(
        name="Volume HU Max",
        description="Maximum HU value in volume (for normalization)",
        default=3000.0
    )
    
    # Visibility toggles
    bpy.types.Scene.dicom_show_volume = BoolProperty(
        name="Show Volume",
        default=True,
        description="Show/hide the main volume",
        update=update_volume_visibility
    )
    bpy.types.Scene.dicom_show_fat = BoolProperty(
        name="Show Fat Mesh",
        default=False,
        description="Show/hide fat tissue mesh",
        update=lambda self, context: update_tissue_visibility(self, context, "CT_Fat", "dicom_show_fat")
    )
    bpy.types.Scene.dicom_show_fluid = BoolProperty(
        name="Show Fluid Mesh",
        default=False,
        description="Show/hide fluid mesh",
        update=lambda self, context: update_tissue_visibility(self, context, "CT_Fluid", "dicom_show_fluid")
    )
    bpy.types.Scene.dicom_show_soft = BoolProperty(
        name="Show Soft Tissue Mesh",
        default=False,
        description="Show/hide soft tissue mesh",
        update=lambda self, context: update_tissue_visibility(self, context, "CT_SoftTissue", "dicom_show_soft")
    )
    bpy.types.Scene.dicom_show_bone = BoolProperty(
        name="Show Bone Mesh",
        default=False,
        description="Show/hide bone mesh",
        update=lambda self, context: update_tissue_visibility(self, context, "CT_Bone_S", "dicom_show_bone")
    )
    
    # Tissue opacity controls (for volume material color ramp alpha)
    bpy.types.Scene.dicom_tissue_alpha_fat = FloatProperty(
        name="Fat",
        default=ALPHA_FAT_DEFAULT,
        min=0.0,
        max=1.0,
        description="Opacity of fat tissue in volume rendering",
        update=lambda self, context: update_tissue_alpha(self, context)
    )
    bpy.types.Scene.dicom_tissue_alpha_soft = FloatProperty(
        name="Soft Tissue",
        default=ALPHA_SOFT_DEFAULT,
        min=0.0,
        max=1.0,
        description="Opacity of soft tissue in volume rendering",
        update=lambda self, context: update_tissue_alpha(self, context)
    )
    bpy.types.Scene.dicom_tissue_alpha_bone = FloatProperty(
        name="Bone",
        default=ALPHA_BONE_DEFAULT,
        min=0.0,
        max=1.0,
        description="Opacity of bone in volume rendering",
        update=lambda self, context: update_tissue_alpha(self, context)
    )
    
    # Measurement results
    bpy.types.Scene.dicom_fat_volume_ml = FloatProperty(
        name="Fat Volume",
        default=0.0,
        description="Calculated fat tissue volume in mL",
        precision=2
    )
    bpy.types.Scene.dicom_fluid_volume_ml = FloatProperty(
        name="Fluid Volume",
        default=0.0,
        description="Calculated fluid volume in mL",
        precision=2
    )
    bpy.types.Scene.dicom_soft_volume_ml = FloatProperty(
        name="Soft Tissue Volume",
        default=0.0,
        description="Calculated soft tissue volume in mL",
        precision=2
    )
    
    # Measurement mask visibility
    bpy.types.Scene.dicom_show_fat_mask = BoolProperty(
        name="Show Fat Mask",
        default=False,
        description="Show/hide fat measurement mask"
    )
    bpy.types.Scene.dicom_show_fluid_mask = BoolProperty(
        name="Show Fluid Mask",
        default=False,
        description="Show/hide fluid measurement mask"
    )
    bpy.types.Scene.dicom_show_soft_mask = BoolProperty(
        name="Show Soft Mask",
        default=False,
        description="Show/hide soft tissue measurement mask"
    )

def unregister_scene_props():
    """Unregister scene properties"""
    del bpy.types.Scene.dicom_import_folder
    del bpy.types.Scene.dicom_series_collection
    del bpy.types.Scene.dicom_series_data
    del bpy.types.Scene.dicom_preview_series_index
    del bpy.types.Scene.dicom_preview_slice_index
    del bpy.types.Scene.dicom_preview_slice_count
    del bpy.types.Scene.dicom_show_series_list
    del bpy.types.Scene.dicom_show_spatial_info
    del bpy.types.Scene.dicom_patient_data
    del bpy.types.Scene.dicom_debug_pyramid
    del bpy.types.Scene.dicom_active_tool
    del bpy.types.Scene.dicom_show_tissue_thresholds
    del bpy.types.Scene.dicom_fat_min
    del bpy.types.Scene.dicom_fat_max
    del bpy.types.Scene.dicom_fluid_min
    del bpy.types.Scene.dicom_fluid_max
    del bpy.types.Scene.dicom_soft_min
    del bpy.types.Scene.dicom_soft_max
    del bpy.types.Scene.dicom_bone_min
    del bpy.types.Scene.dicom_volume_data_path
    del bpy.types.Scene.dicom_volume_spacing
    del bpy.types.Scene.dicom_volume_unique_id
    del bpy.types.Scene.dicom_volume_hu_min
    del bpy.types.Scene.dicom_volume_hu_max
    del bpy.types.Scene.dicom_show_volume
    del bpy.types.Scene.dicom_show_fat
    del bpy.types.Scene.dicom_show_fluid
    del bpy.types.Scene.dicom_show_soft
    del bpy.types.Scene.dicom_show_bone
    del bpy.types.Scene.dicom_tissue_alpha_fat
    del bpy.types.Scene.dicom_tissue_alpha_soft
    del bpy.types.Scene.dicom_tissue_alpha_bone
    del bpy.types.Scene.dicom_fat_volume_ml
    del bpy.types.Scene.dicom_fluid_volume_ml
    del bpy.types.Scene.dicom_soft_volume_ml
    del bpy.types.Scene.dicom_show_fat_mask
    del bpy.types.Scene.dicom_show_fluid_mask
    del bpy.types.Scene.dicom_show_soft_mask

classes = (
    DicomSeriesProperty,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    register_scene_props()

def unregister():
    unregister_scene_props()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)