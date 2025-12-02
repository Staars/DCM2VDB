"""Property definitions for DICOM importer"""

import bpy
from bpy.props import StringProperty, FloatProperty, IntProperty, CollectionProperty, BoolProperty, EnumProperty
from bpy.types import PropertyGroup

# Update callbacks must be defined BEFORE the PropertyGroup classes that use them

def update_tissue_alpha_dynamic(self, context):
    """Update alpha values in volume material color ramp dynamically based on preset"""
    
    # Skip updates during initialization
    if context.scene.get("_dicom_initializing", False):
        return
    
    print(f"[Properties] update_tissue_alpha_dynamic called for tissue: {self.tissue_name}, alpha: {self.alpha}")
    
    # Find the active volume material (could be CT_Volume_Material, MR_Volume_Material, etc.)
    mat = None
    for material in bpy.data.materials:
        if material.name.endswith("_Volume_Material") and material.use_nodes:
            mat = material
            break
    
    if not mat:
        print("[Properties] No volume material found")
        return
    
    print(f"[Properties] Found volume material: {mat.name}")

    # Find the color ramp node
    color_ramp = None
    for node in mat.node_tree.nodes:
        if node.type == 'VALTORGB' and node.label == "Tissue_Colors":
            color_ramp = node
            break
    
    if not color_ramp:
        print("[Properties] Color ramp node 'Tissue_Colors' not found")
        return
    

    # Load the active preset to get tissue order
    from .material_presets import load_preset
    preset_name = context.scene.dicom_active_material_preset
    if not preset_name:
        return
    
    preset = load_preset(preset_name)
    if not preset:
        print(f"[Properties] Failed to load preset: {preset_name}")
        return
    
    # Build a map of tissue_name -> alpha from the collection
    alpha_map = {}
    for tissue_alpha in context.scene.dicom_tissue_alphas:
        alpha_map[tissue_alpha.tissue_name] = tissue_alpha.alpha
    
    # Update color ramp elements
    # Stop 0: Air threshold (always transparent)
    # For each tissue: 2 stops (START and END)
    # START uses previous tissue's alpha (for transition)
    # END uses current tissue's alpha
    
    elements = color_ramp.color_ramp.elements
    tissues = preset.tissues  # Already sorted by order
    
    # Calculate expected number of stops: 2 stops per tissue (no special cases)
    expected_stops = 2 * len(tissues)
    
    if len(elements) < expected_stops:
        print(f"[Properties] Color ramp has {len(elements)} stops, expected {expected_stops}")
        return
    
    stop_idx = 0  # Start from first stop
    
    for tissue in tissues:
        tissue_name = tissue.get('name', '')
        tissue_alpha = alpha_map.get(tissue_name, tissue.get('alpha_default', 1.0))
        
        # START stop - uses current tissue's alpha (sharp transition)
        if stop_idx < len(elements):
            elements[stop_idx].color = (
                elements[stop_idx].color[0],
                elements[stop_idx].color[1],
                elements[stop_idx].color[2],
                tissue_alpha
            )
            stop_idx += 1
        
        # END stop - uses current tissue's alpha (sharp transition)
        if stop_idx < len(elements):
            elements[stop_idx].color = (
                elements[stop_idx].color[0],
                elements[stop_idx].color[1],
                elements[stop_idx].color[2],
                tissue_alpha
            )
            stop_idx += 1
    
    print(f"[Properties] Updated {stop_idx} color ramp stops")

# PropertyGroup classes (defined after update callbacks)

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

class DicomTissueAlphaProperty(PropertyGroup):
    """Dynamic tissue alpha property"""
    tissue_name: StringProperty(
        name="Tissue Name",
        description="Internal tissue identifier"
    )
    tissue_label: StringProperty(
        name="Tissue Label",
        description="Display label for tissue"
    )
    alpha: FloatProperty(
        name="Alpha",
        default=1.0,
        min=0.0,
        max=1.0,
        description="Opacity of tissue in volume rendering",
        update=update_tissue_alpha_dynamic
    )

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
    
    # Patient data (JSON serialized)
    bpy.types.Scene.dicom_patient_data = StringProperty(
        name="DICOM Patient Data",
        description="Serialized patient data (JSON)",
        default=""
    )
    
    # Active material preset
    bpy.types.Scene.dicom_active_material_preset = StringProperty(
        name="Material Preset",
        description="Active material preset name",
        default="ct_standard"
    )
    
    # Dynamic tissue alpha collection
    bpy.types.Scene.dicom_tissue_alphas = CollectionProperty(
        type=DicomTissueAlphaProperty,
        name="Tissue Alphas",
        description="Dynamic tissue opacity values"
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
    

    


def get_tissue_thresholds_from_preset(preset_name="ct_standard"):
    """Get tissue HU thresholds from active preset
    
    Returns dict: {
        'fat': {'min': -120, 'max': -90},
        'liquid': {'min': 0, 'max': 25},
        'soft_tissue': {'min': 35, 'max': 70},
        'connective_tissue': {'min': 75, 'max': 90},
        'bone': {'min': 400, 'max': 1000}
    }
    """
    from .material_presets import load_preset
    
    preset = load_preset(preset_name)
    if not preset:
        return {}
    
    thresholds = {}
    for tissue in preset.tissues:
        thresholds[tissue['name']] = {
            'min': tissue.get('hu_min', 0),
            'max': tissue.get('hu_max', 0)
        }
    return thresholds

def initialize_tissue_alphas(context, preset_name="ct_standard", silent=False):
    """Initialize tissue alpha collection from preset
    
    Args:
        context: Blender context
        preset_name: Name of preset to load
        silent: If True, suppress update callbacks during initialization
    """
    from .material_presets import load_preset
    
    preset = load_preset(preset_name)
    if not preset:
        print(f"[Properties] Failed to load preset {preset_name}")
        return
    
    # Clear existing alphas
    context.scene.dicom_tissue_alphas.clear()
    
    # Temporarily disable updates by setting a flag
    if silent:
        context.scene["_dicom_initializing"] = True
    
    # Add tissue alphas from preset (tissues are already sorted by order)
    for tissue in preset.tissues:
        tissue_alpha = context.scene.dicom_tissue_alphas.add()
        tissue_alpha.tissue_name = tissue.get('name', '')
        tissue_alpha.tissue_label = tissue.get('label', tissue.get('name', '').title())
        tissue_alpha.alpha = tissue.get('alpha_default', 1.0)
    
    # Re-enable updates
    if silent and "_dicom_initializing" in context.scene:
        del context.scene["_dicom_initializing"]
    
    # Store active preset name
    context.scene.dicom_active_material_preset = preset_name
    
    print(f"[Properties] Initialized {len(preset.tissues)} tissue alphas from preset '{preset_name}'")

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
    del bpy.types.Scene.dicom_active_tool
    del bpy.types.Scene.dicom_volume_data_path
    del bpy.types.Scene.dicom_volume_spacing
    del bpy.types.Scene.dicom_volume_unique_id
    del bpy.types.Scene.dicom_volume_hu_min
    del bpy.types.Scene.dicom_volume_hu_max
    del bpy.types.Scene.dicom_active_material_preset
    del bpy.types.Scene.dicom_tissue_alphas

classes = (
    DicomSeriesProperty,
    DicomTissueAlphaProperty,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    register_scene_props()
    
    # Initialize tissue alphas from default preset at startup
    # This ensures the collection is populated and callbacks are active
    # Use a deferred call to ensure scene is ready
    def init_on_startup():
        if bpy.context.scene:
            initialize_tissue_alphas(bpy.context, "ct_standard", silent=True)
            print("[Properties] Initialized tissue alphas at addon startup")
        return None  # Run once and stop
    
    bpy.app.timers.register(init_on_startup, first_interval=0.1)

def unregister():
    unregister_scene_props()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)