"""Property definitions for DICOM importer"""

import bpy
from bpy.props import StringProperty, FloatProperty, IntProperty, CollectionProperty
from bpy.types import PropertyGroup

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

def unregister_scene_props():
    """Unregister scene properties"""
    del bpy.types.Scene.dicom_import_folder
    del bpy.types.Scene.dicom_series_collection
    del bpy.types.Scene.dicom_series_data
    del bpy.types.Scene.dicom_preview_series_index
    del bpy.types.Scene.dicom_preview_slice_index
    del bpy.types.Scene.dicom_preview_slice_count

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