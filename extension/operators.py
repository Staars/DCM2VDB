"""Operator classes for DICOM import"""

import bpy
import os
from bpy.props import StringProperty, IntProperty
from bpy.types import Operator

from .dicom_io import PYDICOM_AVAILABLE, gather_dicom_files, organize_by_series, load_slice, log
from .volume import create_volume
from .preview import load_and_display_slice

class IMPORT_OT_dicom_scan(Operator):
    """Scan folder for DICOM series"""
    bl_idname = "import.dicom_scan"
    bl_label = "Scan DICOM Folder"
    bl_options = {'REGISTER'}
    
    directory: StringProperty(subtype="DIR_PATH")

    def execute(self, context):
        if not PYDICOM_AVAILABLE:
            self.report({'ERROR'}, "pydicom not installed. Install with: pip install pydicom pillow")
            return {'CANCELLED'}
        
        if not self.directory or not os.path.isdir(self.directory):
            self.report({'ERROR'}, "Invalid folder")
            return {'CANCELLED'}
        
        # Gather and organize files
        self.report({'INFO'}, "Scanning folder...")
        files = gather_dicom_files(self.directory)
        
        if not files:
            self.report({'ERROR'}, "No DICOM files found")
            return {'CANCELLED'}
        
        series_list = organize_by_series(files)
        
        if not series_list:
            self.report({'ERROR'}, "No valid DICOM series found")
            return {'CANCELLED'}
        
        # Store in scene
        context.scene.dicom_import_folder = self.directory
        context.scene.dicom_series_collection.clear()
        
        for series in series_list:
            item = context.scene.dicom_series_collection.add()
            item.uid = series['uid']
            item.description = series['description']
            item.modality = series['modality']
            item.number = series['number']
            item.instance_count = series['instance_count']
            item.rows = series['rows']
            item.cols = series['cols']
            item.window_center = series['window_center'] if series['window_center'] is not None else 0.0
            item.window_width = series['window_width'] if series['window_width'] is not None else 0.0
        
        context.scene.dicom_series_data = str(series_list)
        
        self.report({'INFO'}, f"Found {len(series_list)} series with {len(files)} total files")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class IMPORT_OT_dicom_preview(Operator):
    """Preview DICOM series in Image Editor"""
    bl_idname = "import.dicom_preview"
    bl_label = "Preview Series"
    bl_options = {'REGISTER'}
    
    series_index: IntProperty()

    def execute(self, context):
        # Get series data
        series_list = eval(context.scene.dicom_series_data)
        if self.series_index >= len(series_list):
            self.report({'ERROR'}, "Invalid series index")
            return {'CANCELLED'}
        
        series = series_list[self.series_index]
        
        # Store preview info in scene
        context.scene.dicom_preview_series_index = self.series_index
        context.scene.dicom_preview_slice_index = 0
        context.scene.dicom_preview_slice_count = len(series['files'])
        
        # Load first slice
        try:
            load_and_display_slice(context, series['files'][0], series)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load slice: {e}")
            return {'CANCELLED'}
        
        # Try to switch to image editor
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.ui_type = 'IMAGE_EDITOR'
                for space in area.spaces:
                    if space.type == 'IMAGE_EDITOR':
                        space.image = bpy.data.images.get("DICOM_Preview")
                break
        
        self.report({'INFO'}, f"Loaded series with {len(series['files'])} slices")
        return {'FINISHED'}

class IMPORT_OT_dicom_preview_slice(Operator):
    """Load a specific slice for preview"""
    bl_idname = "import.dicom_preview_slice"
    bl_label = "Load Slice"
    bl_options = {'INTERNAL'}
    
    slice_index: IntProperty()
    
    def execute(self, context):
        series_list = eval(context.scene.dicom_series_data)
        series_idx = context.scene.dicom_preview_series_index
        
        if series_idx >= len(series_list):
            return {'CANCELLED'}
        
        series = series_list[series_idx]
        
        if self.slice_index >= len(series['files']) or self.slice_index < 0:
            return {'CANCELLED'}
        
        context.scene.dicom_preview_slice_index = self.slice_index
        
        # Load and display the slice
        load_and_display_slice(context, series['files'][self.slice_index], series)
        
        return {'FINISHED'}

class IMAGE_OT_dicom_scroll(Operator):
    """Scroll through DICOM slices with mouse wheel"""
    bl_idname = "image.dicom_scroll"
    bl_label = "Scroll DICOM Slices"
    bl_options = {'INTERNAL'}
    
    direction: IntProperty(default=0)
    
    def execute(self, context):
        if context.scene.dicom_preview_slice_count > 0:
            current = context.scene.dicom_preview_slice_index
            new_index = current + self.direction
            new_index = max(0, min(context.scene.dicom_preview_slice_count - 1, new_index))
            
            if new_index != current:
                # Load the slice directly
                series_list = eval(context.scene.dicom_series_data)
                series_idx = context.scene.dicom_preview_series_index
                
                if series_idx < len(series_list):
                    series = series_list[series_idx]
                    if new_index < len(series['files']):
                        context.scene.dicom_preview_slice_index = new_index
                        load_and_display_slice(context, series['files'][new_index], series)
        
        return {'FINISHED'}

class IMPORT_OT_dicom_import_series(Operator):
    """Import selected DICOM series as 3D volume"""
    bl_idname = "import.dicom_import_series"
    bl_label = "Import Series as Volume"
    bl_options = {'REGISTER', 'UNDO'}
    
    series_index: IntProperty()

    def execute(self, context):
        series_list = eval(context.scene.dicom_series_data)
        if self.series_index >= len(series_list):
            self.report({'ERROR'}, "Invalid series index")
            return {'CANCELLED'}
        
        series = series_list[self.series_index]
        
        # Load all slices
        wm = context.window_manager
        wm.progress_begin(0, len(series['files']))
        slices = []
        
        for i, path in enumerate(series['files']):
            wm.progress_update(i)
            try:
                slices.append(load_slice(path))
            except Exception as e:
                log(f"Failed to load {path}: {e}")
        
        wm.progress_end()
        
        if len(slices) < 2:
            self.report({'ERROR'}, "Need at least 2 valid slices")
            return {'CANCELLED'}
        
        try:
            create_volume(slices)
            self.report({'INFO'}, f"Imported {len(slices)} slices as volume")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create volume: {e}")
            return {'CANCELLED'}

classes = (
    IMPORT_OT_dicom_scan,
    IMPORT_OT_dicom_preview,
    IMPORT_OT_dicom_preview_slice,
    IMAGE_OT_dicom_scroll,
    IMPORT_OT_dicom_import_series,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)