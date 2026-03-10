"""Visualization, display mode, and series management operators"""

import bpy
import os
import json
import numpy as np
from bpy.props import StringProperty, IntProperty
from bpy.types import Operator

from ..dicom_io import load_slice
from ..volume_creation import create_volume
from ..patient import Patient
from ..utils import SimpleLogger

log = SimpleLogger()


class IMPORT_OT_dicom_visualize_series(Operator):
    """Visualize a series from the loaded patient"""
    bl_idname = "import.dicom_visualize_series"
    bl_label = "Visualize Series"
    bl_options = {'REGISTER', 'UNDO'}
    
    series_uid: StringProperty()
    
    def execute(self, context):
        # Load patient from scene
        if not context.scene.dicom_patient_data:
            self.report({'ERROR'}, "No patient loaded")
            return {'CANCELLED'}
        
        try:
            patient = Patient.from_json(context.scene.dicom_patient_data)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load patient data: {e}")
            return {'CANCELLED'}
        
        # Find the series
        series = patient.get_series_by_uid(self.series_uid)
        if not series:
            self.report({'ERROR'}, f"Series not found: {self.series_uid}")
            return {'CANCELLED'}
        
        # Build absolute file paths
        file_paths = []
        for rel_path in series.file_paths:
            abs_path = os.path.join(patient.dicom_root_path, rel_path)
            file_paths.append(abs_path)
        
        # Load slices
        wm = context.window_manager
        wm.progress_begin(0, len(file_paths))
        slices = []
        
        for i, path in enumerate(file_paths):
            wm.progress_update(i)
            slice_data = load_slice(path)
            if slice_data is not None:
                slices.append(slice_data)
            else:
                log.warning(f"Skipped slice (no pixel data): {path}")
        
        wm.progress_end()
        
        if len(slices) < 2:
            self.report({'ERROR'}, "Need at least 2 valid slices")
            return {'CANCELLED'}
        
        try:
            # Create volume with series number for unique naming
            vol_obj = create_volume(slices, series_number=series.series_number)
            
            # Update series state
            series.is_loaded = True
            series.is_visible = True
            
            # Store volume object reference
            patient.volume_objects[self.series_uid] = vol_obj.name
            
            # Save updated patient data
            context.scene.dicom_patient_data = patient.to_json()
            
            # Automatically calculate tissue volumes for this series
            from ..measurements.tissue_volumes import calculate_and_store_tissue_volumes
            calculate_and_store_tissue_volumes(context, series)
            
            # Save updated measurements
            context.scene.dicom_patient_data = patient.to_json()
            
            self.report({'INFO'}, f"Visualized: {series.series_description} ({len(slices)} slices)")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create volume: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class IMPORT_OT_dicom_toggle_series_selection(Operator):
    """Toggle series selection for processing"""
    bl_idname = "import.dicom_toggle_series_selection"
    bl_label = "Toggle Series Selection"
    bl_options = {'REGISTER', 'UNDO'}
    
    series_uid: StringProperty()
    
    def execute(self, context):
        if not context.scene.dicom_patient_data:
            return {'CANCELLED'}
        
        try:
            patient = Patient.from_json(context.scene.dicom_patient_data)
            series = patient.get_series_by_uid(self.series_uid)
            
            if series:
                series.is_selected = not series.is_selected
                context.scene.dicom_patient_data = patient.to_json()
                
                status = "selected" if series.is_selected else "deselected"
                self.report({'INFO'}, f"Series {series.series_number} {status}")
            
            return {'FINISHED'}
        except:
            return {'CANCELLED'}


class IMPORT_OT_dicom_toggle_series_visibility(Operator):
    """Toggle volume/bone visibility for a specific series"""
    bl_idname = "import.dicom_toggle_series_visibility"
    bl_label = "Toggle Series Visibility"
    bl_options = {'REGISTER', 'UNDO'}
    
    series_uid: StringProperty()
    visibility_type: StringProperty()  # 'volume' or 'bone'
    
    def execute(self, context):
        if not context.scene.dicom_patient_data:
            return {'CANCELLED'}
        
        try:
            patient = Patient.from_json(context.scene.dicom_patient_data)
            series = patient.get_series_by_uid(self.series_uid)
            
            if series and series.is_loaded:
                if self.visibility_type == 'volume':
                    series.show_volume = not series.show_volume
                    # Update actual object visibility
                    vol_obj = bpy.data.objects.get(f"CT_Volume_S{series.series_number}")
                    if vol_obj:
                        vol_obj.hide_viewport = not series.show_volume
                elif self.visibility_type == 'bone':
                    series.show_bone = not series.show_bone
                    # Update bone mesh modifier visibility
                    bone_obj = bpy.data.objects.get(f"CT_Bone_S{series.series_number}")
                    if bone_obj:
                        for mod in bone_obj.modifiers:
                            if mod.type == 'NODES':
                                mod.show_viewport = series.show_bone
                                break
                
                context.scene.dicom_patient_data = patient.to_json()
            
            return {'FINISHED'}
        except:
            return {'CANCELLED'}


class IMPORT_OT_dicom_toggle_display_mode(Operator):
    """Toggle between volume and mesh display mode (4 tissue meshes)"""
    bl_idname = "import.dicom_toggle_display_mode"
    bl_label = "Toggle Display Mode"
    bl_options = {'REGISTER', 'UNDO'}
    
    mode: StringProperty(default='VOLUME')  # 'VOLUME' or 'MESH'
    
    def execute(self, context):
        if not context.scene.dicom_patient_data:
            return {'CANCELLED'}
        
        try:
            patient = Patient.from_json(context.scene.dicom_patient_data)
        except:
            return {'CANCELLED'}
        
        show_mesh = (self.mode == 'MESH')
        count = 0
        
        # Toggle each volume and its 4 tissue meshes
        for obj_name in patient.volume_objects.values():
            vol_obj = bpy.data.objects.get(obj_name)
            if not vol_obj:
                continue
            
            # Hide/show the volume object itself
            vol_obj.hide_viewport = show_mesh
            vol_obj.hide_render = show_mesh
            
            # Find and toggle the 4 tissue mesh objects
            tissue_names = ["CT_Fat", "CT_Fluid", "CT_SoftTissue", "CT_Bone"]
            for tissue_name in tissue_names:
                tissue_obj = bpy.data.objects.get(tissue_name)
                if tissue_obj:
                    # Find the geometry nodes modifier
                    for mod in tissue_obj.modifiers:
                        if mod.type == 'NODES':
                            mod.show_viewport = show_mesh
                            mod.show_render = show_mesh
                            count += 1
                            break
        
        mode_name = "Mesh" if show_mesh else "Volume"
        self.report({'INFO'}, f"Switched to {mode_name} mode ({count} tissue meshes)")
        
        # Force viewport update
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
        return {'FINISHED'}


class IMPORT_OT_dicom_calculate_volume(Operator):
    """Calculate tissue volume for a specific HU range"""
    bl_idname = "import.dicom_calculate_volume"
    bl_label = "Calculate Volume"
    bl_options = {'REGISTER', 'UNDO'}
    
    tissue_type: StringProperty()  # 'fat', 'fluid', 'soft'
    
    def execute(self, context):
        from ..measurements.tissue_volumes import calculate_tissue_volume
        
        scn = context.scene
        
        # Check if volume data is available
        if not scn.dicom_volume_data_path or not os.path.exists(scn.dicom_volume_data_path):
            self.report({'ERROR'}, "No volume data available. Please load a DICOM series first.")
            return {'CANCELLED'}
        
        try:
            # Load volume data
            vol_array = np.load(scn.dicom_volume_data_path)
            
            # Parse spacing
            spacing = json.loads(scn.dicom_volume_spacing)  # [X, Y, Z] in mm
            pixel_spacing = (spacing[1], spacing[0])  # (row, col) = (Y, X)
            slice_thickness = spacing[2]  # Z
            
            # Get HU thresholds based on tissue type (from preset)
            from ..properties import get_tissue_thresholds_from_preset
            thresholds = get_tissue_thresholds_from_preset(scn.dicom_active_material_preset)
            
            # Map tissue_type to preset tissue name
            tissue_map = {
                'fat': 'fat',
                'fluid': 'liquid',
                'soft': 'soft_tissue'
            }
            
            preset_tissue_name = tissue_map.get(self.tissue_type)
            if not preset_tissue_name or preset_tissue_name not in thresholds:
                self.report({'ERROR'}, f"Unknown tissue type: {self.tissue_type}")
                return {'CANCELLED'}
            
            tissue_range = thresholds[preset_tissue_name]
            hu_min = tissue_range.get('min', 0)
            hu_max = tissue_range.get('max', 0)
            
            # Calculate volume
            volume_ml = calculate_tissue_volume(vol_array, hu_min, hu_max, pixel_spacing, slice_thickness)
            
            self.report({'INFO'}, f"{self.tissue_type.title()}: {volume_ml:.2f} mL")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to calculate volume: {e}")
            log.error(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class IMPORT_OT_dicom_bake_bone_mesh(Operator):
    """Convert geometry nodes bone mesh to real mesh"""
    bl_idname = "import.dicom_bake_bone_mesh"
    bl_label = "Bake Bone Meshes"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Find all bone objects with geometry nodes
        bone_objects = []
        for obj in bpy.data.objects:
            if 'Bone' in obj.name:
                for mod in obj.modifiers:
                    if mod.type == 'NODES':
                        bone_objects.append(obj)
                        break
        
        if not bone_objects:
            self.report({'WARNING'}, "No bone objects with geometry nodes found")
            return {'CANCELLED'}
        
        # Deselect all
        bpy.ops.object.select_all(action='DESELECT')
        
        # Select bone objects
        for obj in bone_objects:
            obj.select_set(True)
        
        # Set one as active
        context.view_layer.objects.active = bone_objects[0]
        
        # Call Blender's built-in operator
        try:
            result = bpy.ops.object.visual_geometry_to_objects()
            
            if result == {'FINISHED'}:
                self.report({'INFO'}, f"Baked {len(bone_objects)} bone mesh(es)")
                log.info(f"Successfully baked {len(bone_objects)} bone objects")
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, "Bake operation did not complete successfully")
                return {'CANCELLED'}
                
        except AttributeError:
            self.report({'ERROR'}, "Visual Geometry to Objects operator not available in this Blender version")
            log.error("bpy.ops.object.visual_geometry_to_objects not found")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to bake: {e}")
            log.error(f"Bake error: {e}")
            return {'CANCELLED'}
