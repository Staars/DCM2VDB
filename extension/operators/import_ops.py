"""Import and patient loading operators"""

import bpy
import os
import json
from bpy.props import StringProperty, IntProperty
from bpy.types import Operator

from ..dicom_io import PYDICOM_AVAILABLE, gather_dicom_files, organize_by_series, load_slice, load_patient_from_folder
from ..volume_creation import create_volume
from ..patient import Patient
from ..utils import SimpleLogger

import numpy as np

log = SimpleLogger()


class IMPORT_OT_dicom_load_patient(Operator):
    """Load patient data from DICOM folder (automatic)"""
    bl_idname = "import.dicom_load_patient"
    bl_label = "Load Patient"
    bl_options = {'REGISTER'}
    
    directory: StringProperty(subtype="DIR_PATH")

    def execute(self, context):
        if not PYDICOM_AVAILABLE:
            self.report({'ERROR'}, "pydicom not installed. Install with: pip install pydicom pillow")
            return {'CANCELLED'}
        
        if not self.directory or not os.path.isdir(self.directory):
            self.report({'ERROR'}, "Invalid folder")
            return {'CANCELLED'}
        
        try:
            # Clear patient data FIRST to prevent UI from accessing deleted objects
            context.scene.dicom_patient_data = ""
            context.scene.dicom_active_tool = 'NONE'
            
            # COMPLETE CLEANUP before loading new patient
            self._cleanup_all(context)
            
            # Load patient data (automatic - loads all primary series)
            patient = load_patient_from_folder(self.directory)
            
            # Serialize and store in scene
            context.scene.dicom_patient_data = patient.to_json()
            
            # Report summary
            self.report({'INFO'}, 
                f"Loaded patient: {patient.patient_name} ({patient.patient_id})")
            self.report({'INFO'}, 
                f"✓ {len(patient.series)} primary series (from {patient.primary_count} images)")
            
            if patient.secondary_count > 0:
                self.report({'INFO'}, 
                    f"ℹ Ignored {patient.secondary_count} secondary images")
            
            if patient.non_image_count > 0:
                self.report({'INFO'}, 
                    f"ℹ Ignored {patient.non_image_count} non-image files")
            
            # Log series details
            log.debug(f"Primary series loaded:")
            for series in patient.series:
                log.info(f"  - {series.series_description} ({series.modality}): {series.slice_count} slices")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load patient: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
    def _cleanup_all(self, context):
        """Complete cleanup: remove all DICOM objects, materials, images, and reset properties"""
        log.info("=" * 60)
        log.info("COMPLETE CLEANUP - Removing all DICOM data")
        log.info("=" * 60)
        
        # 1. Remove all DICOM objects (modality-aware naming)
        removed_count = 0
        for obj in list(bpy.data.objects):
            # Check if it's a DICOM object (CT_, MR_, or other modality prefixes)
            if any(pattern in obj.name for pattern in ['_Volume_S', '_Volume', '_Bone_S', '_Bone', '_Mesh_S', '_Mesh', 'DEBUG_Pyramid']):
                obj_name = obj.name
                bpy.data.objects.remove(obj, do_unlink=True)
                log.info(f"Removed object: {obj_name}")
                removed_count += 1
        log.debug(f"Removed {removed_count} DICOM objects")
        
        # 2. Remove all DICOM materials (modality-aware)
        removed_mat_count = 0
        for mat in list(bpy.data.materials):
            if mat.name.endswith("_Volume_Material") or mat.name.endswith("_Bone_Material") or mat.name.endswith("_Mesh_Material"):
                mat_name = mat.name  # Save name before removing
                bpy.data.materials.remove(mat)
                log.info(f"Removed material: {mat_name}")
                removed_mat_count += 1
        log.debug(f"Removed {removed_mat_count} DICOM materials")
        
        # 3. Remove preview images
        preview_img = bpy.data.images.get("DICOM_Preview")
        if preview_img:
            bpy.data.images.remove(preview_img)
            log.debug("Removed preview image")
        
        # 4. Reset all scene properties
        scn = context.scene
        scn.dicom_volume_data_path = ""
        scn.dicom_volume_spacing = ""
        scn.dicom_volume_unique_id = ""
        scn.dicom_volume_hu_min = -1024.0
        scn.dicom_volume_hu_max = 3000.0
        scn.dicom_preview_slice_index = 0
        scn.dicom_preview_slice_count = 0
        scn.dicom_preview_series_index = -1
        scn.dicom_series_data = ""
        scn.dicom_active_material_preset = "ct_standard"
        scn.dicom_tissue_alphas.clear()
        log.info("Reset all scene properties")
        
        # 5. Clean up temp files
        import tempfile
        import glob
        temp_dir = tempfile.gettempdir()
        for pattern in ["ct_volume_*.vdb", "ct_volume_*.npy", "ct_*_*.vdb", "dicom_preview_*.png"]:
            for filepath in glob.glob(os.path.join(temp_dir, pattern)):
                try:
                    os.remove(filepath)
                    log.debug(f"Removed temp file: {os.path.basename(filepath)}")
                except:
                    pass
        
        log.debug("Cleanup complete - ready for fresh patient load")
        log.debug("=" * 60)
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


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
            item.number = series['number'] if series['number'] is not None else 0
            item.instance_count = series['instance_count']
            item.rows = series['rows']
            item.cols = series['cols']
            item.window_center = series['window_center'] if series['window_center'] is not None else 0.0
            item.window_width = series['window_width'] if series['window_width'] is not None else 0.0
        
        context.scene.dicom_series_data = json.dumps(series_list)
        
        self.report({'INFO'}, f"Found {len(series_list)} series with {len(files)} total files")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class IMPORT_OT_dicom_import_series(Operator):
    """Import selected DICOM series as 3D volume"""
    bl_idname = "import.dicom_import_series"
    bl_label = "Import Series as Volume"
    bl_options = {'REGISTER', 'UNDO'}
    
    series_index: IntProperty()

    def execute(self, context):
        series_list = json.loads(context.scene.dicom_series_data)
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
            create_volume(slices)
            
            self.report({'INFO'}, f"Imported {len(slices)} slices as volume")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create volume: {e}")
            return {'CANCELLED'}


class IMPORT_OT_dicom_reload_patient(Operator):
    """Reload the current patient (clears volumes and reimports with current settings)"""
    bl_idname = "import.dicom_reload_patient"
    bl_label = "Reload Patient"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scn = context.scene
        
        if not scn.dicom_patient_data:
            self.report({'WARNING'}, "No patient loaded")
            return {'CANCELLED'}
        
        try:
            patient = Patient.from_json(scn.dicom_patient_data)
            
            # Clear all volumes and meshes
            log.info("Clearing all volumes and meshes for reload...")
            for obj in list(bpy.data.objects):
                if obj.type in {'VOLUME', 'MESH'} and ('_Volume_' in obj.name or '_Bone_' in obj.name or '_Skin_' in obj.name):
                    bpy.data.objects.remove(obj, do_unlink=True)
            
            # Clear materials
            for mat in list(bpy.data.materials):
                if '_Material' in mat.name:
                    bpy.data.materials.remove(mat, do_unlink=True)
            
            # Reset all series states
            for series in patient.series:
                series.is_loaded = False
                series.is_visible = False
                series.tissue_volumes = {}
            
            # Reset tool to NONE to force re-visualization
            scn.dicom_active_tool = 'NONE'
            
            # Save updated patient data
            scn.dicom_patient_data = patient.to_json()
            
            self.report({'INFO'}, "Patient cleared. Select Visualization tool to reload with current settings.")
            log.info("Patient reload complete - ready for re-visualization")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to reload: {e}")
            log.error(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
