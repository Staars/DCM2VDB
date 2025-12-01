"""Operator classes for DICOM import"""

import bpy
import os
from bpy.props import StringProperty, IntProperty
from bpy.types import Operator

from .dicom_io import PYDICOM_AVAILABLE, gather_dicom_files, organize_by_series, load_slice, log, load_patient_from_folder
from .volume import create_volume
from .preview import load_and_display_slice
from .patient import Patient

import numpy as np
import os

# Global storage for preview collections
preview_collections = {}

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
            log(f"Primary series loaded:")
            for series in patient.series:
                log(f"  - {series.series_description} ({series.modality}): {series.slice_count} slices")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load patient: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
    def _cleanup_all(self, context):
        """Complete cleanup: remove all DICOM objects, materials, images, and reset properties"""
        log("=" * 60)
        log("COMPLETE CLEANUP - Removing all DICOM data")
        log("=" * 60)
        
        # 1. Remove all DICOM objects (modality-aware naming)
        removed_count = 0
        for obj in list(bpy.data.objects):
            # Check if it's a DICOM object (CT_, MR_, or other modality prefixes)
            if any(pattern in obj.name for pattern in ['_Volume_S', '_Volume', '_Bone_S', '_Bone', '_Mesh_S', '_Mesh', 'DEBUG_Pyramid']):
                obj_name = obj.name
                bpy.data.objects.remove(obj, do_unlink=True)
                log(f"Removed object: {obj_name}")
                removed_count += 1
        log(f"Removed {removed_count} DICOM objects")
        
        # 2. Remove all DICOM materials (modality-aware)
        removed_mat_count = 0
        for mat in list(bpy.data.materials):
            if mat.name.endswith("_Volume_Material") or mat.name.endswith("_Bone_Material") or mat.name.endswith("_Mesh_Material"):
                mat_name = mat.name  # Save name before removing
                bpy.data.materials.remove(mat)
                log(f"Removed material: {mat_name}")
                removed_mat_count += 1
        log(f"Removed {removed_mat_count} DICOM materials")
        
        # 3. Remove preview images
        preview_img = bpy.data.images.get("DICOM_Preview")
        if preview_img:
            bpy.data.images.remove(preview_img)
            log("Removed preview image")
        
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
        log("Reset all scene properties")
        
        # 5. Clean up temp files
        import tempfile
        temp_dir = tempfile.gettempdir()
        import glob
        for pattern in ["ct_volume_*.vdb", "ct_volume_*.npy", "ct_*_*.vdb", "dicom_preview_*.png"]:
            for filepath in glob.glob(os.path.join(temp_dir, pattern)):
                try:
                    os.remove(filepath)
                    log(f"Removed temp file: {os.path.basename(filepath)}")
                except:
                    pass
        
        log("Cleanup complete - ready for fresh patient load")
        log("=" * 60)
    
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
        
        context.scene.dicom_series_data = str(series_list)
        
        self.report({'INFO'}, f"Found {len(series_list)} series with {len(files)} total files")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class IMPORT_OT_dicom_preview(Operator):
    """Load DICOM series for Image Editor preview"""
    bl_idname = "import.dicom_preview"
    bl_label = "Preview Series"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        # Get series data from stored index
        series_list = eval(context.scene.dicom_series_data)
        series_idx = context.scene.dicom_preview_series_index
        
        if series_idx >= len(series_list):
            self.report({'ERROR'}, "Invalid series index")
            return {'CANCELLED'}
        
        series = series_list[series_idx]
        
        # Store preview info in scene
        context.scene.dicom_preview_slice_index = 0
        context.scene.dicom_preview_slice_count = len(series['files'])
        
        # Load first slice
        try:
            load_and_display_slice(context, series['files'][0], series)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load slice: {e}")
            return {'CANCELLED'}
        
        # Try to load in existing Image Editor (inline the code)
        image_editor_found = False
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    for space in area.spaces:
                        if space.type == 'IMAGE_EDITOR':
                            space.image = bpy.data.images.get("DICOM_Preview")
                            image_editor_found = True
                            break
        
        if image_editor_found:
            self.report({'INFO'}, f"Preview loaded in Image Editor with {len(series['files'])} slices. Use mouse wheel to scroll.")
        else:
            self.report({'WARNING'}, f"Preview ready with {len(series['files'])} slices. Open an Image Editor to view.")
        
        return {'FINISHED'}

class IMPORT_OT_dicom_preview_popup(Operator):
    """Show DICOM preview matrix in popup menu"""
    bl_idname = "import.dicom_preview_popup"
    bl_label = "DICOM Series Preview"
    bl_options = {'INTERNAL'}
    
    series_index: IntProperty()
    
    def execute(self, context):
        return {'FINISHED'}
    
    def invoke(self, context, event):
        global preview_collections
        import bpy.utils.previews
        from PIL import Image
        import tempfile
        
        # Load preview images for matrix view
        series_list = eval(context.scene.dicom_series_data)
        if self.series_index >= len(series_list):
            return {'CANCELLED'}
        
        series = series_list[self.series_index]
        
        # Store preview info
        context.scene.dicom_preview_series_index = self.series_index
        context.scene.dicom_preview_slice_count = len(series['files'])
        
        # Clear old preview collection
        if "main" in preview_collections:
            bpy.utils.previews.remove(preview_collections["main"])
        
        # Create new preview collection
        pcoll = bpy.utils.previews.new()
        
        # Load up to 100 preview images (10x10 grid)
        max_previews = 100
        step = max(1, len(series['files']) // max_previews)
        
        for i, idx in enumerate(range(0, len(series['files']), step)):
            if i >= max_previews:
                break
            
            try:
                slice_data = load_slice(series['files'][idx])
                if slice_data is None:
                    continue
                pixels = slice_data["pixels"]
                
                # Apply window/level
                wc = series.get('window_center') or slice_data.get('window_center')
                ww = series.get('window_width') or slice_data.get('window_width')
                
                if wc is not None and ww is not None and ww > 0:
                    low = wc - ww / 2
                    high = wc + ww / 2
                    pixels_windowed = np.clip(pixels, low, high)
                    normalized = ((pixels_windowed - low) / ww * 255).astype(np.uint8)
                else:
                    pmin, pmax = np.percentile(pixels, [1, 99])
                    if pmax > pmin:
                        normalized = np.clip((pixels - pmin) / (pmax - pmin) * 255, 0, 255).astype(np.uint8)
                    else:
                        normalized = np.zeros_like(pixels, dtype=np.uint8)
                
                # Save as temporary image
                temp_path = os.path.join(tempfile.gettempdir(), f"dicom_preview_{i}.png")
                img = Image.fromarray(normalized, mode='L')
                img = img.resize((128, 128), Image.Resampling.LANCZOS)
                img.save(temp_path)
                
                pcoll.load(f"slice_{i}", temp_path, 'IMAGE')
            except Exception as e:
                log(f"Failed to create preview {i}: {e}")
        
        preview_collections["main"] = pcoll
        
        # Show popup
        return context.window_manager.invoke_popup(self, width=600)
    
    def draw(self, context):
        layout = self.layout
        
        # Show series info
        series_list = eval(context.scene.dicom_series_data)
        if context.scene.dicom_preview_series_index < len(series_list):
            series = series_list[context.scene.dicom_preview_series_index]
            row = layout.row()
            row.label(text=f"{series['description']} - {series['instance_count']} slices", icon='IMAGE_DATA')
            
            # Button to open in Image Editor
            row.operator(IMPORT_OT_dicom_preview.bl_idname, text="Open in Image Editor", icon='IMAGE')
        
        layout.separator()
        
        # Draw 10x10 grid of preview images
        global preview_collections
        if "main" in preview_collections:
            pcoll = preview_collections["main"]
            
            # Create grid
            grid = layout.grid_flow(row_major=True, columns=10, align=True)
            
            for i in range(100):
                key = f"slice_{i}"
                if key in pcoll:
                    preview = pcoll[key]
                    grid.template_icon(icon_value=preview.icon_id, scale=2.0)
                else:
                    break
        else:
            layout.label(text="No preview available", icon='ERROR')

class IMPORT_OT_dicom_open_in_editor(Operator):
    """Open DICOM preview in Image Editor"""
    bl_idname = "import.dicom_open_in_editor"
    bl_label = "Open in Image Editor"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        # Only look for existing Image Editor areas, never switch automatically
        image_editor_found = False
        
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    # Found one, set the image
                    for space in area.spaces:
                        if space.type == 'IMAGE_EDITOR':
                            space.image = bpy.data.images.get("DICOM_Preview")
                            image_editor_found = True
                            self.report({'INFO'}, "Preview loaded in Image Editor")
                            break
        
        if not image_editor_found:
            self.report({'WARNING'}, "No Image Editor found. Please manually open an Image Editor area first.")
        
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
            slice_data = load_slice(path)
            if slice_data is not None:
                slices.append(slice_data)
            else:
                log(f"Skipped slice (no pixel data): {path}")
        
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
                log(f"Skipped slice (no pixel data): {path}")
        
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
            self._calculate_all_volumes(context, series)
            
            # Save updated measurements
            context.scene.dicom_patient_data = patient.to_json()
            
            self.report({'INFO'}, f"Visualized: {series.series_description} ({len(slices)} slices)")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create volume: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
    def _calculate_all_volumes(self, context, series):
        """Calculate all tissue volumes automatically for a specific series"""
        from .measurements import calculate_tissue_volume
        
        scn = context.scene
        
        # Check if volume data is available
        if not scn.dicom_volume_data_path or not os.path.exists(scn.dicom_volume_data_path):
            return
        
        try:
            # Load volume data
            vol_array = np.load(scn.dicom_volume_data_path)
            
            # Parse spacing
            spacing = eval(scn.dicom_volume_spacing)  # [X, Y, Z] in mm
            pixel_spacing = (spacing[1], spacing[0])  # (row, col) = (Y, X)
            slice_thickness = spacing[2]  # Z
            
            # Calculate all tissue volumes and store in series (dynamic from preset)
            from .properties import get_tissue_thresholds_from_preset
            thresholds = get_tissue_thresholds_from_preset(scn.dicom_active_material_preset)
            
            # Clear previous measurements
            series.tissue_volumes = {}
            
            # Calculate volume for each tissue defined in preset
            for tissue_name, tissue_range in thresholds.items():
                volume_ml = calculate_tissue_volume(
                    vol_array,
                    tissue_range.get('min', 0),
                    tissue_range.get('max', 0),
                    pixel_spacing, slice_thickness
                )
                if volume_ml > 0:
                    series.tissue_volumes[tissue_name] = volume_ml
            
            log(f"Tissue volumes calculated for series {series.series_number}")
            
        except Exception as e:
            log(f"Failed to calculate volumes: {e}")

class IMPORT_OT_dicom_preview_series(Operator):
    """Preview series in Image Editor (2D slice viewer)"""
    bl_idname = "import.dicom_preview_series"
    bl_label = "Preview in Image Editor"
    bl_options = {'REGISTER'}
    
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
        
        # Clear old DICOM_Preview image to force fresh load
        if "DICOM_Preview" in bpy.data.images:
            bpy.data.images.remove(bpy.data.images["DICOM_Preview"])
            log("Cleared old DICOM_Preview image")
        
        # Store preview info in scene (for scrolling and slice navigation)
        context.scene.dicom_preview_slice_index = 0
        context.scene.dicom_preview_slice_count = len(file_paths)
        context.scene.dicom_preview_series_index = 0  # Always use index 0 for single series
        
        # Store series data in old format for compatibility with preview_slice and scroll operators
        series_list = [{
            'files': file_paths,
            'window_center': series.window_center,
            'window_width': series.window_width,
        }]
        context.scene.dicom_series_data = str(series_list)
        
        # Load first slice
        try:
            load_and_display_slice(context, file_paths[0], series_list[0])
            self.report({'INFO'}, f"Preview ready: {series.series_description} ({len(file_paths)} slices)")
            self.report({'INFO'}, "Open an Image Editor to view. Use mouse wheel to scroll.")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load preview: {e}")
            return {'CANCELLED'}

class IMPORT_OT_dicom_set_tool(Operator):
    """Set the active DICOM tool"""
    bl_idname = "import.dicom_set_tool"
    bl_label = "Set Tool"
    bl_options = {'REGISTER', 'UNDO'}
    
    tool: StringProperty()
    
    def execute(self, context):
        context.scene.dicom_active_tool = self.tool
        
        # If switching to VISUALIZATION tool, auto-visualize all selected series
        if self.tool == 'VISUALIZATION' and context.scene.dicom_patient_data:
            try:
                patient = Patient.from_json(context.scene.dicom_patient_data)
                
                # Clear all previously loaded volumes and meshes
                self._clear_all_volumes(context)
                
                # Reset all series loaded states
                for s in patient.series:
                    s.is_loaded = False
                    s.is_visible = False
                
                selected_series = [s for s in patient.series if s.is_selected]
                
                log(f"Total series: {len(patient.series)}, Selected: {len(selected_series)}")
                for s in patient.series:
                    log(f"  Series {s.series_number}: selected={s.is_selected}, loaded={s.is_loaded}")
                
                if selected_series:
                    self.report({'INFO'}, f"Auto-visualizing {len(selected_series)} selected series...")
                    
                    for series in selected_series:
                        # Build absolute file paths
                        file_paths = [os.path.join(patient.dicom_root_path, rel_path) 
                                     for rel_path in series.file_paths]
                        
                        # Load slices
                        from .dicom_io import load_slice
                        slices = []
                        for path in file_paths:
                            slice_data = load_slice(path)
                            if slice_data is not None:
                                slices.append(slice_data)
                        
                        if len(slices) >= 2:
                            # Create volume
                            from .volume import create_volume
                            vol_obj = create_volume(slices, series_number=series.series_number)
                            
                            # Update series state
                            series.is_loaded = True
                            series.is_visible = True
                            
                            # Store volume object reference
                            patient.volume_objects[series.series_instance_uid] = vol_obj.name
                            
                            # Calculate measurements
                            self._calculate_volumes_for_series(context, series)
                            
                            log(f"Auto-visualized series {series.series_number}")
                    
                    # Save updated patient data
                    context.scene.dicom_patient_data = patient.to_json()
                    self.report({'INFO'}, f"Visualized {len(selected_series)} series")
                else:
                    self.report({'INFO'}, "All selected series already loaded")
            except Exception as e:
                log(f"Auto-visualization error: {e}")
                import traceback
                traceback.print_exc()
        
        self.report({'INFO'}, f"Switched to {self.tool} tool")
        return {'FINISHED'}
    
    def _clear_all_volumes(self, context):
        """Clear all DICOM volumes and meshes from the scene"""
        import bpy
        
        # Remove all objects with DICOM naming patterns
        for obj in list(bpy.data.objects):
            if any(pattern in obj.name for pattern in ['_Volume_', '_Mesh_', 'DICOM_']):
                bpy.data.objects.remove(obj, do_unlink=True)
                log(f"Removed object: {obj.name}")
        
        # Remove volume materials
        for mat in list(bpy.data.materials):
            if mat.name.endswith('_Volume_Material'):
                bpy.data.materials.remove(mat)
                log(f"Removed material: {mat.name}")
        
        log("Cleared all DICOM volumes and meshes")
    
    def _calculate_volumes_for_series(self, context, series):
        """Calculate tissue volumes for a series"""
        from .measurements import calculate_tissue_volume
        
        scn = context.scene
        if not scn.dicom_volume_data_path or not os.path.exists(scn.dicom_volume_data_path):
            return
        
        try:
            vol_array = np.load(scn.dicom_volume_data_path)
            spacing = eval(scn.dicom_volume_spacing)
            pixel_spacing = (spacing[1], spacing[0])
            slice_thickness = spacing[2]
            
            # Calculate all tissue volumes and store in series (dynamic from preset)
            from .properties import get_tissue_thresholds_from_preset
            thresholds = get_tissue_thresholds_from_preset(scn.dicom_active_material_preset)
            
            # Clear previous measurements
            series.tissue_volumes = {}
            
            # Calculate volume for each tissue defined in preset
            for tissue_name, tissue_range in thresholds.items():
                volume_ml = calculate_tissue_volume(
                    vol_array,
                    tissue_range.get('min', 0),
                    tissue_range.get('max', 0),
                    pixel_spacing, slice_thickness
                )
                if volume_ml > 0:
                    series.tissue_volumes[tissue_name] = volume_ml
        except Exception as e:
            log(f"Volume calculation error: {e}")

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
        from .measurements import calculate_tissue_volume
        
        scn = context.scene
        
        # Check if volume data is available
        if not scn.dicom_volume_data_path or not os.path.exists(scn.dicom_volume_data_path):
            self.report({'ERROR'}, "No volume data available. Please load a DICOM series first.")
            return {'CANCELLED'}
        
        try:
            # Load volume data
            vol_array = np.load(scn.dicom_volume_data_path)
            
            # Parse spacing
            spacing = eval(scn.dicom_volume_spacing)  # [X, Y, Z] in mm
            pixel_spacing = (spacing[1], spacing[0])  # (row, col) = (Y, X)
            slice_thickness = spacing[2]  # Z
            
            # Get HU thresholds based on tissue type (from preset)
            from .properties import get_tissue_thresholds_from_preset
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
            
            # Note: Result is calculated but not stored anywhere now (legacy operator)
            # Measurements are now stored per-series in patient data
            
            self.report({'INFO'}, f"{self.tissue_type.title()}: {volume_ml:.2f} mL")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to calculate volume: {e}")
            log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

classes = (
    IMPORT_OT_dicom_load_patient,
    IMPORT_OT_dicom_visualize_series,
    IMPORT_OT_dicom_preview_series,
    IMPORT_OT_dicom_set_tool,
    IMPORT_OT_dicom_toggle_series_selection,
    IMPORT_OT_dicom_toggle_series_visibility,
    IMPORT_OT_dicom_toggle_display_mode,
    IMPORT_OT_dicom_calculate_volume,
    IMPORT_OT_dicom_scan,
    IMPORT_OT_dicom_preview,
    IMPORT_OT_dicom_preview_popup,
    IMPORT_OT_dicom_open_in_editor,
    IMPORT_OT_dicom_preview_slice,
    IMAGE_OT_dicom_scroll,
    IMPORT_OT_dicom_import_series,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    # Clean up preview collections
    global preview_collections
    for pcoll in preview_collections.values():
        bpy.utils.previews.remove(pcoll)
    preview_collections.clear()
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)