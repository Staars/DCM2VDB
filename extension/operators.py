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
    debug_mode: bpy.props.BoolProperty(
        name="Debug Mode",
        description="Also create a test pyramid for comparison",
        default=False
    )

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
            
            # Create debug pyramid if requested
            if self.debug_mode or context.scene.dicom_debug_pyramid:
                self.create_debug_pyramid(context)
            
            self.report({'INFO'}, f"Imported {len(slices)} slices as volume")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create volume: {e}")
            return {'CANCELLED'}
    
    def create_debug_pyramid(self, context):
        """Create a simple pyramid volume for debugging"""
        import openvdb as vdb
        import tempfile
        
        log("Creating debug pyramid...")
        
        # Create a 100x100x100 grid with pyramid shape
        size = 100
        pyramid = np.zeros((size, size, size), dtype=np.float32)
        
        for z in range(size):
            # Pyramid gets narrower as we go up
            width = int(size * (1.0 - z/size))
            if width > 0:
                center = size // 2
                start = center - width // 2
                end = center + width // 2
                pyramid[z, start:end, start:end] = float(z) / size * 1000.0  # Gradient from 0 to 1000
        
        log(f"Pyramid value range: {pyramid.min():.1f} to {pyramid.max():.1f}")
        
        # Transpose to (x, y, z)
        pyramid_transposed = np.transpose(pyramid, (2, 1, 0))
        
        # Create VDB
        temp_vdb = os.path.join(tempfile.gettempdir(), "debug_pyramid.vdb")
        grid = vdb.FloatGrid()
        grid.copyFromArray(pyramid_transposed)
        grid.name = "density"
        
        # 1mm voxels
        transform_matrix = [
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1]
        ]
        grid.transform = vdb.createLinearTransform(transform_matrix)
        vdb.write(temp_vdb, grids=[grid])
        
        # Import
        bpy.ops.object.volume_import(filepath=temp_vdb, files=[{"name": "debug_pyramid.vdb"}])
        pyramid_obj = context.active_object
        pyramid_obj.name = "DEBUG_Pyramid"
        pyramid_obj.location = (0.15, 0, 0)  # Offset to the right
        
        # Simple material
        mat = bpy.data.materials.new("DEBUG_Pyramid_Material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        out = nodes.new("ShaderNodeOutputMaterial")
        prin = nodes.new("ShaderNodeVolumePrincipled")
        vol_info = nodes.new("ShaderNodeVolumeInfo")
        
        # Simple ramp
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
        ramp.color_ramp.elements[1].color = (1, 0, 0, 1)  # Red for visibility
        
        # Scale node
        math = nodes.new("ShaderNodeMath")
        math.operation = 'MULTIPLY'
        math.inputs[1].default_value = 0.01
        
        mat.node_tree.links.new(vol_info.outputs["Density"], ramp.inputs["Fac"])
        mat.node_tree.links.new(ramp.outputs["Color"], prin.inputs["Color"])
        mat.node_tree.links.new(vol_info.outputs["Density"], math.inputs[0])
        mat.node_tree.links.new(math.outputs["Value"], prin.inputs["Density"])
        mat.node_tree.links.new(prin.outputs["Volume"], out.inputs["Volume"])
        
        pyramid_obj.data.materials.append(mat)
        pyramid_obj.data.display.density = 0.01
        
        log("Debug pyramid created - should appear as red pyramid next to DICOM volume")

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
            try:
                slices.append(load_slice(path))
            except Exception as e:
                log(f"Failed to load {path}: {e}")
        
        wm.progress_end()
        
        if len(slices) < 2:
            self.report({'ERROR'}, "Need at least 2 valid slices")
            return {'CANCELLED'}
        
        try:
            # Create volume
            vol_obj = create_volume(slices)
            
            # Update series state
            series.is_loaded = True
            series.is_visible = True
            
            # Store volume object reference
            patient.volume_objects[self.series_uid] = vol_obj.name
            
            # Save updated patient data
            context.scene.dicom_patient_data = patient.to_json()
            
            self.report({'INFO'}, f"Visualized: {series.series_description} ({len(slices)} slices)")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create volume: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

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
        
        # Store preview info in scene (for scrolling)
        context.scene.dicom_preview_slice_index = 0
        context.scene.dicom_preview_slice_count = len(file_paths)
        
        # Create a temporary series dict for compatibility with old preview system
        series_dict = {
            'files': file_paths,
            'window_center': series.window_center,
            'window_width': series.window_width,
        }
        
        # Load first slice
        try:
            load_and_display_slice(context, file_paths[0], series_dict)
            self.report({'INFO'}, f"Preview ready: {series.series_description} ({len(file_paths)} slices)")
            self.report({'INFO'}, "Open an Image Editor to view. Use mouse wheel to scroll.")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load preview: {e}")
            return {'CANCELLED'}

classes = (
    IMPORT_OT_dicom_load_patient,
    IMPORT_OT_dicom_visualize_series,
    IMPORT_OT_dicom_preview_series,
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