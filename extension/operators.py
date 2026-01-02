"""Operator classes for DICOM import"""

import bpy
import os
from bpy.props import StringProperty, IntProperty
from bpy.types import Operator

from .dicom_io import PYDICOM_AVAILABLE, gather_dicom_files, organize_by_series, load_slice, load_patient_from_folder
from .volume import create_volume
from .preview import load_and_display_slice
from .patient import Patient
from .utils import SimpleLogger

import numpy as np

log = SimpleLogger()

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
        temp_dir = tempfile.gettempdir()
        import glob
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

class IMPORT_OT_dicom_preview(Operator):
    """Load DICOM series for Image Editor preview"""
    bl_idname = "import.dicom_preview"
    bl_label = "Preview Series"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        import json
        
        # Get series data from stored index
        series_list = json.loads(context.scene.dicom_series_data)
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
        
        # Try to load in existing Image Editor
        from .ui_utils import set_image_in_all_editors
        
        image_editor_found = set_image_in_all_editors(
            context, 
            bpy.data.images.get("DICOM_Preview")
        )
        
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
        import json
        
        # Load preview images for matrix view
        series_list = json.loads(context.scene.dicom_series_data)
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
        
        from .constants import MAX_PREVIEW_IMAGES
        
        # Load up to MAX_PREVIEW_IMAGES preview images (10x10 grid)
        max_previews = MAX_PREVIEW_IMAGES
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
                from .constants import PREVIEW_THUMBNAIL_SIZE
                
                temp_path = os.path.join(tempfile.gettempdir(), f"dicom_preview_{i}.png")
                img = Image.fromarray(normalized, mode='L')
                img = img.resize((PREVIEW_THUMBNAIL_SIZE, PREVIEW_THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
                img.save(temp_path)
                
                pcoll.load(f"slice_{i}", temp_path, 'IMAGE')
            except Exception as e:
                log.error(f"Failed to create preview {i}: {e}")
        
        preview_collections["main"] = pcoll
        
        from .constants import PREVIEW_POPUP_WIDTH
        
        # Show popup
        return context.window_manager.invoke_popup(self, width=PREVIEW_POPUP_WIDTH)
    
    def draw(self, context):
        layout = self.layout
        import json
        
        # Show series info
        series_list = json.loads(context.scene.dicom_series_data)
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
            
            from .constants import MAX_PREVIEW_IMAGES
            
            for i in range(MAX_PREVIEW_IMAGES):
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
        from .ui_utils import set_image_in_all_editors
        
        # Only look for existing Image Editor areas, never switch automatically
        image_editor_found = set_image_in_all_editors(
            context,
            bpy.data.images.get("DICOM_Preview")
        )
        
        if image_editor_found:
            self.report({'INFO'}, "Preview loaded in Image Editor")
        else:
            self.report({'WARNING'}, "No Image Editor found. Please manually open an Image Editor area first.")
        
        return {'FINISHED'}

class IMPORT_OT_dicom_preview_slice(Operator):
    """Load a specific slice for preview"""
    bl_idname = "import.dicom_preview_slice"
    bl_label = "Load Slice"
    bl_options = {'INTERNAL'}
    
    slice_index: IntProperty()
    
    def execute(self, context):
        import json
        
        series_list = json.loads(context.scene.dicom_series_data)
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

class IMAGE_OT_dicom_set_cursor_3d(Operator):
    """Set 3D cursor from 2D image double-click"""
    bl_idname = "image.dicom_set_cursor_3d"
    bl_label = "Set 3D Cursor from DICOM Image"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return (context.area and context.area.type == 'IMAGE_EDITOR' and
                context.scene.dicom_preview_slice_count > 0 and
                "DICOM_Preview" in bpy.data.images)
    
    def invoke(self, context, event):
        self.set_cursor_from_mouse(context, event)
        return {'FINISHED'}
    
    def set_cursor_from_mouse(self, context, event):
        import json
        
        # Get mouse position in image space
        region = context.region
        space = context.space_data
        
        if not space or space.type != 'IMAGE_EDITOR':
            self.report({'ERROR'}, "Not in Image Editor")
            return {'CANCELLED'}
        
        img = bpy.data.images.get("DICOM_Preview")
        if not img:
            self.report({'ERROR'}, "No DICOM preview image")
            return {'CANCELLED'}
        
        # Get mouse coordinates in region
        mouse_x = event.mouse_region_x
        mouse_y = event.mouse_region_y
        
        # Image dimensions
        img_width, img_height = img.size
        
        # Convert region coordinates to normalized texture coordinates (0-1)
        view2d = region.view2d
        texture_x, texture_y = view2d.region_to_view(mouse_x, mouse_y)
        
        log.debug(f"Mouse region: ({mouse_x}, {mouse_y})")
        log.debug(f"Texture coords (0-1): ({texture_x:.4f}, {texture_y:.4f})")
        log.debug(f"Image size: ({img_width}, {img_height})")
        
        # Check if within image bounds (0-1 range)
        if texture_x < 0 or texture_x > 1 or texture_y < 0 or texture_y > 1:
            log.debug("Click outside image bounds - ignoring")
            return {'CANCELLED'}
        
        # Convert to pixel coordinates
        pixel_x = int(texture_x * img_width)
        pixel_y = int(texture_y * img_height)
        
        # Clamp to image bounds
        pixel_x = max(0, min(pixel_x, img_width - 1))
        pixel_y = max(0, min(pixel_y, img_height - 1))
        
        log.debug(f"Pixel coords: ({pixel_x}, {pixel_y})")
        
        # Get current slice information
        slice_index = context.scene.dicom_preview_slice_index
        
        # Load patient and series data
        if not context.scene.dicom_patient_data:
            self.report({'WARNING'}, "No patient data loaded")
            return {'CANCELLED'}
        
        try:
            from .patient import Patient
            patient = Patient.from_json(context.scene.dicom_patient_data)
            
            # Get the series being previewed
            series_list = json.loads(context.scene.dicom_series_data)
            if not series_list:
                return {'CANCELLED'}
            
            series_data = series_list[0]
            file_paths = series_data['files']
            
            if slice_index >= len(file_paths):
                return {'CANCELLED'}
            
            # Load the current slice to get spatial information
            from .dicom_io import load_slice
            slice_data = load_slice(file_paths[slice_index])
            
            if not slice_data:
                self.report({'WARNING'}, "Failed to load slice data")
                return {'CANCELLED'}
            
            # Get DICOM spatial information
            position = slice_data.get('position', [0, 0, 0])  # ImagePositionPatient
            orientation = slice_data.get('orientation', [1, 0, 0, 0, 1, 0])  # ImageOrientationPatient
            pixel_spacing = slice_data.get('pixel_spacing', (1.0, 1.0))  # (row, col) spacing in mm
            
            log.debug(f"Slice {slice_index + 1}/{len(file_paths)}")
            log.debug(f"ImagePositionPatient: {position}")
            log.debug(f"ImageOrientationPatient: {orientation}")
            log.debug(f"PixelSpacing: {pixel_spacing}")
            
            # Convert 2D pixel coordinates to 3D patient coordinates
            # DICOM coordinate system:
            # - ImagePositionPatient: position of top-left pixel (0,0)
            # - ImageOrientationPatient: [row_x, row_y, row_z, col_x, col_y, col_z]
            # - PixelSpacing: [row_spacing, col_spacing]
            
            import numpy as np
            
            # Row and column direction vectors
            row_vec = np.array(orientation[:3])  # Direction along rows (Y in image)
            col_vec = np.array(orientation[3:])  # Direction along columns (X in image)
            
            log.debug(f"Row vector: {row_vec}")
            log.debug(f"Col vector: {col_vec}")
            
            # Image is flipped in preview, so adjust Y coordinate
            pixel_y_flipped = img_height - 1 - pixel_y
            
            log.debug(f"Pixel Y flipped: {pixel_y} -> {pixel_y_flipped}")
            
            # Calculate 3D position
            # Position = ImagePosition + (col * col_spacing * col_vec) + (row * row_spacing * row_vec)
            pos_3d = np.array(position) + \
                     (pixel_x * pixel_spacing[1] * col_vec) + \
                     (pixel_y_flipped * pixel_spacing[0] * row_vec)
            
            log.debug(f"3D position (DICOM): {pos_3d}")
            
            # Find the volume object for THIS series
            # Get series number from the DICOM file
            series_number = slice_data["ds"].SeriesNumber if hasattr(slice_data["ds"], "SeriesNumber") else None
            
            volume_obj = None
            if series_number:
                # Try to find volume with matching series number
                for obj in bpy.data.objects:
                    if obj.type == 'VOLUME' and f'_S{series_number}' in obj.name:
                        volume_obj = obj
                        break
            
            # Fallback: use any volume if series number not found
            if not volume_obj:
                for obj in bpy.data.objects:
                    if obj.type == 'VOLUME' and '_Volume_' in obj.name:
                        volume_obj = obj
                        break
            
            if volume_obj:
                # Use the volume's actual world location as reference (this is the pivot = top-left)
                vol_loc = volume_obj.location
                
                # Image editor uses Cartesian: (0,0) = bottom-left, (0,height) = top-left
                # DICOM ImagePositionPatient = top-left = (0, height) in image coords
                # So we need to flip Y: pixel_y_from_top = height - pixel_y
                pixel_y_from_top = img_height - pixel_y
                
                # Calculate image height in mm
                img_height_mm = img_height * pixel_spacing[0]
                
                # Calculate DICOM position for this pixel
                # Starting from ImagePositionPatient (top-left), move by pixel offsets
                dicom_x = position[0] + pixel_x * pixel_spacing[1]
                dicom_y = position[1] + pixel_y_from_top * pixel_spacing[0]
                
                # Calculate offset from pivot (ImagePositionPatient)
                relative_x = dicom_x - position[0]
                relative_y = dicom_y - position[1]
                
                from .constants import MM_TO_METERS
                
                # Convert to meters
                offset_x = relative_x * MM_TO_METERS
                offset_y = relative_y * MM_TO_METERS
                
                # Volume Y length in meters
                volume_y_length = img_height_mm * MM_TO_METERS
                
                from .constants import MM_TO_METERS
                
                blender_pos = (
                    vol_loc[0] - offset_x,
                    vol_loc[1] - (volume_y_length - offset_y) + volume_y_length,
                    pos_3d[2] * MM_TO_METERS
                )
                
                log.debug("===== FINAL COORDINATES =====")
                log.debug(f"Image click: pixel ({pixel_x}, {pixel_y}) from bottom")
                log.debug(f"Pixel from top: {pixel_y_from_top}")
                log.debug(f"Volume location (pivot): {vol_loc}")
                log.debug(f"DICOM position: ({dicom_x:.2f}, {dicom_y:.2f})")
                log.debug(f"Offset from pivot: ({relative_x:.2f}, {relative_y:.2f}) mm")
                log.debug(f"Blender offset: ({offset_x:.4f}, {offset_y:.4f}) m")
                log.debug(f"3D position (before centering): {blender_pos}")
            else:
                log.error("No volume object found!")
                return {'CANCELLED'}
            
            # Apply centering offset ONLY to Z if volumes were centered at origin
            # X and Y are already correct because they're calculated from vol_loc which is already centered
            # Z needs adjustment because it's calculated from DICOM position directly
            if context.scene.dicom_centering_offset:
                import json
                try:
                    offset_data = json.loads(context.scene.dicom_centering_offset)
                    centering_offset_z = offset_data['z']
                    
                    log.debug("=" * 60)
                    log.debug("APPLYING CENTERING OFFSET TO CURSOR (Z only)")
                    log.debug(f"Centering offset Z: {centering_offset_z}")
                    
                    # Apply offset only to Z
                    blender_pos = (
                        blender_pos[0],  # X unchanged
                        blender_pos[1],  # Y unchanged
                        blender_pos[2] - centering_offset_z  # Z adjusted
                    )
                    
                    log.debug(f"3D position (after centering): {blender_pos}")
                    log.debug("=" * 60)
                except Exception as e:
                    log.error(f"Failed to apply centering offset: {e}")
            
            from .ui_utils import refresh_all_3d_views
            
            # Set 3D cursor
            context.scene.cursor.location = blender_pos
            
            self.report({'INFO'}, 
                f"3D Cursor set to ({blender_pos[0]:.3f}, {blender_pos[1]:.3f}, {blender_pos[2]:.3f})")
            
            # Redraw 3D views
            refresh_all_3d_views(context)
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to set 3D cursor: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

class IMAGE_OT_dicom_scroll(Operator):
    """Scroll through DICOM slices with mouse wheel"""
    bl_idname = "image.dicom_scroll"
    bl_label = "Scroll DICOM Slices"
    bl_options = {'INTERNAL'}
    
    direction: IntProperty(default=0)
    
    def execute(self, context):
        import json
        
        if context.scene.dicom_preview_slice_count > 0:
            current = context.scene.dicom_preview_slice_index
            new_index = current + self.direction
            new_index = max(0, min(context.scene.dicom_preview_slice_count - 1, new_index))
            
            if new_index != current:
                # Load the slice directly
                series_list = json.loads(context.scene.dicom_series_data)
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
        import json
        
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
            from .measurements import calculate_and_store_tissue_volumes
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

class IMPORT_OT_dicom_preview_series(Operator):
    """Preview series in Image Editor (2D slice viewer)"""
    bl_idname = "import.dicom_preview_series"
    bl_label = "Preview in Image Editor"
    bl_options = {'REGISTER'}
    
    series_uid: StringProperty()
    
    def execute(self, context):
        import json

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
            from .ui_utils import clear_image_from_all_editors
            
            old_img = bpy.data.images["DICOM_Preview"]
            # First, clear it from all Image Editors
            clear_image_from_all_editors(context, old_img)
            # Now remove the image
            bpy.data.images.remove(old_img)
            log.debug("Cleared old DICOM_Preview image")
        
        # Store preview info in scene (for scrolling and slice navigation)
        # IMPORTANT: Reset to slice 0 for new series
        context.scene.dicom_preview_slice_index = 0
        context.scene.dicom_preview_slice_count = len(file_paths)
        context.scene.dicom_preview_series_index = 0  # Always use index 0 for single series
        
        # Store series data in old format for compatibility with preview_slice and scroll operators
        series_list = [{
            'files': file_paths,
            'window_center': series.window_center,
            'window_width': series.window_width,
        }]
        context.scene.dicom_series_data = json.dumps(series_list)
        
        # Load first slice
        try:
            load_and_display_slice(context, file_paths[0], series_list[0])
            
            # Automatically set DICOM_Preview as active in any open Image Editor
            # and force a complete refresh
            dicom_img = bpy.data.images.get("DICOM_Preview")
            if dicom_img:
                from .ui_utils import set_image_in_all_editors
                
                image_editor_found = set_image_in_all_editors(
                    context, 
                    dicom_img, 
                    clear_first=True  # Force refresh
                )
                
                if image_editor_found:
                    self.report({'INFO'}, f"Preview loaded in Image Editor: {series.series_description} ({len(file_paths)} slices)")
                else:
                    self.report({'INFO'}, f"Preview ready: {series.series_description} ({len(file_paths)} slices)")
                    self.report({'INFO'}, "Open an Image Editor to view. Use mouse wheel to scroll.")
            else:
                self.report({'INFO'}, f"Preview ready: {series.series_description} ({len(file_paths)} slices)")
            
            self.report({'INFO'}, "Double-click on image to set 3D cursor. Use mouse wheel to scroll.")
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
        
        self.report({'INFO'}, f"Setting tool to: {self.tool}")
        self.report({'INFO'}, f"Center at origin: {context.scene.center_volume_at_origin}")
        
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
                
                log.info(f"Total series: {len(patient.series)}, Selected: {len(selected_series)}")
                for s in patient.series:
                    log.debug(f"  Series {s.series_number}: selected={s.is_selected}, loaded={s.is_loaded}")
                
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
                            from .measurements import calculate_and_store_tissue_volumes
                            calculate_and_store_tissue_volumes(context, series)
                            
                            log.info(f"Auto-visualized series {series.series_number}")
                    
                    # Save updated patient data
                    context.scene.dicom_patient_data = patient.to_json()
                    
                    # Calculate and visualize centering transform if enabled
                    if context.scene.center_volume_at_origin:
                        log.info("Center at origin is ENABLED - calling centering calculation")
                        self._calculate_centering_transform(context, patient)
                    else:
                        log.info("Center at origin is DISABLED")
                    
                    self.report({'INFO'}, f"Visualized {len(selected_series)} series")
                else:
                    log.info("No selected series to visualize (already loaded)")
                    
                    # Still calculate centering if enabled, even if volumes already exist
                    if context.scene.center_volume_at_origin:
                        log.info("Center at origin is ENABLED - calling centering calculation for existing volumes")
                        self._calculate_centering_transform(context, patient)
                    
                    self.report({'INFO'}, "All selected series already loaded")
            except Exception as e:
                log.error(f"Auto-visualization error: {e}")
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
                obj_name = obj.name  # Save name before removing
                bpy.data.objects.remove(obj, do_unlink=True)
                log.debug(f"Removed object: {obj_name}")
        
        # Remove volume materials
        for mat in list(bpy.data.materials):
            if mat.name.endswith('_Volume_Material'):
                mat_name = mat.name  # Save name before removing
                bpy.data.materials.remove(mat)
                log.debug(f"Removed material: {mat_name}")
        
        log.info("Cleared all DICOM volumes and meshes")
    
    def _calculate_centering_transform(self, context, patient):
        """Calculate the transform needed to center volumes at origin and create debug Empty"""
        import bpy
        import mathutils
        
        log.info("=" * 60)
        log.info("CALCULATING CENTERING TRANSFORM")
        log.info("=" * 60)
        
        # Step 1: Find all volume objects
        volume_objects = []
        for series_uid, vol_name in patient.volume_objects.items():
            vol_obj = bpy.data.objects.get(vol_name)
            if vol_obj and vol_obj.type == 'VOLUME':
                volume_objects.append(vol_obj)
                log.info(f"Found volume: {vol_obj.name} at location {vol_obj.location}")
        
        if not volume_objects:
            log.warning("No volume objects found for centering")
            return
        
        # Step 2: Find the volume with the lowest Z position
        lowest_vol = min(volume_objects, key=lambda v: v.location.z)
        lowest_z = lowest_vol.location.z
        
        log.info("=" * 60)
        log.info("LOWEST VOLUME DETAILS:")
        log.info(f"  Name: {lowest_vol.name}")
        log.info(f"  Location (world): {lowest_vol.location}")
        log.info(f"  Dimensions (world): {lowest_vol.dimensions}")
        log.info(f"  Rotation (euler): {lowest_vol.rotation_euler}")
        log.info(f"  Scale: {lowest_vol.scale}")
        log.info(f"  Lowest Z: {lowest_z:.4f} m")
        
        # IMPORTANT: Volume objects in Blender have a quirk where matrix_world is identity
        # We need to manually calculate the world-space bounding box from location + rotation + dimensions
        
        import mathutils
        
        # Get volume properties
        loc = lowest_vol.location
        rot = lowest_vol.rotation_euler
        dims = lowest_vol.dimensions
        
        # Create transformation matrix manually
        mat_loc = mathutils.Matrix.Translation(loc)
        mat_rot = rot.to_matrix().to_4x4()
        mat_transform = mat_loc @ mat_rot
        
        log.info(f"  Manual transformation matrix:")
        for row in mat_transform:
            log.info(f"    {row}")
        
        # Get local bounding box corners (8 corners of a box from 0 to dimensions)
        # Volume bounding box in local space goes from (0,0,0) to (dim_x, dim_y, dim_z)
        local_corners = [
            mathutils.Vector((0, 0, 0)),
            mathutils.Vector((dims.x, 0, 0)),
            mathutils.Vector((0, dims.y, 0)),
            mathutils.Vector((dims.x, dims.y, 0)),
            mathutils.Vector((0, 0, dims.z)),
            mathutils.Vector((dims.x, 0, dims.z)),
            mathutils.Vector((0, dims.y, dims.z)),
            mathutils.Vector((dims.x, dims.y, dims.z)),
        ]
        
        # Transform to world space
        bbox_corners_world = [mat_transform @ corner for corner in local_corners]
        
        log.info(f"  Bounding box corners (world space after manual transformation):")
        for i, corner in enumerate(bbox_corners_world):
            log.info(f"    Corner {i}: ({corner.x:.4f}, {corner.y:.4f}, {corner.z:.4f})")
        
        # Find min/max of bounding box in world space
        bbox_min_x = min(c.x for c in bbox_corners_world)
        bbox_max_x = max(c.x for c in bbox_corners_world)
        bbox_min_y = min(c.y for c in bbox_corners_world)
        bbox_max_y = max(c.y for c in bbox_corners_world)
        bbox_min_z = min(c.z for c in bbox_corners_world)
        bbox_max_z = max(c.z for c in bbox_corners_world)
        
        log.info(f"  Bounding box X range: {bbox_min_x:.4f} to {bbox_max_x:.4f} (center: {(bbox_min_x + bbox_max_x)/2:.4f})")
        log.info(f"  Bounding box Y range: {bbox_min_y:.4f} to {bbox_max_y:.4f} (center: {(bbox_min_y + bbox_max_y)/2:.4f})")
        log.info(f"  Bounding box Z range: {bbox_min_z:.4f} to {bbox_max_z:.4f} (center: {(bbox_min_z + bbox_max_z)/2:.4f})")
        log.info(f"  Volume location Z: {lowest_vol.location.z:.4f}")
        log.info(f"  Difference (location.z - bbox_min_z): {lowest_vol.location.z - bbox_min_z:.4f}")
        log.info("=" * 60)
        
        # Step 3: Calculate the center point of the lowest plane using bounding box
        # The bounding box is already in world space, so just use it directly
        
        center_point = (
            (bbox_min_x + bbox_max_x) / 2.0,  # X center from bbox
            (bbox_min_y + bbox_max_y) / 2.0,  # Y center from bbox
            bbox_min_z                         # Z at lowest plane
        )
        
        log.info("=" * 60)
        log.info("CALCULATED CENTER POINT:")
        log.info(f"  Center point (from bbox): ({center_point[0]:.4f}, {center_point[1]:.4f}, {center_point[2]:.4f})")
        log.info("=" * 60)
        
        log.info(f"Calculated center point of lowest plane: {center_point}")
        
        # Step 4: Calculate the transform offset (center_point -> origin)
        transform_offset = (
            center_point[0],  # X offset to move center to 0
            center_point[1],  # Y offset to move center to 0
            center_point[2]   # Z offset to move center to 0
        )
        
        log.info(f"Transform offset to center at origin: {transform_offset}")
        
        # Step 5: Store the transform offset in scene property for later use (cursor positioning, etc.)
        import json
        context.scene.dicom_centering_offset = json.dumps({
            'x': transform_offset[0],
            'y': transform_offset[1],
            'z': transform_offset[2]
        })
        
        log.info("Stored centering offset in scene property")
        
        # Step 6: Apply the transform to ALL volumes and meshes
        log.info("=" * 60)
        log.info("APPLYING TRANSFORM TO VOLUMES AND MESHES")
        log.info("=" * 60)
        
        # Find all mesh objects (tissue meshes - VOLUME type objects with geometry nodes)
        mesh_objects = []
        for obj in bpy.data.objects:
            if obj.type == 'VOLUME':
                # Check if it's a tissue mesh (has modality prefix, tissue name, and series number)
                # Examples: CT_Bone_S1, MR_Soft_Tissue_S2, etc.
                has_modality = any(pattern in obj.name for pattern in ['CT_', 'MR_', 'PET_'])
                has_series = '_S' in obj.name
                # Exclude the main volume objects (they have "_Volume_" in the name)
                is_main_volume = '_Volume_' in obj.name
                
                if has_modality and has_series and not is_main_volume:
                    mesh_objects.append(obj)
                    log.info(f"Found mesh object: {obj.name}")
        
        # Apply transform to all volumes
        all_objects = volume_objects + mesh_objects
        log.info(f"Applying transform to {len(all_objects)} objects ({len(volume_objects)} volumes + {len(mesh_objects)} meshes)")
        
        for obj in all_objects:
            old_location = obj.location.copy()
            obj.location = (
                obj.location.x - transform_offset[0],
                obj.location.y - transform_offset[1],
                obj.location.z - transform_offset[2]
            )
            log.info(f"  {obj.type} {obj.name}: {old_location} -> {obj.location}")
        
        # Create debug Empty at origin (for verification)
        # Removed - no longer needed for production use
        
        log.info("Transform applied successfully!")
        log.info("=" * 60)

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
        import json
        
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
            log.error(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
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
            from .patient import Patient
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


classes = (
    IMPORT_OT_dicom_load_patient,
    IMPORT_OT_dicom_visualize_series,
    IMPORT_OT_dicom_preview_series,
    IMPORT_OT_dicom_set_tool,
    IMPORT_OT_dicom_toggle_series_selection,
    IMPORT_OT_dicom_toggle_display_mode,
    IMPORT_OT_dicom_calculate_volume,
    IMPORT_OT_dicom_reload_patient,
    IMPORT_OT_dicom_scan,
    IMPORT_OT_dicom_preview,
    IMPORT_OT_dicom_preview_popup,
    IMPORT_OT_dicom_open_in_editor,
    IMPORT_OT_dicom_preview_slice,
    IMAGE_OT_dicom_set_cursor_3d,
    IMAGE_OT_dicom_scroll,
    IMPORT_OT_dicom_import_series,
)

addon_keymaps = []

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Add keymap for double-click in Image Editor
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Image', space_type='IMAGE_EDITOR')
        kmi = km.keymap_items.new('image.dicom_set_cursor_3d', 'LEFTMOUSE', 'DOUBLE_CLICK')
        addon_keymaps.append((km, kmi))

def unregister():
    # Remove keymaps
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()
    
    # Clean up preview collections
    global preview_collections
    for pcoll in preview_collections.values():
        bpy.utils.previews.remove(pcoll)
    preview_collections.clear()
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)