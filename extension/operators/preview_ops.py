"""Preview and slice navigation operators"""

import bpy
import os
import json
import numpy as np
from bpy.props import StringProperty, IntProperty
from bpy.types import Operator

from ..dicom_io import load_slice
from ..preview import load_and_display_slice
from ..patient import Patient
from ..utils import SimpleLogger

log = SimpleLogger()

# Global storage for preview collections
preview_collections = {}


def _show_mask_overlay_for_slice(context, slice_index):
    """Show stored segmentation mask overlay for a slice (if available).
    
    Called after slice navigation to re-display propagated masks.
    """
    mask_path = context.scene.get("medsam_mask_stack_path", "")
    if not mask_path:
        return
    
    import os
    if not os.path.exists(mask_path):
        return
    
    try:
        data = np.load(mask_path)
        mask_stack = data['masks']
    except Exception:
        return
    
    if slice_index >= mask_stack.shape[0]:
        return
    
    mask = mask_stack[slice_index]
    if mask.sum() == 0:
        return
    
    # Load slice data for base image reconstruction
    series_list = json.loads(context.scene.dicom_series_data)
    if not series_list:
        return
    
    series_data = series_list[0]
    file_paths = series_data['files']
    
    if slice_index >= len(file_paths):
        return
    
    slice_data = load_slice(file_paths[slice_index])
    if not slice_data:
        return
    
    # Visualize the mask overlay (pass series_data for correct windowing)
    from ..ml.segmentation_ops import visualize_mask_overlay
    visualize_mask_overlay(context, mask, slice_data, series_data)


class IMPORT_OT_dicom_preview(Operator):
    """Load DICOM series for Image Editor preview"""
    bl_idname = "import.dicom_preview"
    bl_label = "Preview Series"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
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
        from ..ui_utils import set_image_in_all_editors
        
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
        
        from ..constants import MAX_PREVIEW_IMAGES
        
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
                from ..constants import PREVIEW_THUMBNAIL_SIZE
                
                temp_path = os.path.join(tempfile.gettempdir(), f"dicom_preview_{i}.png")
                img = Image.fromarray(normalized, mode='L')
                img = img.resize((PREVIEW_THUMBNAIL_SIZE, PREVIEW_THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
                img.save(temp_path)
                
                pcoll.load(f"slice_{i}", temp_path, 'IMAGE')
            except Exception as e:
                log.error(f"Failed to create preview {i}: {e}")
        
        preview_collections["main"] = pcoll
        
        from ..constants import PREVIEW_POPUP_WIDTH
        
        # Show popup
        return context.window_manager.invoke_popup(self, width=PREVIEW_POPUP_WIDTH)
    
    def draw(self, context):
        layout = self.layout
        
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
            
            from ..constants import MAX_PREVIEW_IMAGES
            
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
        from ..ui_utils import set_image_in_all_editors
        
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
        # Capture annotation prompts from current slice before navigating
        try:
            from ..ml.annotation_prompts import capture_annotation_prompts
            capture_annotation_prompts(context)
        except Exception:
            pass
        
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
        
        # Show stored mask overlay for new slice (if available)
        _show_mask_overlay_for_slice(context, self.slice_index)
        
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
            row_vec = np.array(orientation[:3])
            col_vec = np.array(orientation[3:])
            
            log.debug(f"Row vector: {row_vec}")
            log.debug(f"Col vector: {col_vec}")
            
            # Image is flipped in preview, so adjust Y coordinate
            pixel_y_flipped = img_height - 1 - pixel_y
            
            log.debug(f"Pixel Y flipped: {pixel_y} -> {pixel_y_flipped}")
            
            # Calculate 3D position
            pos_3d = np.array(position) + \
                     (pixel_x * pixel_spacing[1] * col_vec) + \
                     (pixel_y_flipped * pixel_spacing[0] * row_vec)
            
            log.debug(f"3D position (DICOM): {pos_3d}")
            
            # Find the volume object for THIS series
            series_number = slice_data["ds"].SeriesNumber if hasattr(slice_data["ds"], "SeriesNumber") else None
            
            volume_obj = None
            if series_number:
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
                vol_loc = volume_obj.location
                
                pixel_y_from_top = img_height - pixel_y
                
                img_height_mm = img_height * pixel_spacing[0]
                
                dicom_x = position[0] + pixel_x * pixel_spacing[1]
                dicom_y = position[1] + pixel_y_from_top * pixel_spacing[0]
                
                relative_x = dicom_x - position[0]
                relative_y = dicom_y - position[1]
                
                from ..constants import MM_TO_METERS
                
                offset_x = relative_x * MM_TO_METERS
                offset_y = relative_y * MM_TO_METERS
                
                volume_y_length = img_height_mm * MM_TO_METERS
                
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
            if context.scene.dicom_centering_offset:
                try:
                    offset_data = json.loads(context.scene.dicom_centering_offset)
                    centering_offset_z = offset_data['z']
                    
                    log.debug("=" * 60)
                    log.debug("APPLYING CENTERING OFFSET TO CURSOR (Z only)")
                    log.debug(f"Centering offset Z: {centering_offset_z}")
                    
                    blender_pos = (
                        blender_pos[0],
                        blender_pos[1],
                        blender_pos[2] - centering_offset_z
                    )
                    
                    log.debug(f"3D position (after centering): {blender_pos}")
                    log.debug("=" * 60)
                except Exception as e:
                    log.error(f"Failed to apply centering offset: {e}")
            
            from ..ui_utils import refresh_all_3d_views
            
            # Set 3D cursor
            context.scene.cursor.location = blender_pos
            
            # ALSO store point for MedSAM segmentation
            try:
                points = json.loads(context.scene.medsam_points)
            except:
                points = []
            
            # Add point with slice index
            points.append({
                'x': pixel_x,
                'y': pixel_y_flipped,
                'slice_index': slice_index,
                'label': 1  # Positive point
            })
            
            context.scene.medsam_points = json.dumps(points)
            log.debug(f"Added MedSAM point: ({pixel_x}, {pixel_y_flipped}) on slice {slice_index}")
            
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
        if context.scene.dicom_preview_slice_count > 0:
            current = context.scene.dicom_preview_slice_index
            
            # Capture annotation prompts from current slice before navigating
            try:
                from ..ml.annotation_prompts import capture_annotation_prompts
                capture_annotation_prompts(context)
            except Exception:
                pass
            
            # For 4D series, constrain scrolling to current time point
            if context.scene.dicom_preview_is_4d:
                slices_per_tp = context.scene.dicom_preview_slices_per_time_point
                current_tp = context.scene.dicom_preview_time_point_index
                
                # Calculate slice range for current time point
                tp_start = current_tp * slices_per_tp
                tp_end = tp_start + slices_per_tp - 1
                
                # Calculate new index within time point
                new_index = current + self.direction
                new_index = max(tp_start, min(tp_end, new_index))
            else:
                # Regular series - scroll through all slices
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
                        
                        # Show stored mask overlay for new slice (if available)
                        _show_mask_overlay_for_slice(context, new_index)
        
        return {'FINISHED'}


class IMAGE_OT_dicom_scroll_time_point(Operator):
    """Scroll through 4D time points"""
    bl_idname = "image.dicom_scroll_time_point"
    bl_label = "Scroll Time Points"
    bl_options = {'INTERNAL'}
    
    direction: IntProperty(default=0)
    
    def execute(self, context):
        if not context.scene.dicom_preview_is_4d:
            return {'CANCELLED'}
        
        current_tp = context.scene.dicom_preview_time_point_index
        new_tp = current_tp + self.direction
        new_tp = max(0, min(context.scene.dicom_preview_time_point_count - 1, new_tp))
        
        if new_tp != current_tp:
            context.scene.dicom_preview_time_point_index = new_tp
            
            # Jump to first slice of new time point
            slices_per_tp = context.scene.dicom_preview_slices_per_time_point
            new_slice_index = new_tp * slices_per_tp
            
            # Load the slice
            series_list = json.loads(context.scene.dicom_series_data)
            series_idx = context.scene.dicom_preview_series_index
            
            if series_idx < len(series_list):
                series = series_list[series_idx]
                if new_slice_index < len(series['files']):
                    context.scene.dicom_preview_slice_index = new_slice_index
                    load_and_display_slice(context, series['files'][new_slice_index], series)
        
        return {'FINISHED'}


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
            from ..ui_utils import clear_image_from_all_editors
            
            old_img = bpy.data.images["DICOM_Preview"]
            # First, clear it from all Image Editors
            clear_image_from_all_editors(context, old_img)
            # Now remove the image
            bpy.data.images.remove(old_img)
            log.debug("Cleared old DICOM_Preview image")
        
        # Store preview info in scene (for scrolling and slice navigation)
        context.scene.dicom_preview_slice_index = 0
        context.scene.dicom_preview_slice_count = len(file_paths)
        context.scene.dicom_preview_series_index = 0
        
        # Store 4D metadata if applicable
        if series.is_4d:
            context.scene.dicom_preview_is_4d = True
            context.scene.dicom_preview_time_point_index = 0
            context.scene.dicom_preview_time_point_count = series.num_time_points
            context.scene.dicom_preview_slices_per_time_point = series.time_points[0]['slice_count']
            log.info(f"4D preview: {series.num_time_points} time points, {series.time_points[0]['slice_count']} slices per time point")
        else:
            context.scene.dicom_preview_is_4d = False
            context.scene.dicom_preview_time_point_index = 0
            context.scene.dicom_preview_time_point_count = 0
            context.scene.dicom_preview_slices_per_time_point = 0
        
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
            dicom_img = bpy.data.images.get("DICOM_Preview")
            if dicom_img:
                from ..ui_utils import set_image_in_all_editors
                
                image_editor_found = set_image_in_all_editors(
                    context, 
                    dicom_img, 
                    clear_first=True
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


def cleanup_preview_collections():
    """Clean up preview collections on unregister"""
    global preview_collections
    for pcoll in preview_collections.values():
        bpy.utils.previews.remove(pcoll)
    preview_collections.clear()
