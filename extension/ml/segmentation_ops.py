"""
MedSAM2 segmentation operators for Blender

Provides operators to:
- Capture annotation strokes as prompts
- Run AI segmentation on the seed slice
- Propagate segmentation bidirectionally through the volume
- Clear prompts and overlays
"""

import bpy
from bpy.types import Operator
import numpy as np

from .annotation_prompts import (
    capture_annotation_prompts,
    get_prompts_for_slice,
    flatten_prompts_for_medsam,
    prompts_from_mask,
    accept_propagation,
)


class IMAGE_OT_medsam_add_point(Operator):
    """Add a point prompt for MedSAM2 segmentation"""
    bl_idname = "image.medsam_add_point"
    bl_label = "Add MedSAM Point"
    bl_options = {'REGISTER', 'UNDO'}
    
    positive: bpy.props.BoolProperty(
        name="Positive",
        description="Positive (include) or negative (exclude) point",
        default=True
    )
    
    @classmethod
    def poll(cls, context):
        return (context.area and context.area.type == 'IMAGE_EDITOR' and
                context.scene.dicom_preview_slice_count > 0 and
                "DICOM_Preview" in bpy.data.images)
    
    def invoke(self, context, event):
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
        
        # Get mouse coordinates
        mouse_x = event.mouse_region_x
        mouse_y = event.mouse_region_y
        
        # Image dimensions
        img_width, img_height = img.size
        
        # Convert to normalized texture coordinates
        view2d = region.view2d
        texture_x, texture_y = view2d.region_to_view(mouse_x, mouse_y)
        
        # Check bounds
        if texture_x < 0 or texture_x > 1 or texture_y < 0 or texture_y > 1:
            return {'CANCELLED'}
        
        # Convert to pixel coordinates
        pixel_x = int(texture_x * img_width)
        pixel_y = int(texture_y * img_height)
        
        # Clamp
        pixel_x = max(0, min(pixel_x, img_width - 1))
        pixel_y = max(0, min(pixel_y, img_height - 1))
        
        # Image is flipped in preview
        pixel_y_flipped = img_height - 1 - pixel_y
        
        # For now, just report
        label = "positive" if self.positive else "negative"
        self.report({'INFO'}, f"Added {label} point at ({pixel_x}, {pixel_y_flipped})")
        
        return {'FINISHED'}


def _load_slice_rgb(context, slice_index):
    """Load a DICOM slice as RGB image for MedSAM2.
    
    Returns:
        tuple: (rgb_image, slice_data, series_data) or (None, None, None)
    """
    import json
    from ..dicom_io import load_slice
    from ..utils import SimpleLogger
    from ..constants import PERCENTILE_MIN, PERCENTILE_MAX
    
    log = SimpleLogger()
    
    if not context.scene.dicom_series_data:
        return None, None, None
    
    series_list = json.loads(context.scene.dicom_series_data)
    if not series_list:
        return None, None, None
    
    series_data = series_list[0]
    file_paths = series_data['files']
    
    if slice_index >= len(file_paths):
        return None, None, None
    
    slice_data = load_slice(file_paths[slice_index])
    if not slice_data:
        return None, None, None
    
    pixels = slice_data["pixels"]
    
    # Apply window/level
    wc = series_data.get('window_center') or slice_data.get('window_center')
    ww = series_data.get('window_width') or slice_data.get('window_width')
    
    if wc is not None and ww is not None and ww > 0:
        low = wc - ww / 2
        high = wc + ww / 2
        pixels_windowed = np.clip(pixels, low, high)
        normalized = ((pixels_windowed - low) / ww * 255).astype(np.uint8)
    else:
        pmin, pmax = np.percentile(pixels, [PERCENTILE_MIN, PERCENTILE_MAX])
        if pmax > pmin:
            normalized = np.clip((pixels - pmin) / (pmax - pmin) * 255, 0, 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(pixels, dtype=np.uint8)
    
    rgb_image = np.stack([normalized] * 3, axis=-1)
    return rgb_image, slice_data, series_data


def visualize_mask_overlay(context, mask, slice_data, series_data=None):
    """Overlay mask on DICOM preview image.
    
    Composites a red semi-transparent overlay of the segmentation mask
    onto the current DICOM preview image. Uses the same windowing logic
    as load_and_display_slice (series-level override, then per-slice).
    
    Args:
        context: Blender context
        mask: Binary mask (H, W) as uint8 or float
        slice_data: Dict with 'pixels', optional 'window_center'/'window_width'
        series_data: Optional series dict with 'window_center'/'window_width' overrides
    """
    from PIL import Image
    from ..utils import SimpleLogger
    
    log = SimpleLogger()
    
    img = bpy.data.images.get("DICOM_Preview")
    if not img:
        return
    
    img_width, img_height = img.size
    mask_height, mask_width = mask.shape
    
    # Resize mask if needed
    if mask_width != img_width or mask_height != img_height:
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((img_width, img_height), Image.BILINEAR)
        mask = np.array(mask_pil) / 255.0
    else:
        mask = mask.astype(np.float32)
    
    # Get base image from slice data
    # Use same windowing as load_and_display_slice: series override first
    pixels = slice_data["pixels"]
    
    if series_data:
        wc = series_data.get('window_center') or slice_data.get('window_center')
        ww = series_data.get('window_width') or slice_data.get('window_width')
    else:
        wc = slice_data.get('window_center')
        ww = slice_data.get('window_width')
    
    if wc is not None and ww is not None and ww > 0:
        low = wc - ww / 2
        high = wc + ww / 2
        pixels_windowed = np.clip(pixels, low, high)
        normalized = ((pixels_windowed - low) / ww).astype(np.float32)
    else:
        from ..constants import PERCENTILE_MIN, PERCENTILE_MAX
        pmin, pmax = np.percentile(pixels, [PERCENTILE_MIN, PERCENTILE_MAX])
        if pmax > pmin:
            normalized = np.clip((pixels - pmin) / (pmax - pmin), 0, 1).astype(np.float32)
        else:
            normalized = np.zeros_like(pixels, dtype=np.float32)
    
    height, width = normalized.shape
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    
    rgba[:, :, 0] = normalized
    rgba[:, :, 1] = normalized
    rgba[:, :, 2] = normalized
    rgba[:, :, 3] = 1.0
    
    # Red semi-transparent mask overlay
    mask_alpha = 0.4
    rgba[:, :, 0] = np.where(mask > 0.5,
                              rgba[:, :, 0] * (1 - mask_alpha) + mask_alpha,
                              rgba[:, :, 0])
    
    # Flip for Blender
    rgba = np.flipud(rgba)
    
    img.pixels[:] = rgba.ravel()
    img.update()
    
    for area in context.screen.areas:
        if area.type == 'IMAGE_EDITOR':
            area.tag_redraw()
    
    log.info("Mask overlay applied")


class IMAGE_OT_medsam_segment(Operator):
    """Run MedSAM2 segmentation with annotation prompts and propagate through volume"""
    bl_idname = "image.medsam_segment"
    bl_label = "Run MedSAM Segmentation"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return (context.scene.dicom_preview_slice_count > 0 and
                "DICOM_Preview" in bpy.data.images)
    
    def execute(self, context):
        import json
        from ..utils import SimpleLogger
        
        log = SimpleLogger()
        
        slice_index = context.scene.dicom_preview_slice_index
        slice_count = context.scene.dicom_preview_slice_count
        
        # Step 1: Capture current annotation strokes as prompts
        annotation_prompts = capture_annotation_prompts(context)
        
        # Step 2: Get prompts (from annotations or stored backend)
        prompt_set = annotation_prompts or get_prompts_for_slice(context, slice_index)
        
        if not prompt_set:
            # Fallback: try legacy medsam_points
            try:
                all_points = json.loads(context.scene.medsam_points)
                slice_points = [p for p in all_points if p['slice_index'] == slice_index]
                if slice_points:
                    prompt_set = {
                        "points": [{"x": p['x'], "y": p['y'], "label": p['label']} 
                                   for p in slice_points]
                    }
            except:
                pass
        
        if not prompt_set:
            self.report({'WARNING'}, "No prompts found. Draw annotations or double-click to add points.")
            return {'CANCELLED'}
        
        # Step 3: Flatten prompts for MedSAM
        coords, labels = flatten_prompts_for_medsam(prompt_set)
        
        if len(coords) == 0:
            self.report({'WARNING'}, "No valid prompts.")
            return {'CANCELLED'}
        
        log.info(f"Segmenting slice {slice_index} with {len(coords)} prompts")
        
        # Step 4: Load seed slice
        rgb_image, slice_data, series_data = _load_slice_rgb(context, slice_index)
        if rgb_image is None:
            self.report({'ERROR'}, "Failed to load slice")
            return {'CANCELLED'}
        
        orig_height, orig_width = rgb_image.shape[:2]
        
        # Step 5: Run segmentation on seed slice
        try:
            from . import get_predictor
            predictor = get_predictor()
            
            log.info(f"Running MedSAM2 on seed slice {slice_index}...")
            result = predictor.segment(rgb_image, coords, labels)
            
            seed_mask = result['mask']
            seed_iou = result['iou']
            
            log.info(f"Seed segmentation: IoU={seed_iou:.4f}, "
                     f"positive pixels={seed_mask.sum()}/{seed_mask.size}")
            
            if seed_mask.sum() == 0:
                self.report({'WARNING'}, f"No object found on seed slice. IoU={seed_iou:.4f}")
                return {'CANCELLED'}
            
        except Exception as e:
            log.error(f"Segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Segmentation failed: {e}")
            return {'CANCELLED'}
        
        # Step 6: Initialize mask stack
        mask_stack = np.zeros((slice_count, orig_height, orig_width), dtype=np.uint8)
        mask_stack[slice_index] = seed_mask
        
        # Step 7: Propagate in both directions
        total_slices = 1  # seed already done
        
        for direction in (+1, -1):
            dir_name = "forward" if direction > 0 else "backward"
            log.info(f"Propagating {dir_name} from slice {slice_index}...")
            
            prev_mask = seed_mask
            prev_idx = slice_index
            
            while True:
                idx = prev_idx + direction
                
                # Bounds check
                if idx < 0 or idx >= slice_count:
                    log.info(f"  {dir_name}: reached boundary at slice {idx}")
                    break
                
                # Derive prompts from previous mask
                prop_prompts = prompts_from_mask(prev_mask)
                if not prop_prompts:
                    log.info(f"  {dir_name}: no prompts derivable from mask at slice {prev_idx}")
                    break
                
                prop_coords, prop_labels = flatten_prompts_for_medsam(prop_prompts)
                
                # Load next slice
                rgb_next, _, _ = _load_slice_rgb(context, idx)
                if rgb_next is None:
                    log.info(f"  {dir_name}: failed to load slice {idx}")
                    break
                
                # Segment
                try:
                    result = predictor.segment(rgb_next, prop_coords, prop_labels)
                    new_mask = result['mask']
                    new_iou = result['iou']
                except Exception as e:
                    log.error(f"  {dir_name}: segmentation failed on slice {idx}: {e}")
                    break
                
                # Accept/reject
                if not accept_propagation(prev_mask, new_mask, new_iou):
                    log.info(f"  {dir_name}: propagation stopped at slice {idx} "
                             f"(IoU={new_iou:.4f})")
                    break
                
                mask_stack[idx] = new_mask
                prev_mask = new_mask
                prev_idx = idx
                total_slices += 1
                
                log.info(f"  {dir_name}: accepted slice {idx} "
                         f"(IoU={new_iou:.4f}, area={new_mask.sum()})")
        
        # Step 8: Store mask stack
        self._store_mask_stack(context, mask_stack)
        
        # Step 9: Clear annotation strokes (consumed as prompts)
        self._clear_annotations(context)
        
        # Step 10: Visualize on current slice
        self._visualize_mask(context, mask_stack[slice_index], slice_data, series_data)
        
        self.report({'INFO'}, 
                    f"Segmentation complete: {total_slices} slices. "
                    f"Seed IoU={seed_iou:.4f}")
        
        return {'FINISHED'}
    
    def _store_mask_stack(self, context, mask_stack):
        """Store the mask stack as a temporary file and reference it."""
        import os
        import tempfile
        from ..utils import SimpleLogger
        
        log = SimpleLogger()
        
        # Save as compressed numpy archive
        cache_dir = os.path.join(tempfile.gettempdir(), "dcm2vdb_masks")
        os.makedirs(cache_dir, exist_ok=True)
        
        uid = context.scene.get("dicom_volume_unique_id", "unknown")
        mask_path = os.path.join(cache_dir, f"mask_stack_{uid}.npz")
        
        np.savez_compressed(mask_path, masks=mask_stack)
        log.info(f"Saved mask stack ({mask_stack.shape}) to {mask_path}")
        
        # Store path in scene for later use
        context.scene["medsam_mask_stack_path"] = mask_path
    
    def _clear_annotations(self, context):
        """Clear annotation strokes after they've been consumed as prompts."""
        from .annotation_prompts import _get_annotation_data
        
        annotation = _get_annotation_data(context)
        if not annotation:
            return
        
        for layer in annotation.layers:
            for frame in list(layer.frames):
                layer.frames.remove(frame)
    
    def _visualize_mask(self, context, mask, slice_data, series_data=None):
        """Overlay mask on DICOM preview image"""
        visualize_mask_overlay(context, mask, slice_data, series_data)


class IMAGE_OT_medsam_clear_points(Operator):
    """Clear all MedSAM prompts and segmentation overlays"""
    bl_idname = "image.medsam_clear_points"
    bl_label = "Clear MedSAM Points"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        import json
        from ..dicom_io import load_slice
        from ..utils import SimpleLogger
        from ..patient import Patient
        
        log = SimpleLogger()
        
        # Clear legacy points
        context.scene.medsam_points = "[]"
        
        # Clear stored mask stack
        if "medsam_mask_stack_path" in context.scene:
            del context.scene["medsam_mask_stack_path"]
        
        # Clear segmentation prompts from patient backend
        if context.scene.dicom_patient_data:
            try:
                patient = Patient.from_json(context.scene.dicom_patient_data)
                if patient.series:
                    patient.series[0].segmentation_prompts = {}
                    context.scene.dicom_patient_data = patient.to_json()
            except Exception as e:
                log.error(f"Failed to clear backend prompts: {e}")
        
        # Clear annotation strokes
        self._clear_annotations(context)
        
        # Reload original DICOM image
        slice_index = context.scene.dicom_preview_slice_index
        if context.scene.dicom_series_data:
            try:
                series_list = json.loads(context.scene.dicom_series_data)
                if series_list:
                    series_data = series_list[0]
                    file_paths = series_data['files']
                    if slice_index < len(file_paths):
                        from ..preview import load_and_display_slice
                        load_and_display_slice(context, file_paths[slice_index], series_data)
            except Exception as e:
                log.error(f"Failed to clear overlay: {e}")
        
        self.report({'INFO'}, "Cleared all prompts and overlay")
        return {'FINISHED'}
    
    def _clear_annotations(self, context):
        """Clear annotation strokes from the Image Editor."""
        from .annotation_prompts import _get_annotation_data
        from ..utils import SimpleLogger
        
        log = SimpleLogger()
        
        annotation = _get_annotation_data(context)
        if not annotation:
            return
        
        for layer in annotation.layers:
            # Clear all frames/strokes
            for frame in list(layer.frames):
                layer.frames.remove(frame)
        
        log.info("Cleared all annotation strokes")


class IMAGE_OT_medsam_show_overlay(Operator):
    """Show stored segmentation mask overlay for current slice"""
    bl_idname = "image.medsam_show_overlay"
    bl_label = "Show Mask Overlay"
    bl_options = {'INTERNAL'}
    
    @classmethod
    def poll(cls, context):
        return ("medsam_mask_stack_path" in context.scene and
                context.scene.dicom_preview_slice_count > 0)
    
    def execute(self, context):
        import os
        from ..utils import SimpleLogger
        
        log = SimpleLogger()
        
        mask_path = context.scene.get("medsam_mask_stack_path", "")
        if not mask_path or not os.path.exists(mask_path):
            return {'CANCELLED'}
        
        try:
            data = np.load(mask_path)
            mask_stack = data['masks']
        except Exception as e:
            log.error(f"Failed to load mask stack: {e}")
            return {'CANCELLED'}
        
        slice_index = context.scene.dicom_preview_slice_index
        if slice_index >= mask_stack.shape[0]:
            return {'CANCELLED'}
        
        mask = mask_stack[slice_index]
        
        # Only show overlay if this slice has a mask
        if mask.sum() == 0:
            return {'FINISHED'}
        
        # Load slice data for base image
        import json
        from ..dicom_io import load_slice
        
        series_list = json.loads(context.scene.dicom_series_data)
        if not series_list:
            return {'CANCELLED'}
        
        series_data = series_list[0]
        file_paths = series_data['files']
        
        if slice_index >= len(file_paths):
            return {'CANCELLED'}
        
        slice_data = load_slice(file_paths[slice_index])
        if not slice_data:
            return {'CANCELLED'}
        
        visualize_mask_overlay(context, mask, slice_data, series_data)
        
        return {'FINISHED'}


# Registration
classes = (
    IMAGE_OT_medsam_add_point,
    IMAGE_OT_medsam_segment,
    IMAGE_OT_medsam_clear_points,
    IMAGE_OT_medsam_show_overlay,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
