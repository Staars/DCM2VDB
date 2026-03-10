"""
Annotation-to-prompt conversion for MedSAM2 segmentation.

Reads Blender Image Editor annotation strokes and converts them
into canonical prompt objects (bounding boxes + points) for MedSAM2.

Annotation layers named "Positive" produce positive prompts (box + centroid).
Annotation layers named "Negative" produce negative point prompts.
All other layers are treated as positive.
"""

import bpy
import numpy as np
import json
from ..utils import SimpleLogger
from ..patient import Patient

log = SimpleLogger()

# Layer name conventions
POSITIVE_LAYER_NAME = "Positive"
NEGATIVE_LAYER_NAME = "Negative"


def _annotation_co_to_pixel(co, img_w, img_h):
    """Convert annotation stroke point coordinates to pixel coordinates.
    
    Blender Image Editor annotation strokes use normalized UV coordinates
    where (0,0) is bottom-left and (1,1) is top-right.
    DICOM pixel space has (0,0) at top-left, so we invert Y.
    """
    x, y = co.x, co.y
    
    # UV (0-1) -> pixel coordinates
    px = int(round(x * img_w))
    py = int(round((1.0 - y) * img_h))  # UV y=0 is bottom, pixel y=0 is top
    
    return max(0, min(px, img_w - 1)), max(0, min(py, img_h - 1))


def _get_annotation_data(context):
    """Get annotation data from the Image Editor context.
    
    Returns the annotation datablock, or None.
    Blender 5.0: Scene.annotation (renamed from Scene.grease_pencil)
    """
    # Blender 5.0+: context.annotation_data
    if hasattr(context, 'annotation_data') and context.annotation_data:
        return context.annotation_data
    
    # Blender 5.0: Scene.annotation (renamed from Scene.grease_pencil)
    if hasattr(context.scene, 'annotation') and context.scene.annotation:
        return context.scene.annotation
    
    # Legacy fallback (Blender < 5.0)
    if hasattr(context.scene, 'grease_pencil') and context.scene.grease_pencil:
        return context.scene.grease_pencil
    
    return None


def _is_negative_layer(layer):
    """Check if an annotation layer is a negative prompt layer."""
    return NEGATIVE_LAYER_NAME.lower() in layer.info.lower()


def _is_layer_hidden(layer):
    """Check if an annotation layer is hidden.
    
    Blender 5.0: annotation_hide (renamed from hide)
    """
    if hasattr(layer, 'annotation_hide'):
        return layer.annotation_hide
    if hasattr(layer, 'hide'):
        return layer.hide
    return False


def capture_annotation_prompts(context):
    """Capture current annotation strokes and convert to prompt objects.
    
    Reads all annotation layers/strokes visible in the current Image Editor,
    converts them to canonical prompt format, and stores them in the 
    active series' segmentation_prompts.
    
    Returns:
        dict or None: The prompt set for the current slice, or None if no annotations.
    """
    img = bpy.data.images.get("DICOM_Preview")
    if not img:
        return None
    
    img_w, img_h = img.size
    slice_index = context.scene.dicom_preview_slice_index
    
    annotation = _get_annotation_data(context)
    if not annotation:
        return None
    
    points = []
    boxes = []
    
    for layer in annotation.layers:
        if _is_layer_hidden(layer):
            continue
        
        is_negative = _is_negative_layer(layer)
        
        # Get active frame (current frame's strokes)
        frame = layer.active_frame
        if not frame:
            continue
        
        for stroke in frame.strokes:
            if len(stroke.points) == 0:
                continue
            
            # Convert all stroke points to pixel coordinates
            pixel_coords = []
            for pt in stroke.points:
                px, py = _annotation_co_to_pixel(pt.co, img_w, img_h)
                pixel_coords.append((px, py))
            
            if is_negative:
                # Negative strokes: sample evenly spaced negative points
                n_samples = min(8, max(1, len(pixel_coords)))
                if len(pixel_coords) <= n_samples:
                    sampled = pixel_coords
                else:
                    indices = np.linspace(0, len(pixel_coords) - 1, n_samples, dtype=int)
                    sampled = [pixel_coords[i] for i in indices]
                
                for px, py in sampled:
                    points.append({"x": px, "y": py, "label": 0})
            else:
                # Positive strokes: compute bounding box + centroid
                xs = [c[0] for c in pixel_coords]
                ys = [c[1] for c in pixel_coords]
                
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                
                # Only create a box if the stroke has extent (not a single dot)
                if x1 - x0 > 2 or y1 - y0 > 2:
                    boxes.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
                
                # Add centroid as positive point
                cx = int(round(sum(xs) / len(xs)))
                cy = int(round(sum(ys) / len(ys)))
                points.append({"x": cx, "y": cy, "label": 1})
    
    if not points and not boxes:
        return None
    
    prompt_set = {}
    if points:
        prompt_set["points"] = points
    if boxes:
        prompt_set["boxes"] = boxes
    
    # Store in patient backend
    _store_prompts_for_slice(context, slice_index, prompt_set)
    
    # Mirror to scene.medsam_points for backward compatibility
    _mirror_to_scene_points(context, slice_index, prompt_set)
    
    log.info(f"Captured annotation prompts for slice {slice_index}: "
             f"{len(points)} points, {len(boxes)} boxes")
    
    return prompt_set


def _store_prompts_for_slice(context, slice_index, prompt_set):
    """Store prompt set in the active series' segmentation_prompts."""
    if not context.scene.dicom_patient_data:
        return
    
    try:
        patient = Patient.from_json(context.scene.dicom_patient_data)
        if not patient.series:
            return
        
        # Use first series (the one being previewed)
        series = patient.series[0]
        series.segmentation_prompts[str(slice_index)] = prompt_set
        
        # Save back
        context.scene.dicom_patient_data = patient.to_json()
    except Exception as e:
        log.error(f"Failed to store prompts: {e}")


def _mirror_to_scene_points(context, slice_index, prompt_set):
    """Mirror prompt points to scene.medsam_points for backward compat."""
    try:
        all_points = json.loads(context.scene.medsam_points)
    except:
        all_points = []
    
    # Remove old points for this slice
    all_points = [p for p in all_points if p.get('slice_index') != slice_index]
    
    # Add new points from prompt set
    for pt in prompt_set.get("points", []):
        all_points.append({
            'x': pt['x'],
            'y': pt['y'],
            'slice_index': slice_index,
            'label': pt['label']
        })
    
    # Add box corners as labeled points (label 2=top-left, 3=bottom-right)
    for box in prompt_set.get("boxes", []):
        all_points.append({
            'x': box['x0'], 'y': box['y0'],
            'slice_index': slice_index, 'label': 2
        })
        all_points.append({
            'x': box['x1'], 'y': box['y1'],
            'slice_index': slice_index, 'label': 3
        })
    
    context.scene.medsam_points = json.dumps(all_points)


def get_prompts_for_slice(context, slice_index):
    """Get stored prompts for a given slice from patient backend.
    
    Returns:
        dict or None: {"points": [...], "boxes": [...]} or None
    """
    if not context.scene.dicom_patient_data:
        return None
    
    try:
        patient = Patient.from_json(context.scene.dicom_patient_data)
        if not patient.series:
            return None
        
        series = patient.series[0]
        return series.segmentation_prompts.get(str(slice_index))
    except:
        return None


def flatten_prompts_for_medsam(prompt_set):
    """Convert a canonical prompt set to MedSAM2 coords/labels arrays.
    
    Returns:
        tuple: (coords_array, labels_array) both np.float32
               coords shape (N, 2), labels shape (N,)
    """
    coords = []
    labels = []
    
    if not prompt_set:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
    
    # Add box prompts first (label 2 = top-left corner, label 3 = bottom-right corner)
    for box in prompt_set.get("boxes", []):
        coords.append([box['x0'], box['y0']])
        labels.append(2)
        coords.append([box['x1'], box['y1']])
        labels.append(3)
    
    # Add point prompts
    for pt in prompt_set.get("points", []):
        coords.append([pt['x'], pt['y']])
        labels.append(pt['label'])
    
    return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.float32)


def prompts_from_mask(mask):
    """Derive prompt set from a binary segmentation mask.
    
    Used for propagation: converts the previous slice's mask into
    prompts for the next slice (bounding box + centroid + support points).
    
    Args:
        mask: Binary mask (H, W) as uint8
        
    Returns:
        dict: Canonical prompt set with boxes and points
    """
    if mask is None or mask.sum() == 0:
        return None
    
    # Find foreground pixels
    ys, xs = np.where(mask > 0)
    
    if len(xs) == 0:
        return None
    
    # Bounding box with small margin
    margin = 5
    h, w = mask.shape
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(w - 1, int(xs.max()) + margin)
    y1 = min(h - 1, int(ys.max()) + margin)
    
    # Centroid
    cx = int(round(xs.mean()))
    cy = int(round(ys.mean()))
    
    prompt_set = {
        "boxes": [{"x0": x0, "y0": y0, "x1": x1, "y1": y1}],
        "points": [{"x": cx, "y": cy, "label": 1}]
    }
    
    return prompt_set


def accept_propagation(prev_mask, new_mask, iou, min_iou=0.3, 
                       area_ratio_range=(0.2, 5.0)):
    """Decide whether to accept a propagated segmentation result.
    
    Args:
        prev_mask: Previous slice mask (H, W) uint8
        new_mask: New slice mask (H, W) uint8
        iou: IoU prediction score
        min_iou: Minimum IoU threshold
        area_ratio_range: (min_ratio, max_ratio) of new/prev area
        
    Returns:
        bool: Whether to accept the propagation
    """
    # Reject empty masks
    new_area = new_mask.sum()
    if new_area == 0:
        log.info("Propagation rejected: empty mask")
        return False
    
    # Check IoU threshold
    if iou < min_iou:
        log.info(f"Propagation rejected: IoU {iou:.4f} < {min_iou}")
        return False
    
    # Check area ratio
    prev_area = prev_mask.sum()
    if prev_area > 0:
        ratio = new_area / prev_area
        if ratio < area_ratio_range[0] or ratio > area_ratio_range[1]:
            log.info(f"Propagation rejected: area ratio {ratio:.2f} outside range {area_ratio_range}")
            return False
    
    # Check bounding box overlap
    prev_ys, prev_xs = np.where(prev_mask > 0)
    new_ys, new_xs = np.where(new_mask > 0)
    
    if len(prev_xs) > 0 and len(new_xs) > 0:
        # Compute bbox overlap
        prev_box = (prev_xs.min(), prev_ys.min(), prev_xs.max(), prev_ys.max())
        new_box = (new_xs.min(), new_ys.min(), new_xs.max(), new_ys.max())
        
        # Intersection
        ix0 = max(prev_box[0], new_box[0])
        iy0 = max(prev_box[1], new_box[1])
        ix1 = min(prev_box[2], new_box[2])
        iy1 = min(prev_box[3], new_box[3])
        
        if ix1 < ix0 or iy1 < iy0:
            log.info("Propagation rejected: no bounding box overlap")
            return False
    
    return True
