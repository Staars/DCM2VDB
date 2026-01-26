"""UI panel definitions"""

import bpy
from .utils import SimpleLogger
from bpy.types import Panel
from pathlib import Path

import os

# Get logger for this extension
log = SimpleLogger()

from .operators import (
    IMPORT_OT_dicom_load_patient,
    IMPORT_OT_dicom_visualize_series,
    IMPORT_OT_dicom_preview_series,
    IMPORT_OT_dicom_scan,
    IMPORT_OT_dicom_preview_popup,
    IMPORT_OT_dicom_preview_slice,
    IMPORT_OT_dicom_import_series,
    IMPORT_OT_dicom_open_in_editor
)
from .patient import Patient
from .preview import generate_series_preview_icons

# Global preview collection
preview_collection = None

def get_preview_collection():
    """Get or create preview collection."""
    global preview_collection
    if preview_collection is None:
        import bpy.utils.previews
        preview_collection = bpy.utils.previews.new()
    return preview_collection

class VIEW3D_PT_dicom_patient(Panel):
    """Patient-centric DICOM panel"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOM"
    bl_label = "DICOM Patient"
    bl_context = "objectmode"
    
    def draw(self, context):
        layout = self.layout
        scn = context.scene
        
        # Check if patient is loaded
        if not scn.dicom_patient_data:
            # No patient loaded
            box = layout.box()
            box.label(text="No patient loaded", icon='INFO')
            box.operator(IMPORT_OT_dicom_load_patient.bl_idname, text="Load Patient...", icon='FILE_FOLDER')
            return
        
        # Load patient data
        try:
            patient = Patient.from_json(scn.dicom_patient_data)
        except Exception as e:
            layout.label(text=f"Error loading patient: {e}", icon='ERROR')
            return
        
        # Patient info and actions
        box = layout.box()
        box.label(text=f"Patient: {patient.patient_name}", icon='USER')
        if patient.patient_id:
            box.label(text=f"ID: {patient.patient_id}")
        if patient.study_description:
            box.label(text=f"Study: {patient.study_description}")
        if patient.study_date:
            box.label(text=f"Date: {patient.study_date}")
        
        box.operator(IMPORT_OT_dicom_load_patient.bl_idname, text="Load Patient", icon='FILE_FOLDER')
        
        # Series list (collapsible) - always visible
        layout.separator()
        box = layout.box()
        
        # Reload button at the top
        row = box.row()
        row.operator("import.dicom_reload_patient", text="Reload", icon='FILE_REFRESH')
        box.separator()
        
        # Collapsible header
        row = box.row()
        row.prop(scn, "dicom_show_series_list",
            icon="TRIA_DOWN" if scn.dicom_show_series_list else "TRIA_RIGHT",
            icon_only=True, emboss=False
        )
        row.label(text=f"Primary Series ({len(patient.series)})", icon='RENDERLAYERS')
        
        # Only show series list if expanded
        if scn.dicom_show_series_list:
            groups = patient.get_series_by_frame_of_reference()
            
            for frame_uid, series_list in groups.items():
                # Series in this frame (no frame header)
                for series in series_list:
                    # Series header with checkbox
                    header_row = box.row(align=True)
                    
                    # Selection checkbox
                    op = header_row.operator(
                        "import.dicom_toggle_series_selection",
                        text="",
                        icon='CHECKBOX_HLT' if series.is_selected else 'CHECKBOX_DEHLT',
                        emboss=False
                    )
                    op.series_uid = series.series_instance_uid
                    
                    # Series info: Modality: dimensions - Series: number
                    series_info = f"{series.modality}: {series.cols}×{series.rows}×{series.slice_count} - Series: {series.series_number}"
                    header_row.label(text=series_info)
                    
                    # Preview icons (5 per line)
                    try:
                        pcoll = get_preview_collection()
                        icon_ids = generate_series_preview_icons(series, patient.dicom_root_path, pcoll)
                        
                        if icon_ids:
                            icon_row = box.row(align=True)
                            for icon_id in icon_ids:
                                icon_row.template_icon(icon_value=icon_id, scale=1.5)
                    except Exception as e:
                        log.error(f"Failed to generate preview icons: {e}")
                    
                    # Preview button (always available)
                    action_row = box.row(align=True)
                    preview_text = "Preview 4D" if series.is_4d else "Preview"
                    op = action_row.operator(IMPORT_OT_dicom_preview_series.bl_idname, text=preview_text, icon='IMAGE_DATA')
                    op.series_uid = series.series_instance_uid
                    
                    box.separator()
        
        # Tool selection
        layout.separator()
        
        if scn.dicom_active_tool == 'NONE':
            # Show tool selector
            box = layout.box()
            box.label(text="Select Tool:", icon='TOOL_SETTINGS')
            
            # Detect modality from loaded series
            modality = "CT"  # default
            if patient.series:
                modality = patient.series[0].modality or "CT"
            
            # Denoising options (before visualization)
            denoise_box = box.box()
            denoise_box.label(text="Denoising", icon='SMOOTHCURVE')
            
            row = denoise_box.row()
            row.prop(scn, "denoise_enabled", text="Enable Denoising")
            
            if scn.denoise_enabled:
                row = denoise_box.row()
                row.prop(scn, "denoise_method", text="")
                
                row = denoise_box.row()
                row.prop(scn, "denoise_strength", slider=True)
                
                denoise_box.label(text="Progress shown in console", icon='INFO')
            
            # Centering options
            center_box = box.box()
            center_box.label(text="Volume Positioning", icon='OBJECT_ORIGIN')
            
            row = center_box.row()
            row.prop(scn, "center_volume_at_origin", text="Center at Origin")
            
            # Material preset selection
            preset_box = box.box()
            preset_box.label(text="Material Preset", icon='MATERIAL')
            
            row = preset_box.row()
            row.prop(scn, "dicom_material_preset", text="")
            
            box.separator()
            
            # Visualization tool (available)
            row = box.row()
            row.scale_y = 1.5
            button_text = f"Visualization {modality}"
            op = row.operator("import.dicom_set_tool", text=button_text, icon='SHADING_RENDERED')
            op.tool = 'VISUALIZATION'
        else:
            # Show active tool and change button
            box = layout.box()
            row = box.row()
            row.label(text=f"Tool: {scn.dicom_active_tool.title()}", icon='TOOL_SETTINGS')
            op = row.operator("import.dicom_set_tool", text="Change", icon='LOOP_BACK')
            op.tool = 'NONE'

class VIEW3D_PT_dicom_visualization(Panel):
    """Visualization tool panel"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOM"
    bl_label = "Visualization"
    bl_parent_id = "VIEW3D_PT_dicom_patient"
    bl_context = "objectmode"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        return (context.scene.dicom_patient_data and 
                context.scene.dicom_active_tool == 'VISUALIZATION')
    
    def draw(self, context):
        layout = self.layout
        scn = context.scene
        
        # Load patient data
        try:
            patient = Patient.from_json(scn.dicom_patient_data)
        except Exception as e:
            layout.label(text=f"Error: {e}", icon='ERROR')
            return
        
        # Tissue opacity controls (if any volume material exists)
        volume_material = None
        for mat in bpy.data.materials:
            if mat.name.endswith("_Volume_Material"):
                volume_material = mat
                break
        
        if volume_material:
            layout.separator()
            
            # Material Preset Selector
            preset_box = layout.box()
            preset_box.label(text="Material Preset:", icon='MATERIAL')
            row = preset_box.row()
            row.prop(scn, "dicom_material_preset", text="")
            
            layout.separator()
            
            box = layout.box()
            box.label(text="Tissue Opacity:", icon='SHADING_RENDERED')
            
            # Dynamically draw sliders based on tissue alphas collection
            if len(scn.dicom_tissue_alphas) > 0:
                col = box.column(align=True)
                for tissue_alpha in scn.dicom_tissue_alphas:
                    row = col.row()
                    row.label(text=tissue_alpha.tissue_label)
                    row.prop(tissue_alpha, "alpha", text="", slider=True)
            else:
                box.label(text="No tissue data loaded", icon='INFO')
            
            # Tissue Volumes section (collapsible, at bottom of opacity box)
            box.separator()
            row = box.row()
            row.prop(scn, "dicom_show_tissue_volumes",
                icon="TRIA_DOWN" if scn.dicom_show_tissue_volumes else "TRIA_RIGHT",
                icon_only=True, emboss=False
            )
            row.label(text="Tissue Volumes", icon='GRAPH')
            
            if scn.dicom_show_tissue_volumes:
                # Show tissue volumes for all loaded series
                groups = patient.get_series_by_frame_of_reference()
                has_volumes = False
                
                for frame_uid, series_list in groups.items():
                    selected_series = [s for s in series_list if s.is_selected]
                    
                    for series in selected_series:
                        if series.is_loaded and series.tissue_volumes:
                            has_volumes = True
                            col = box.column(align=True)
                            col.separator()
                            col.label(text=f"Series {series.series_number}:", icon='RENDERLAYERS')
                            
                            # Load preset to get tissue labels
                            from .presets.material_presets import load_preset
                            preset = load_preset(scn.dicom_active_material_preset)
                            
                            # Create tissue name -> label mapping
                            tissue_labels = {}
                            if preset:
                                for tissue in preset.tissues:
                                    tissue_labels[tissue['name']] = tissue.get('label', tissue['name'].title())
                            
                            # Display all measured tissue volumes
                            for tissue_name, volume_ml in series.tissue_volumes.items():
                                if volume_ml > 0:
                                    label = tissue_labels.get(tissue_name, tissue_name.title())
                                    row = col.row()
                                    row.label(text=f"  {label}: {volume_ml:.2f} mL")
                
                if not has_volumes:
                    col = box.column(align=True)
                    col.label(text="No measurements available", icon='INFO')
            
            layout.separator()
        
        # Bake bone meshes button
        # Check if there are any bone objects with geometry nodes
        has_bone_objects = False
        for obj in bpy.data.objects:
            if 'Bone' in obj.name:
                for mod in obj.modifiers:
                    if mod.type == 'NODES':
                        has_bone_objects = True
                        break
                if has_bone_objects:
                    break
        
        if has_bone_objects:
            box = layout.box()
            box.label(text="Mesh Tools:", icon='MESH_DATA')
            row = box.row()
            row.scale_y = 1.2
            row.operator("import.dicom_bake_bone_mesh", text="Bake Bone Meshes", icon='MESH_CUBE')
            layout.separator()
        
        # Tool-specific actions for each series (ONLY SELECTED SERIES)
        groups = patient.get_series_by_frame_of_reference()
        
        for frame_uid, series_list in groups.items():
            # Filter to only selected series
            selected_series = [s for s in series_list if s.is_selected]
            
            if not selected_series:
                continue  # Skip this frame if no selected series
            
            box = layout.box()
            
            # Series actions (no frame header)
            for series in selected_series:
                # Series header
                row = box.row(align=True)
                row.label(text=f"Series {series.series_number}", icon='RENDERLAYERS')

class IMAGE_EDITOR_PT_dicom_controls(Panel):
    """DICOM controls panel in Image Editor"""
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "DICOM"
    bl_label = "DICOM Navigation"
    
    @classmethod
    def poll(cls, context):
        # Only show if DICOM preview is active
        return (context.scene.dicom_preview_slice_count > 0 and
                "DICOM_Preview" in bpy.data.images)
    
    def draw(self, context):
        layout = self.layout
        scn = context.scene
        
        # Auto-select DICOM_Preview image if not already selected
        dicom_img = bpy.data.images.get("DICOM_Preview")
        if dicom_img and context.space_data.image != dicom_img:
            context.space_data.image = dicom_img
        
        import json
        series_list = json.loads(scn.dicom_series_data)
        series = None
        if scn.dicom_preview_series_index < len(series_list):
            series = series_list[scn.dicom_preview_series_index]
            layout.label(text=f"Series: {series.get('description', 'No Description')}", icon='IMAGE_DATA')
        
        # 4D Time Point Navigation (if applicable)
        if scn.dicom_preview_is_4d:
            from .operators import IMAGE_OT_dicom_scroll_time_point
            
            box = layout.box()
            box.label(text=f"Time Point: {scn.dicom_preview_time_point_index + 1} / {scn.dicom_preview_time_point_count}", icon='TIME')
            
            row = box.row(align=True)
            op = row.operator(IMAGE_OT_dicom_scroll_time_point.bl_idname, text="First", icon='REW')
            op.direction = -scn.dicom_preview_time_point_index
            
            op = row.operator(IMAGE_OT_dicom_scroll_time_point.bl_idname, text="", icon='TRIA_LEFT')
            op.direction = -1
            
            op = row.operator(IMAGE_OT_dicom_scroll_time_point.bl_idname, text="", icon='TRIA_RIGHT')
            op.direction = 1
            
            op = row.operator(IMAGE_OT_dicom_scroll_time_point.bl_idname, text="Last", icon='FF')
            op.direction = scn.dicom_preview_time_point_count - scn.dicom_preview_time_point_index - 1
        
        # Slice info
        box = layout.box()
        if scn.dicom_preview_is_4d:
            # Show slice within current time point
            slices_per_tp = scn.dicom_preview_slices_per_time_point
            current_tp = scn.dicom_preview_time_point_index
            slice_in_tp = scn.dicom_preview_slice_index - (current_tp * slices_per_tp) + 1
            box.label(text=f"Slice (in time point): {slice_in_tp} / {slices_per_tp}")
        else:
            box.label(text=f"Slice: {scn.dicom_preview_slice_index + 1} / {scn.dicom_preview_slice_count}")
        
        # Navigation buttons
        row = box.row(align=True)
        op = row.operator(IMPORT_OT_dicom_preview_slice.bl_idname, text="First", icon='REW')
        if scn.dicom_preview_is_4d:
            op.slice_index = scn.dicom_preview_time_point_index * scn.dicom_preview_slices_per_time_point
        else:
            op.slice_index = 0
        
        op = row.operator(IMPORT_OT_dicom_preview_slice.bl_idname, text="", icon='TRIA_LEFT')
        op.slice_index = max(0, scn.dicom_preview_slice_index - 1)
        
        op = row.operator(IMPORT_OT_dicom_preview_slice.bl_idname, text="", icon='TRIA_RIGHT')
        op.slice_index = min(scn.dicom_preview_slice_count - 1, scn.dicom_preview_slice_index + 1)
        
        op = row.operator(IMPORT_OT_dicom_preview_slice.bl_idname, text="Last", icon='FF')
        if scn.dicom_preview_is_4d:
            op.slice_index = (scn.dicom_preview_time_point_index + 1) * scn.dicom_preview_slices_per_time_point - 1
        else:
            op.slice_index = scn.dicom_preview_slice_count - 1
        
        # Quick jump
        if scn.dicom_preview_slice_count > 20 and not scn.dicom_preview_is_4d:
            box.label(text="Quick Jump:")
            row = box.row(align=True)
            step = max(1, scn.dicom_preview_slice_count // 10)
            for idx in range(0, scn.dicom_preview_slice_count, step):
                op = row.operator(IMPORT_OT_dicom_preview_slice.bl_idname, text=str(idx+1))
                op.slice_index = idx
        
        layout.separator()
        layout.label(text="Use mouse wheel to scroll", icon='MOUSE_MOVE')
        
        # Spatial Information (collapsible)
        if series:
            layout.separator()
            box = layout.box()
            
            # Collapsible header
            row = box.row()
            row.prop(scn, "dicom_show_spatial_info",
                icon="TRIA_DOWN" if scn.dicom_show_spatial_info else "TRIA_RIGHT",
                icon_only=True, emboss=False
            )
            row.label(text="Spatial Information", icon='ORIENTATION_GIMBAL')
            
            if scn.dicom_show_spatial_info:
                col = box.column(align=True)
                
                # Get series data
                rows = series.get('rows', 0)
                cols = series.get('cols', 0)
                slices = scn.dicom_preview_slice_count
                
                # Dimensions section
                col.separator()
                col.label(text="Dimensions:", icon='MESH_GRID')
                col.label(text=f"  Matrix: {cols}×{rows}×{slices}")
                
                # Get first file to read spacing and position
                files = series.get('files', [])
                if files:
                    from .dicom_io import load_slice
                    first_slice = load_slice(files[0])
                    last_slice = load_slice(files[-1]) if len(files) > 1 else first_slice
                    
                    if first_slice:
                        pixel_spacing = first_slice.get('pixel_spacing', (1.0, 1.0))
                        slice_thickness = first_slice.get('slice_thickness', 1.0)
                        
                        # Calculate FOV
                        fov_x = cols * pixel_spacing[1]  # pixel_spacing[1] is column spacing
                        fov_y = rows * pixel_spacing[0]  # pixel_spacing[0] is row spacing
                        fov_z = slices * slice_thickness
                        
                        col.label(text=f"  FOV: {fov_x:.1f}×{fov_y:.1f}×{fov_z:.1f} mm")
                        col.label(text=f"  Voxel: {pixel_spacing[1]:.3f}×{pixel_spacing[0]:.3f}×{slice_thickness:.3f} mm")
                        
                        # Aspect ratio
                        aspect_x = 1.0
                        aspect_y = pixel_spacing[0] / pixel_spacing[1]
                        aspect_z = slice_thickness / pixel_spacing[1]
                        col.label(text=f"  Aspect: {aspect_x:.2f}:{aspect_y:.2f}:{aspect_z:.2f}")
                        
                        # Position section
                        col.separator()
                        col.label(text="Position (Patient):", icon='EMPTY_ARROWS')
                        
                        position_first = first_slice.get('position', [0, 0, 0])
                        col.label(text=f"  First: ({position_first[0]:.1f}, {position_first[1]:.1f}, {position_first[2]:.1f}) mm")
                        
                        if last_slice and len(files) > 1:
                            position_last = last_slice.get('position', position_first)
                            col.label(text=f"  Last: ({position_last[0]:.1f}, {position_last[1]:.1f}, {position_last[2]:.1f}) mm")
                            
                            # Calculate center
                            center = [
                                (position_first[0] + position_last[0]) / 2,
                                (position_first[1] + position_last[1]) / 2,
                                (position_first[2] + position_last[2]) / 2
                            ]
                            col.label(text=f"  Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) mm")
                        
                        # Orientation section
                        col.separator()
                        col.label(text="Orientation:", icon='ORIENTATION_VIEW')
                        
                        orientation = first_slice.get('orientation', [1, 0, 0, 0, 1, 0])
                        # Determine plane from orientation
                        import numpy as np
                        row_cosines = np.array(orientation[:3])
                        col_cosines = np.array(orientation[3:])
                        normal = np.cross(row_cosines, col_cosines)
                        
                        # Check which axis the normal is closest to
                        abs_normal = np.abs(normal)
                        max_idx = np.argmax(abs_normal)
                        
                        if max_idx == 0:
                            plane = "Sagittal"
                        elif max_idx == 1:
                            plane = "Coronal"
                        else:
                            plane = "Axial"
                        
                        col.label(text=f"  Plane: {plane}")
                        col.label(text=f"  Row: [{orientation[0]:.2f}, {orientation[1]:.2f}, {orientation[2]:.2f}]")
                        col.label(text=f"  Col: [{orientation[3]:.2f}, {orientation[4]:.2f}, {orientation[5]:.2f}]")


class VIEW3D_PT_dicom_debug(Panel):
    """Debug and performance testing panel"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOM"
    bl_label = "Debug & Performance"
    bl_context = "objectmode"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Compute backend info
        try:
            from .compute_backend import backend_name, get_backend_info
            
            box = layout.box()
            box.label(text="Compute Backend", icon='SYSTEM')
            
            info = get_backend_info()
            box.label(text=f"Backend: {info['name'].upper()}")
            
            if info['gpu_accelerated']:
                box.label(text=f"Device: {info['device']}", icon='CHECKMARK')
                
                # Show MLX-specific info
                if info['name'] == 'mlx':
                    if 'architecture' in info:
                        box.label(text=f"Architecture: {info['architecture']}")
                    if 'total_memory' in info:
                        box.label(text=f"Memory: {info['total_memory']}")
                    if 'recommended_memory' in info:
                        box.label(text=f"Recommended: {info['recommended_memory']}")
                
                # Show CuPy-specific info
                elif info['name'] == 'cupy':
                    if 'memory_total' in info:
                        box.label(text=f"GPU Memory: {info['memory_free']} / {info['memory_total']}")
                    if 'compute_capability' in info:
                        box.label(text=f"Compute: {info['compute_capability']}")
            else:
                box.label(text="CPU only (no GPU)", icon='INFO')
            
            # Test button
            box.operator("dicom.test_compute_backend", text="Run Performance Test", icon='PLAY')
            
        except Exception as e:
            layout.label(text=f"Backend error: {e}", icon='ERROR')


classes = (
    VIEW3D_PT_dicom_patient,
    VIEW3D_PT_dicom_visualization,
    IMAGE_EDITOR_PT_dicom_controls,
    VIEW3D_PT_dicom_debug,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    # Clean up preview collection
    global preview_collection
    if preview_collection is not None:
        bpy.utils.previews.remove(preview_collection)
        preview_collection = None
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)