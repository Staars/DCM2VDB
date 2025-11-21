"""UI panel definitions"""

import bpy
from bpy.types import Panel
from pathlib import Path

import os
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
        
        box.operator(IMPORT_OT_dicom_load_patient.bl_idname, text="Reload", icon='FILE_REFRESH')
        
        # Series list (collapsible) - always visible
        layout.separator()
        box = layout.box()
        
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
                        print(f"[DICOM Panel] Failed to generate preview icons: {e}")
                    
                    # Preview button (always available)
                    action_row = box.row(align=True)
                    op = action_row.operator(IMPORT_OT_dicom_preview_series.bl_idname, text="Preview", icon='IMAGE_DATA')
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
            
            # Visualization tool (available)
            row = box.row()
            row.scale_y = 1.5
            button_text = f"Visualization {modality}"
            op = row.operator("import.dicom_set_tool", text=button_text, icon='SHADING_RENDERED')
            op.tool = 'VISUALIZATION'
            
            # Future tools (disabled)
            box.label(text="Coming Soon:", icon='TIME')
            col = box.column(align=True)
            col.enabled = False
            col.operator("import.dicom_set_tool", text="Measurement", icon='DRIVER_DISTANCE')
            col.operator("import.dicom_set_tool", text="Segmentation", icon='MOD_MASK')
            col.operator("import.dicom_set_tool", text="Registration", icon='ORIENTATION_GIMBAL')
            col.operator("import.dicom_set_tool", text="Export", icon='EXPORT')
            col.operator("import.dicom_set_tool", text="Analysis", icon='GRAPH')
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
        
        # Global display mode toggle for all loaded volumes
        loaded_volumes = [obj_name for obj_name in patient.volume_objects.values() 
                         if bpy.data.objects.get(obj_name)]
        
        if loaded_volumes:
            box = layout.box()
            box.label(text="Display Mode:", icon='SCENE_DATA')
            
            # Volume visibility checkbox
            row = box.row()
            row.scale_y = 1.2
            row.prop(scn, "dicom_show_volume", text="Volume", icon='VOLUME_DATA', toggle=True)
            
            # Bone visibility checkbox
            row = box.row()
            row.scale_y = 1.2
            row.prop(scn, "dicom_show_bone", text="Bone", icon='MESH_DATA', toggle=True)
            
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
                row = box.row()
                row.label(text=f"Series {series.series_number}")
                
                # Visualize button (tool-specific action)
                op = row.operator(IMPORT_OT_dicom_visualize_series.bl_idname, text="Visualize", icon='PLAY')
                op.series_uid = series.series_instance_uid
                
                # Show measurements if loaded
                if series.is_loaded:
                    # Show volume measurements (per-series)
                    if series.fat_volume_ml > 0 or series.fluid_volume_ml > 0 or series.soft_volume_ml > 0:
                        col = box.column(align=True)
                        col.separator()
                        col.label(text="Tissue Volumes:", icon='GRAPH')
                        
                        # Fat
                        if series.fat_volume_ml > 0:
                            row = col.row()
                            row.label(text=f"  Fat: {series.fat_volume_ml:.2f} mL")
                        
                        # Fluid
                        if series.fluid_volume_ml > 0:
                            row = col.row()
                            row.label(text=f"  Fluid: {series.fluid_volume_ml:.2f} mL")
                        
                        # Soft tissue
                        if series.soft_volume_ml > 0:
                            row = col.row()
                            row.label(text=f"  Soft Tissue: {series.soft_volume_ml:.2f} mL")

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
        
        series_list = eval(scn.dicom_series_data)
        if scn.dicom_preview_series_index < len(series_list):
            series = series_list[scn.dicom_preview_series_index]
            layout.label(text=f"Series: {series.get('description', 'No Description')}", icon='IMAGE_DATA')
        
        # Slice info
        box = layout.box()
        box.label(text=f"Slice: {scn.dicom_preview_slice_index + 1} / {scn.dicom_preview_slice_count}")
        
        # Navigation buttons
        row = box.row(align=True)
        op = row.operator(IMPORT_OT_dicom_preview_slice.bl_idname, text="First", icon='REW')
        op.slice_index = 0
        
        op = row.operator(IMPORT_OT_dicom_preview_slice.bl_idname, text="", icon='TRIA_LEFT')
        op.slice_index = max(0, scn.dicom_preview_slice_index - 1)
        
        op = row.operator(IMPORT_OT_dicom_preview_slice.bl_idname, text="", icon='TRIA_RIGHT')
        op.slice_index = min(scn.dicom_preview_slice_count - 1, scn.dicom_preview_slice_index + 1)
        
        op = row.operator(IMPORT_OT_dicom_preview_slice.bl_idname, text="Last", icon='FF')
        op.slice_index = scn.dicom_preview_slice_count - 1
        
        # Quick jump
        if scn.dicom_preview_slice_count > 20:
            box.label(text="Quick Jump:")
            row = box.row(align=True)
            step = max(1, scn.dicom_preview_slice_count // 10)
            for idx in range(0, scn.dicom_preview_slice_count, step):
                op = row.operator(IMPORT_OT_dicom_preview_slice.bl_idname, text=str(idx+1))
                op.slice_index = idx
        
        layout.separator()
        layout.label(text="Use mouse wheel to scroll", icon='MOUSE_MOVE')

classes = (
    VIEW3D_PT_dicom_patient,
    VIEW3D_PT_dicom_visualization,
    IMAGE_EDITOR_PT_dicom_controls,
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