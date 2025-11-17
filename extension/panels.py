"""UI panel definitions"""

import bpy
from bpy.types import Panel
from pathlib import Path

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
        
        # Patient info
        box = layout.box()
        box.label(text=f"Patient: {patient.patient_name}", icon='USER')
        if patient.patient_id:
            box.label(text=f"ID: {patient.patient_id}")
        if patient.study_description:
            box.label(text=f"Study: {patient.study_description}")
        if patient.study_date:
            box.label(text=f"Date: {patient.study_date}")
        
        # Load summary
        box = layout.box()
        box.label(text=f"✓ {len(patient.series)} primary series", icon='CHECKMARK')
        if patient.secondary_count > 0:
            box.label(text=f"ℹ {patient.secondary_count} secondary ignored")
        if patient.non_image_count > 0:
            box.label(text=f"ℹ {patient.non_image_count} non-image ignored")
        
        # Actions
        row = box.row(align=True)
        row.operator(IMPORT_OT_dicom_load_patient.bl_idname, text="Reload", icon='FILE_REFRESH')
        
        # Series list grouped by FrameOfReferenceUID
        layout.separator()
        layout.label(text="Primary Series:", icon='RENDERLAYERS')
        
        groups = patient.get_series_by_frame_of_reference()
        
        for frame_uid, series_list in groups.items():
            box = layout.box()
            
            # Frame of reference header
            if frame_uid != "unknown":
                box.label(text=f"Frame: ...{frame_uid[-8:]}", icon='EMPTY_AXIS')
            else:
                box.label(text="Frame: Unknown", icon='QUESTION')
            
            # Series in this frame
            for series in series_list:
                # Series header
                row = box.row()
                col = row.column()
                col.label(text=f"{series.series_description} ({series.modality})")
                col.label(text=f"  {series.cols}×{series.rows}×{series.slice_count}")
                
                # Preview icons (5 per line)
                try:
                    pcoll = get_preview_collection()
                    icon_ids = generate_series_preview_icons(series, patient.dicom_root_path, pcoll)
                    
                    print(f"[DICOM Panel] Generated {len(icon_ids)} icons for {series.series_description}")
                    
                    if icon_ids:
                        icon_row = box.row(align=True)
                        for icon_id in icon_ids:
                            icon_row.template_icon(icon_value=icon_id, scale=1.5)
                    else:
                        box.label(text="No preview icons generated")
                except Exception as e:
                    print(f"[DICOM Panel] Failed to generate preview icons: {e}")
                    import traceback
                    traceback.print_exc()
                    box.label(text=f"Preview error: {e}")
                
                # Actions
                action_row = box.row(align=True)
                
                # Always show Visualize button (can reload/switch volumes)
                op = action_row.operator(IMPORT_OT_dicom_visualize_series.bl_idname, text="Visualize", icon='PLAY')
                op.series_uid = series.series_instance_uid
                
                # Preview button
                op = action_row.operator(IMPORT_OT_dicom_preview_series.bl_idname, text="Preview", icon='IMAGE_DATA')
                op.series_uid = series.series_instance_uid
                
                # Show loaded status
                if series.is_loaded:
                    action_row.label(text="✓", icon='CHECKMARK')
                
                box.separator()

class VIEW3D_PT_dicom_importer(Panel):
    """Legacy DICOM importer panel (old workflow)"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOM"
    bl_label = "DICOM Import (Legacy)"
    bl_context = "objectmode"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        scn = context.scene
        
        # Step 1: Scan folder
        box = layout.box()
        box.label(text="1. Scan Folder", icon='FILEBROWSER')
        box.operator(IMPORT_OT_dicom_scan.bl_idname, text="Select DICOM Folder", icon='FILE_FOLDER')
        
        if scn.get("dicom_import_folder"):
            box.label(text=f"Folder: {Path(scn.dicom_import_folder).name}", icon='CHECKMARK')
        
        # Step 2: Show series list (collapsible)
        if len(scn.dicom_series_collection) > 0:
            box = layout.box()
            
            # Header with collapse/expand icon
            row = box.row()
            row.prop(scn, "dicom_show_series_list",
                icon="TRIA_DOWN" if scn.dicom_show_series_list else "TRIA_RIGHT",
                icon_only=True, emboss=False
            )
            row.label(text="2. Select Series", icon='RENDERLAYERS')
            row.label(text=f"({len(scn.dicom_series_collection)} found)")
            
            # Only show series list if expanded
            if scn.dicom_show_series_list:
                for i, series in enumerate(scn.dicom_series_collection):
                    row = box.row()
                    col = row.column()
                    col.label(text=f"{series.description} ({series.modality})")
                    
                    # Show dimensions and window info
                    info_text = f"  {series.cols}×{series.rows}×{series.instance_count}"
                    if series.window_width and series.window_width > 0:
                        info_text += f" | W/L: {series.window_center:.0f}/{series.window_width:.0f} HU"
                    col.label(text=info_text)
                    
                    col = row.column()
                    op = col.operator(IMPORT_OT_dicom_preview_popup.bl_idname, text="", icon='IMAGE_DATA', emboss=False)
                    op.series_index = i
                    
                    op = col.operator(IMPORT_OT_dicom_import_series.bl_idname, text="", icon='IMPORT')
                    op.series_index = i
            
            # Debug option
            box.prop(scn, "dicom_debug_pyramid", text="Debug: Create pyramid with import")

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
                "DICOM_Preview" in bpy.data.images and
                context.space_data.image == bpy.data.images.get("DICOM_Preview"))
    
    def draw(self, context):
        layout = self.layout
        scn = context.scene
        
        series_list = eval(scn.dicom_series_data)
        if scn.dicom_preview_series_index < len(series_list):
            series = series_list[scn.dicom_preview_series_index]
            layout.label(text=f"Series: {series['description']}", icon='IMAGE_DATA')
        
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
    VIEW3D_PT_dicom_importer,
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