"""UI panel definitions"""

import bpy
from bpy.types import Panel
from pathlib import Path

from .operators import (
    IMPORT_OT_dicom_scan,
    IMPORT_OT_dicom_preview,
    IMPORT_OT_dicom_preview_slice,
    IMPORT_OT_dicom_import_series
)

class VIEW3D_PT_dicom_importer(Panel):
    """Main DICOM importer panel"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOM"
    bl_label = "DICOM Import"
    bl_context = "objectmode"
    
    def draw(self, context):
        layout = self.layout
        scn = context.scene
        
        # Step 1: Scan folder
        box = layout.box()
        box.label(text="1. Scan Folder", icon='FILEBROWSER')
        box.operator(IMPORT_OT_dicom_scan.bl_idname, text="Select DICOM Folder", icon='FILE_FOLDER')
        
        if scn.get("dicom_import_folder"):
            box.label(text=f"Folder: {Path(scn.dicom_import_folder).name}", icon='CHECKMARK')
        
        # Step 2: Show series list
        if len(scn.dicom_series_collection) > 0:
            box = layout.box()
            box.label(text="2. Select Series", icon='RENDERLAYERS')
            
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
                op = col.operator(IMPORT_OT_dicom_preview.bl_idname, text="", icon='IMAGE_DATA')
                op.series_index = i
                
                op = col.operator(IMPORT_OT_dicom_import_series.bl_idname, text="", icon='IMPORT')
                op.series_index = i
            
            # Step 3: Preview controls
            if scn.dicom_preview_slice_count > 0:
                box = layout.box()
                box.label(text="Preview Controls", icon='IMAGE_DATA')
                
                series_list = eval(scn.dicom_series_data)
                if scn.dicom_preview_series_index < len(series_list):
                    series = series_list[scn.dicom_preview_series_index]
                    box.label(text=f"Series: {series['description']}")
                
                row = box.row()
                row.label(text=f"Slice: {scn.dicom_preview_slice_index + 1} / {scn.dicom_preview_slice_count}")
                
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
                
                # Quick jump buttons (every 10 slices)
                if scn.dicom_preview_slice_count > 20:
                    row = box.row(align=True)
                    row.label(text="Jump to:")
                    step = max(1, scn.dicom_preview_slice_count // 10)
                    for idx in range(0, scn.dicom_preview_slice_count, step):
                        op = row.operator(IMPORT_OT_dicom_preview_slice.bl_idname, text=str(idx+1))
                        op.slice_index = idx

classes = (
    VIEW3D_PT_dicom_importer,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)