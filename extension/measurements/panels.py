"""Measurement panel UI"""

import bpy
from bpy.types import Panel
from ..presets.measurement_templates import list_measurement_templates


class VIEW3D_PT_dicom_measurements(Panel):
    """DICOM Measurements panel in 3D View sidebar"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOM"
    bl_label = "DICOM Measurements"
    bl_context = "objectmode"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        # Only show if patient is loaded
        return context.scene.dicom_patient_data
    
    def draw(self, context):
        layout = self.layout
        scn = context.scene
        
        # Template selection
        box = layout.box()
        box.label(text="Measurement Protocol:", icon='PRESET')
        
        # Get available templates
        templates = list_measurement_templates()
        
        if not templates:
            box.label(text="No templates available", icon='INFO')
        else:
            # Template buttons
            for template_name, template_label, template_desc in templates:
                row = box.row()
                op = row.operator(
                    "dicom.load_measurement_template",
                    text=template_label,
                    icon='IMPORT'
                )
                op.template_name = template_name
        
        # Show loaded template info
        if scn.dicom_measurement_template:
            layout.separator()
            box = layout.box()
            box.label(text=f"Active: {scn.dicom_measurement_template}", icon='CHECKMARK')
            
            # Clear button
            row = box.row()
            row.operator("dicom.clear_measurements", text="Clear All", icon='X')
        
        # Landmarks list
        if len(scn.dicom_landmarks) > 0:
            layout.separator()
            box = layout.box()
            box.label(text="Anatomical Landmarks:", icon='EMPTY_AXIS')
            
            for idx, landmark in enumerate(scn.dicom_landmarks):
                # Landmark row
                row = box.row(align=True)
                
                # Status icon
                if landmark.is_placed:
                    icon = 'CHECKMARK'
                else:
                    icon = 'RADIOBUT_OFF'
                
                row.label(text="", icon=icon)
                row.label(text=landmark.label)
                
                # Assign button
                if not landmark.is_placed:
                    op = row.operator(
                        "dicom.assign_landmark",
                        text="",
                        icon='CURSOR'
                    )
                    op.landmark_index = idx
                else:
                    # Clear button
                    op = row.operator(
                        "dicom.clear_landmark",
                        text="",
                        icon='X'
                    )
                    op.landmark_index = idx
                
                # Show coordinates if placed
                if landmark.is_placed:
                    coord_row = box.row()
                    coord_row.label(text=f"  ({landmark.x:.1f}, {landmark.y:.1f}, {landmark.z:.1f}) mm")
            
            # Measurements results
            if len(scn.dicom_measurements) > 0:
                layout.separator()
                box = layout.box()
                box.label(text="Measurements:", icon='TRACKING')
                
                for measurement in scn.dicom_measurements:
                    row = box.row()
                    
                    # Status icon
                    if measurement.status == 'COMPLETED':
                        icon = 'CHECKMARK'
                        row.label(text="", icon=icon)
                        row.label(text=measurement.label)
                        
                        # Show result
                        result_row = box.row()
                        result_row.label(text=f"  {measurement.value:.2f} {measurement.unit}")
                    else:
                        icon = 'RADIOBUT_OFF'
                        row.label(text="", icon=icon)
                        row.label(text=measurement.label)
                        
                        # Show which landmarks are needed
                        landmark_ids = measurement.landmark_ids.split(',')
                        placed_count = sum(1 for lm_id in landmark_ids 
                                         if any(lm.landmark_id == lm_id and lm.is_placed 
                                               for lm in scn.dicom_landmarks))
                        
                        status_row = box.row()
                        status_row.label(text=f"  Landmarks: {placed_count}/{len(landmark_ids)}")
                    
                    box.separator()
                
                # Export button
                layout.separator()
                row = layout.row()
                row.scale_y = 1.5
                row.operator("dicom.export_measurements_csv", text="Export to CSV", icon='EXPORT')
