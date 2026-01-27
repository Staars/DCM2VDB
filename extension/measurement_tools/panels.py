"""Measurement panel UI"""

import bpy
from bpy.types import Panel
from ..measurement_templates import list_measurement_templates


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
        
        # Measurements list
        if len(scn.dicom_measurements) > 0:
            layout.separator()
            box = layout.box()
            box.label(text="Measurements:", icon='TRACKING')
            
            for idx, measurement in enumerate(scn.dicom_measurements):
                # Measurement header
                row = box.row(align=True)
                
                # Status icon
                if measurement.status == 'COMPLETED':
                    icon = 'CHECKMARK'
                elif measurement.status == 'IN_PROGRESS':
                    icon = 'PLUS'
                else:
                    icon = 'RADIOBUT_OFF'
                
                row.label(text="", icon=icon)
                row.label(text=measurement.label)
                
                # Capture button
                if measurement.status != 'COMPLETED':
                    op = row.operator(
                        "dicom.capture_measurement_point",
                        text="",
                        icon='CURSOR'
                    )
                    op.measurement_index = idx
                
                # Show progress
                points_captured = len(measurement.points)
                points_required = measurement.points_required
                
                sub_row = box.row()
                sub_row.label(text=f"  Points: {points_captured}/{points_required}")
                
                # Show result if completed
                if measurement.status == 'COMPLETED':
                    result_row = box.row()
                    result_row.label(text=f"  Result: {measurement.value:.2f} {measurement.unit}")
                
                box.separator()
            
            # Export button
            layout.separator()
            row = layout.row()
            row.scale_y = 1.5
            row.operator("dicom.export_measurements_csv", text="Export to CSV", icon='EXPORT')
