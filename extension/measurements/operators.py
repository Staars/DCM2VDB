"""Measurement operators"""

import bpy
from bpy.types import Operator
from ..utils import SimpleLogger
from ..presets.measurement_templates import load_measurement_template

log = SimpleLogger()


class DICOM_OT_load_measurement_template(Operator):
    """Load a measurement protocol template"""
    bl_idname = "dicom.load_measurement_template"
    bl_label = "Load Measurement Template"
    bl_options = {'REGISTER', 'UNDO'}
    
    template_name: bpy.props.StringProperty(name="Template Name")
    
    def execute(self, context):
        from .visualization import clear_measurement_visualizations
        
        scn = context.scene
        
        # Clear existing visualizations
        clear_measurement_visualizations()
        
        # Load template
        template = load_measurement_template(self.template_name)
        if not template:
            self.report({'ERROR'}, f"Failed to load template: {self.template_name}")
            return {'CANCELLED'}
        
        # Clear existing measurements
        scn.dicom_measurements.clear()
        scn.dicom_measurement_template = self.template_name
        scn.dicom_active_measurement_index = -1
        
        # Create measurement entries from template
        for measurement_def in template.measurements:
            measurement = scn.dicom_measurements.add()
            measurement.measurement_id = measurement_def['id']
            measurement.label = measurement_def['label']
            measurement.measurement_type = measurement_def['type']
            measurement.points_required = measurement_def['points_required']
            measurement.description = measurement_def.get('description', '')
            measurement.projection_plane = measurement_def.get('projection_plane', 'axial')
            measurement.status = 'PENDING'
            
            # Set unit based on type
            if 'distance' in measurement.measurement_type:
                measurement.unit = 'mm'
            elif 'angle' in measurement.measurement_type:
                measurement.unit = '°'
            elif measurement.measurement_type == 'hu_value':
                measurement.unit = 'HU'
        
        log.info(f"Loaded template '{template.label}' with {len(template.measurements)} measurements")
        self.report({'INFO'}, f"Loaded {len(template.measurements)} measurements")
        
        return {'FINISHED'}


class DICOM_OT_capture_measurement_point(Operator):
    """Capture point from 3D cursor for active measurement"""
    bl_idname = "dicom.capture_measurement_point"
    bl_label = "Capture Point"
    bl_options = {'REGISTER', 'UNDO'}
    
    measurement_index: bpy.props.IntProperty(name="Measurement Index")
    
    def execute(self, context):
        scn = context.scene
        
        if self.measurement_index < 0 or self.measurement_index >= len(scn.dicom_measurements):
            self.report({'ERROR'}, "Invalid measurement index")
            return {'CANCELLED'}
        
        measurement = scn.dicom_measurements[self.measurement_index]
        
        # Check if already completed
        if measurement.status == 'COMPLETED':
            self.report({'WARNING'}, f"{measurement.label} is already completed")
            return {'CANCELLED'}
        
        # Check if we can add more points
        if len(measurement.points) >= measurement.points_required:
            self.report({'WARNING'}, f"{measurement.label} already has all required points")
            return {'CANCELLED'}
        
        # Get 3D cursor location (in meters, convert to mm)
        cursor_loc = context.scene.cursor.location
        
        # Add point
        point = measurement.points.add()
        point.x = cursor_loc.x * 1000  # Convert to mm
        point.y = cursor_loc.y * 1000
        point.z = cursor_loc.z * 1000
        
        # Update status
        if len(measurement.points) < measurement.points_required:
            measurement.status = 'IN_PROGRESS'
        else:
            measurement.status = 'COMPLETED'
            # Calculate result
            self._calculate_measurement(measurement)
        
        # Update visualization
        from .visualization import update_measurement_visualization
        update_measurement_visualization(measurement, self.measurement_index)
        
        point_num = len(measurement.points)
        log.info(f"Captured point {point_num}/{measurement.points_required} for {measurement.label}")
        self.report({'INFO'}, f"Point {point_num}/{measurement.points_required} captured")
        
        return {'FINISHED'}
    
    def _calculate_measurement(self, measurement):
        """Calculate measurement value from captured points"""
        from .calculations import calculate_distance_2d, calculate_distance_3d, calculate_angle_2d, calculate_angle_3d
        
        points = [(p.x, p.y, p.z) for p in measurement.points]
        
        if measurement.measurement_type == 'distance_2d' and len(points) == 2:
            measurement.value = calculate_distance_2d(points[0], points[1], measurement.projection_plane)
        elif measurement.measurement_type == 'distance_3d' and len(points) == 2:
            measurement.value = calculate_distance_3d(points[0], points[1])
        elif measurement.measurement_type == 'angle_2d' and len(points) == 4:
            measurement.value = calculate_angle_2d(points[0], points[1], points[2], points[3], measurement.projection_plane)
        elif measurement.measurement_type == 'angle_3d' and len(points) == 4:
            measurement.value = calculate_angle_3d(points[0], points[1], points[2], points[3])
        elif measurement.measurement_type == 'hu_value' and len(points) == 1:
            # TODO: Implement HU sampling from volume data
            measurement.value = 0.0
            log.warning("HU value sampling not yet implemented")
        
        log.info(f"Calculated {measurement.label}: {measurement.value:.2f} {measurement.unit}")


class DICOM_OT_clear_measurements(Operator):
    """Clear all measurements"""
    bl_idname = "dicom.clear_measurements"
    bl_label = "Clear Measurements"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from .visualization import clear_measurement_visualizations
        
        scn = context.scene
        
        # Clear visualizations
        clear_measurement_visualizations()
        
        # Clear measurements
        scn.dicom_measurements.clear()
        scn.dicom_measurement_template = ""
        scn.dicom_active_measurement_index = -1
        
        log.info("Cleared all measurements")
        self.report({'INFO'}, "Measurements cleared")
        
        return {'FINISHED'}


class DICOM_OT_export_measurements_csv(Operator):
    """Export measurements to CSV file"""
    bl_idname = "dicom.export_measurements_csv"
    bl_label = "Export to CSV"
    bl_options = {'REGISTER'}
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    def execute(self, context):
        from .export import export_measurements_to_csv
        
        success = export_measurements_to_csv(context, self.filepath)
        
        if success:
            self.report({'INFO'}, f"Exported to {self.filepath}")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Export failed")
            return {'CANCELLED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
