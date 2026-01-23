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
        
        # Clear existing data
        scn.dicom_landmarks.clear()
        scn.dicom_measurements.clear()
        scn.dicom_measurement_template = self.template_name
        scn.dicom_active_landmark_index = -1
        
        # Create landmark entries from template
        for landmark_def in template.landmarks:
            landmark = scn.dicom_landmarks.add()
            landmark.landmark_id = landmark_def['id']
            landmark.label = landmark_def['label']
            landmark.description = landmark_def.get('description', '')
            landmark.is_placed = False
        
        # Create measurement entries from template
        for measurement_def in template.measurements:
            measurement = scn.dicom_measurements.add()
            measurement.measurement_id = measurement_def['id']
            measurement.label = measurement_def['label']
            measurement.measurement_type = measurement_def['type']
            measurement.description = measurement_def.get('description', '')
            measurement.projection_plane = measurement_def.get('projection_plane', 'axial')
            measurement.status = 'PENDING'
            
            # Store landmark IDs as comma-separated string
            landmark_ids = measurement_def.get('landmarks', [])
            measurement.landmark_ids = ','.join(landmark_ids)
            
            # Set unit based on type
            if 'distance' in measurement.measurement_type:
                measurement.unit = 'mm'
            elif 'angle' in measurement.measurement_type:
                measurement.unit = '°'
            elif measurement.measurement_type == 'hu_value':
                measurement.unit = 'HU'
        
        log.info(f"Loaded template '{template.label}' with {len(template.landmarks)} landmarks and {len(template.measurements)} measurements")
        self.report({'INFO'}, f"Loaded {len(template.landmarks)} landmarks, {len(template.measurements)} measurements")
        
        return {'FINISHED'}


class DICOM_OT_assign_landmark(Operator):
    """Assign 3D cursor position to landmark"""
    bl_idname = "dicom.assign_landmark"
    bl_label = "Assign Landmark"
    bl_options = {'REGISTER', 'UNDO'}
    
    landmark_index: bpy.props.IntProperty(name="Landmark Index")
    
    def execute(self, context):
        scn = context.scene
        
        if self.landmark_index < 0 or self.landmark_index >= len(scn.dicom_landmarks):
            self.report({'ERROR'}, "Invalid landmark index")
            return {'CANCELLED'}
        
        landmark = scn.dicom_landmarks[self.landmark_index]
        
        # Get 3D cursor location (in meters, convert to mm)
        cursor_loc = context.scene.cursor.location
        
        # Assign position
        landmark.x = cursor_loc.x * 1000  # Convert to mm
        landmark.y = cursor_loc.y * 1000
        landmark.z = cursor_loc.z * 1000
        landmark.is_placed = True
        
        log.info(f"Assigned landmark '{landmark.label}' at ({landmark.x:.2f}, {landmark.y:.2f}, {landmark.z:.2f}) mm")
        self.report({'INFO'}, f"Assigned {landmark.label}")
        
        # Update visualization
        from .visualization import update_landmark_visualization
        update_landmark_visualization(landmark, self.landmark_index)
        
        # Recalculate all measurements that use this landmark
        self._recalculate_measurements(context, landmark.landmark_id)
        
        return {'FINISHED'}
    
    def _recalculate_measurements(self, context, landmark_id):
        """Recalculate measurements that use this landmark"""
        scn = context.scene
        
        for measurement in scn.dicom_measurements:
            # Check if this measurement uses the landmark
            required_landmark_ids = measurement.landmark_ids.split(',')
            if landmark_id not in required_landmark_ids:
                continue
            
            # Check if all required landmarks are placed
            all_placed = True
            landmark_positions = []
            
            for req_id in required_landmark_ids:
                landmark = self._find_landmark(scn, req_id)
                if not landmark or not landmark.is_placed:
                    all_placed = False
                    break
                landmark_positions.append((landmark.x, landmark.y, landmark.z))
            
            if all_placed:
                # Calculate measurement
                self._calculate_measurement(measurement, landmark_positions)
                measurement.status = 'COMPLETED'
                
                # Update visualization
                from .visualization import update_measurement_visualization
                measurement_index = list(scn.dicom_measurements).index(measurement)
                update_measurement_visualization(measurement, measurement_index)
            else:
                measurement.status = 'PENDING'
    
    def _find_landmark(self, scn, landmark_id):
        """Find landmark by ID"""
        for landmark in scn.dicom_landmarks:
            if landmark.landmark_id == landmark_id:
                return landmark
        return None
    
    def _calculate_measurement(self, measurement, points):
        """Calculate measurement value from landmark positions"""
        from .calculations import (
            calculate_distance_2d, 
            calculate_distance_3d, 
            calculate_distance_perpendicular_2d,
            calculate_angle_2d, 
            calculate_angle_3d
        )
        
        if measurement.measurement_type == 'distance_2d' and len(points) == 2:
            measurement.value = calculate_distance_2d(points[0], points[1], measurement.projection_plane)
        elif measurement.measurement_type == 'distance_3d' and len(points) == 2:
            measurement.value = calculate_distance_3d(points[0], points[1])
        elif measurement.measurement_type == 'distance_perpendicular_2d' and len(points) == 4:
            # Points order: ref_point1, ref_point2, point_a, point_b
            measurement.value = calculate_distance_perpendicular_2d(
                points[0], points[1], points[2], points[3], measurement.projection_plane
            )
        elif measurement.measurement_type == 'angle_2d' and len(points) == 4:
            measurement.value = calculate_angle_2d(points[0], points[1], points[2], points[3], measurement.projection_plane)
        elif measurement.measurement_type == 'angle_3d' and len(points) == 4:
            measurement.value = calculate_angle_3d(points[0], points[1], points[2], points[3])
        elif measurement.measurement_type == 'hu_value' and len(points) == 1:
            # TODO: Implement HU sampling from volume data
            measurement.value = 0.0
            log.warning("HU value sampling not yet implemented")
        
        log.info(f"Calculated {measurement.label}: {measurement.value:.2f} {measurement.unit}")


class DICOM_OT_clear_landmark(Operator):
    """Clear a specific landmark"""
    bl_idname = "dicom.clear_landmark"
    bl_label = "Clear Landmark"
    bl_options = {'REGISTER', 'UNDO'}
    
    landmark_index: bpy.props.IntProperty(name="Landmark Index")
    
    def execute(self, context):
        scn = context.scene
        
        if self.landmark_index < 0 or self.landmark_index >= len(scn.dicom_landmarks):
            self.report({'ERROR'}, "Invalid landmark index")
            return {'CANCELLED'}
        
        landmark = scn.dicom_landmarks[self.landmark_index]
        landmark_id = landmark.landmark_id
        
        # Clear landmark
        landmark.is_placed = False
        landmark.x = 0.0
        landmark.y = 0.0
        landmark.z = 0.0
        
        log.info(f"Cleared landmark '{landmark.label}'")
        self.report({'INFO'}, f"Cleared {landmark.label}")
        
        # Clear visualization
        from .visualization import clear_landmark_visualization
        clear_landmark_visualization(self.landmark_index)
        
        # Update measurements that use this landmark
        for measurement in scn.dicom_measurements:
            required_landmark_ids = measurement.landmark_ids.split(',')
            if landmark_id in required_landmark_ids:
                measurement.status = 'PENDING'
                measurement.value = 0.0
        
        return {'FINISHED'}


class DICOM_OT_clear_measurements(Operator):
    """Clear all measurements and landmarks"""
    bl_idname = "dicom.clear_measurements"
    bl_label = "Clear All"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from .visualization import clear_measurement_visualizations
        
        scn = context.scene
        
        # Clear visualizations
        clear_measurement_visualizations()
        
        # Clear data
        scn.dicom_landmarks.clear()
        scn.dicom_measurements.clear()
        scn.dicom_measurement_template = ""
        scn.dicom_active_landmark_index = -1
        
        log.info("Cleared all measurements and landmarks")
        self.report({'INFO'}, "Cleared all")
        
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
