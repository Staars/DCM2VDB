"""Tool selection and auto-visualization operator"""

import bpy
import os
import json
from bpy.props import StringProperty
from bpy.types import Operator

from ..dicom_io import load_slice
from ..volume_creation import create_volume
from ..patient import Patient
from ..utils import SimpleLogger

log = SimpleLogger()


class IMPORT_OT_dicom_set_tool(Operator):
    """Set the active DICOM tool"""
    bl_idname = "import.dicom_set_tool"
    bl_label = "Set Tool"
    bl_options = {'REGISTER', 'UNDO'}
    
    tool: StringProperty()
    
    def execute(self, context):
        context.scene.dicom_active_tool = self.tool
        
        self.report({'INFO'}, f"Setting tool to: {self.tool}")
        self.report({'INFO'}, f"Center at origin: {context.scene.center_volume_at_origin}")
        
        # If switching to VISUALIZATION tool, auto-visualize all selected series
        if self.tool == 'VISUALIZATION' and context.scene.dicom_patient_data:
            try:
                patient = Patient.from_json(context.scene.dicom_patient_data)
                
                # Clear all previously loaded volumes and meshes
                self._clear_all_volumes(context)
                
                # Reset all series loaded states
                for s in patient.series:
                    s.is_loaded = False
                    s.is_visible = False
                
                selected_series = [s for s in patient.series if s.is_selected]
                
                log.info(f"Total series: {len(patient.series)}, Selected: {len(selected_series)}")
                for s in patient.series:
                    log.debug(f"  Series {s.series_number}: selected={s.is_selected}, loaded={s.is_loaded}")
                
                if selected_series:
                    self.report({'INFO'}, f"Auto-visualizing {len(selected_series)} selected series...")
                    
                    for series in selected_series:
                        # Check if this is a 4D series
                        is_4d = hasattr(series, 'is_4d') and series.is_4d
                        
                        if is_4d:
                            # 4D series - pass time points data
                            log.info(f"Visualizing 4D series: {series.series_description}")
                            vol_obj = create_volume(
                                slices=None,  # Not used for 4D
                                series_number=series.series_number,
                                time_points_data=series.time_points
                            )
                            
                            # Update series state
                            series.is_loaded = True
                            series.is_visible = True
                            
                            # Store volume object reference
                            patient.volume_objects[series.series_instance_uid] = vol_obj.name
                            
                            # Calculate measurements for 4D (uses first time point)
                            from ..measurements.tissue_volumes import calculate_and_store_tissue_volumes
                            calculate_and_store_tissue_volumes(context, series)
                            
                            log.info(f"Auto-visualized 4D series {series.series_number}")
                        else:
                            # Regular series
                            file_paths = [os.path.join(patient.dicom_root_path, rel_path) 
                                         for rel_path in series.file_paths]
                            
                            # Load slices
                            slices = []
                            for path in file_paths:
                                slice_data = load_slice(path)
                                if slice_data is not None:
                                    slices.append(slice_data)
                            
                            if len(slices) >= 2:
                                # Create volume
                                vol_obj = create_volume(slices, series_number=series.series_number)
                                
                                # Update series state
                                series.is_loaded = True
                                series.is_visible = True
                                
                                # Store volume object reference
                                patient.volume_objects[series.series_instance_uid] = vol_obj.name
                                
                                # Calculate measurements
                                from ..measurements.tissue_volumes import calculate_and_store_tissue_volumes
                                calculate_and_store_tissue_volumes(context, series)
                                
                                log.info(f"Auto-visualized series {series.series_number}")
                    
                    # Save updated patient data
                    context.scene.dicom_patient_data = patient.to_json()
                    
                    # Calculate and visualize centering transform if enabled
                    if context.scene.center_volume_at_origin:
                        log.info("Center at origin is ENABLED - calling centering calculation")
                        self._calculate_centering_transform(context, patient)
                    else:
                        log.info("Center at origin is DISABLED")
                    
                    self.report({'INFO'}, f"Visualized {len(selected_series)} series")
                else:
                    log.info("No selected series to visualize (already loaded)")
                    
                    # Still calculate centering if enabled, even if volumes already exist
                    if context.scene.center_volume_at_origin:
                        log.info("Center at origin is ENABLED - calling centering calculation for existing volumes")
                        self._calculate_centering_transform(context, patient)
                    
                    self.report({'INFO'}, "All selected series already loaded")
            except Exception as e:
                log.error(f"Auto-visualization error: {e}")
                import traceback
                traceback.print_exc()
        
        self.report({'INFO'}, f"Switched to {self.tool} tool")
        return {'FINISHED'}
    
    def _clear_all_volumes(self, context):
        """Clear all DICOM volumes and meshes from the scene"""
        # Remove all objects with DICOM naming patterns
        for obj in list(bpy.data.objects):
            if any(pattern in obj.name for pattern in ['_Volume_', '_Mesh_', 'DICOM_']):
                obj_name = obj.name
                bpy.data.objects.remove(obj, do_unlink=True)
                log.debug(f"Removed object: {obj_name}")
        
        # Remove volume materials
        for mat in list(bpy.data.materials):
            if mat.name.endswith('_Volume_Material'):
                mat_name = mat.name
                bpy.data.materials.remove(mat)
                log.debug(f"Removed material: {mat_name}")
        
        log.info("Cleared all DICOM volumes and meshes")
    
    def _calculate_centering_transform(self, context, patient):
        """Calculate the transform needed to center volumes at origin and create debug Empty"""
        import mathutils
        
        log.info("=" * 60)
        log.info("CALCULATING CENTERING TRANSFORM")
        log.info("=" * 60)
        
        # Step 1: Find all volume objects
        volume_objects = []
        for series_uid, vol_name in patient.volume_objects.items():
            vol_obj = bpy.data.objects.get(vol_name)
            if vol_obj and vol_obj.type == 'VOLUME':
                volume_objects.append(vol_obj)
                log.info(f"Found volume: {vol_obj.name} at location {vol_obj.location}")
        
        if not volume_objects:
            log.warning("No volume objects found for centering")
            return
        
        # Step 2: Find the volume with the lowest Z position
        lowest_vol = min(volume_objects, key=lambda v: v.location.z)
        lowest_z = lowest_vol.location.z
        
        log.info("=" * 60)
        log.info("LOWEST VOLUME DETAILS:")
        log.info(f"  Name: {lowest_vol.name}")
        log.info(f"  Location (world): {lowest_vol.location}")
        log.info(f"  Dimensions (world): {lowest_vol.dimensions}")
        log.info(f"  Rotation (euler): {lowest_vol.rotation_euler}")
        log.info(f"  Scale: {lowest_vol.scale}")
        log.info(f"  Lowest Z: {lowest_z:.4f} m")
        
        # Get volume properties
        loc = lowest_vol.location
        rot = lowest_vol.rotation_euler
        dims = lowest_vol.dimensions
        
        # Create transformation matrix manually
        mat_loc = mathutils.Matrix.Translation(loc)
        mat_rot = rot.to_matrix().to_4x4()
        mat_transform = mat_loc @ mat_rot
        
        log.info(f"  Manual transformation matrix:")
        for row in mat_transform:
            log.info(f"    {row}")
        
        # Get local bounding box corners
        local_corners = [
            mathutils.Vector((0, 0, 0)),
            mathutils.Vector((dims.x, 0, 0)),
            mathutils.Vector((0, dims.y, 0)),
            mathutils.Vector((dims.x, dims.y, 0)),
            mathutils.Vector((0, 0, dims.z)),
            mathutils.Vector((dims.x, 0, dims.z)),
            mathutils.Vector((0, dims.y, dims.z)),
            mathutils.Vector((dims.x, dims.y, dims.z)),
        ]
        
        # Transform to world space
        bbox_corners_world = [mat_transform @ corner for corner in local_corners]
        
        log.info(f"  Bounding box corners (world space after manual transformation):")
        for i, corner in enumerate(bbox_corners_world):
            log.info(f"    Corner {i}: ({corner.x:.4f}, {corner.y:.4f}, {corner.z:.4f})")
        
        # Find min/max of bounding box in world space
        bbox_min_x = min(c.x for c in bbox_corners_world)
        bbox_max_x = max(c.x for c in bbox_corners_world)
        bbox_min_y = min(c.y for c in bbox_corners_world)
        bbox_max_y = max(c.y for c in bbox_corners_world)
        bbox_min_z = min(c.z for c in bbox_corners_world)
        bbox_max_z = max(c.z for c in bbox_corners_world)
        
        log.info(f"  Bounding box X range: {bbox_min_x:.4f} to {bbox_max_x:.4f} (center: {(bbox_min_x + bbox_max_x)/2:.4f})")
        log.info(f"  Bounding box Y range: {bbox_min_y:.4f} to {bbox_max_y:.4f} (center: {(bbox_min_y + bbox_max_y)/2:.4f})")
        log.info(f"  Bounding box Z range: {bbox_min_z:.4f} to {bbox_max_z:.4f} (center: {(bbox_min_z + bbox_max_z)/2:.4f})")
        log.info(f"  Volume location Z: {lowest_vol.location.z:.4f}")
        log.info(f"  Difference (location.z - bbox_min_z): {lowest_vol.location.z - bbox_min_z:.4f}")
        log.info("=" * 60)
        
        # Step 3: Calculate the center point of the lowest plane using bounding box
        center_point = (
            (bbox_min_x + bbox_max_x) / 2.0,
            (bbox_min_y + bbox_max_y) / 2.0,
            bbox_min_z
        )
        
        log.info("=" * 60)
        log.info("CALCULATED CENTER POINT:")
        log.info(f"  Center point (from bbox): ({center_point[0]:.4f}, {center_point[1]:.4f}, {center_point[2]:.4f})")
        log.info("=" * 60)
        
        log.info(f"Calculated center point of lowest plane: {center_point}")
        
        # Step 4: Calculate the transform offset (center_point -> origin)
        transform_offset = (
            center_point[0],
            center_point[1],
            center_point[2]
        )
        
        log.info(f"Transform offset to center at origin: {transform_offset}")
        
        # Step 5: Store the transform offset in scene property
        context.scene.dicom_centering_offset = json.dumps({
            'x': transform_offset[0],
            'y': transform_offset[1],
            'z': transform_offset[2]
        })
        
        log.info("Stored centering offset in scene property")
        
        # Step 6: Apply the transform to ALL volumes and meshes
        log.info("=" * 60)
        log.info("APPLYING TRANSFORM TO VOLUMES AND MESHES")
        log.info("=" * 60)
        
        # Find all mesh objects (tissue meshes)
        mesh_objects = []
        for obj in bpy.data.objects:
            if obj.type == 'VOLUME':
                has_modality = any(pattern in obj.name for pattern in ['CT_', 'MR_', 'PET_'])
                has_series = '_S' in obj.name
                is_main_volume = '_Volume_' in obj.name
                
                if has_modality and has_series and not is_main_volume:
                    mesh_objects.append(obj)
                    log.info(f"Found mesh object: {obj.name}")
        
        # Apply transform to all volumes
        all_objects = volume_objects + mesh_objects
        log.info(f"Applying transform to {len(all_objects)} objects ({len(volume_objects)} volumes + {len(mesh_objects)} meshes)")
        
        for obj in all_objects:
            old_location = obj.location.copy()
            obj.location = (
                obj.location.x - transform_offset[0],
                obj.location.y - transform_offset[1],
                obj.location.z - transform_offset[2]
            )
            log.info(f"  {obj.type} {obj.name}: {old_location} -> {obj.location}")
        
        log.info("Transform applied successfully!")
        log.info("=" * 60)
