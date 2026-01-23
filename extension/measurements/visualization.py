"""Visualization helpers for measurements"""

import bpy
from ..utils import SimpleLogger

log = SimpleLogger()


def create_landmark_visualization(landmark, landmark_index):
    """Create visual object for a landmark
    
    Args:
        landmark: DicomLandmarkProperty
        landmark_index: Index of landmark in collection
    """
    # Get or create collection for measurement objects
    collection_name = "DICOM_Measurements"
    if collection_name not in bpy.data.collections:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    else:
        collection = bpy.data.collections[collection_name]
    
    # Convert mm to meters for Blender
    location = (landmark.x / 1000, landmark.y / 1000, landmark.z / 1000)
    
    # Create empty with descriptive name
    empty_name = f"{landmark.label}"
    empty = bpy.data.objects.new(empty_name, None)
    empty.location = location
    empty.empty_display_type = 'SPHERE'
    empty.empty_display_size = 0.005  # 5mm radius
    empty.color = (0.0, 1.0, 0.0, 1.0)  # Green
    empty.show_in_front = True
    
    collection.objects.link(empty)
    
    log.debug(f"Created visualization for landmark {landmark.label}")


def clear_landmark_visualization(landmark_index):
    """Remove visualization for a specific landmark"""
    collection_name = "DICOM_Measurements"
    if collection_name not in bpy.data.collections:
        return
    
    collection = bpy.data.collections[collection_name]
    scn = bpy.context.scene
    
    # Get the landmark to find its label
    if landmark_index < 0 or landmark_index >= len(scn.dicom_landmarks):
        return
    
    landmark = scn.dicom_landmarks[landmark_index]
    landmark_name = landmark.label
    
    # Remove object with this name
    for obj in list(collection.objects):
        if obj.name == landmark_name:
            bpy.data.objects.remove(obj, do_unlink=True)


def update_landmark_visualization(landmark, landmark_index):
    """Update visualization for a specific landmark"""
    # Remove old visualization
    clear_landmark_visualization(landmark_index)
    
    # Create new visualization
    create_landmark_visualization(landmark, landmark_index)


def create_measurement_visualization(measurement, measurement_index):
    """Create visual objects for a measurement (lines between landmarks)
    
    Args:
        measurement: DicomMeasurementProperty
        measurement_index: Index of measurement in collection
    """
    if measurement.status != 'COMPLETED':
        return
    
    # Get or create collection
    collection_name = "DICOM_Measurements"
    if collection_name not in bpy.data.collections:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    else:
        collection = bpy.data.collections[collection_name]
    
    # Get landmark positions
    scn = bpy.context.scene
    landmark_ids = measurement.landmark_ids.split(',')
    points_3d = []
    
    for lm_id in landmark_ids:
        for landmark in scn.dicom_landmarks:
            if landmark.landmark_id == lm_id and landmark.is_placed:
                # Convert mm to meters
                points_3d.append((landmark.x / 1000, landmark.y / 1000, landmark.z / 1000))
                break
    
    if len(points_3d) < 2:
        return
    
    # Create lines
    _create_measurement_lines(measurement, measurement_index, collection, points_3d)
    
    # Create result label
    _create_measurement_label(measurement, measurement_index, collection, points_3d)
    
    log.debug(f"Created visualization for measurement {measurement.label}")


def _create_measurement_lines(measurement, measurement_index, collection, points_3d):
    """Create line objects connecting measurement points"""
    import bmesh
    
    # Create mesh for lines
    mesh = bpy.data.meshes.new(f"M{measurement_index}_{measurement.measurement_id}_Lines")
    obj = bpy.data.objects.new(mesh.name, mesh)
    
    bm = bmesh.new()
    
    # Create vertices
    verts = [bm.verts.new(p) for p in points_3d]
    
    # Create edges based on measurement type
    if 'distance' in measurement.measurement_type and len(verts) == 2:
        # Single line for distance
        bm.edges.new([verts[0], verts[1]])
    elif 'angle' in measurement.measurement_type and len(verts) == 4:
        # Two lines for angle
        bm.edges.new([verts[0], verts[1]])  # Line 1
        bm.edges.new([verts[2], verts[3]])  # Line 2
    
    bm.to_mesh(mesh)
    bm.free()
    
    # Set display properties
    obj.display_type = 'WIRE'
    obj.show_in_front = True
    obj.color = (1.0, 0.5, 0.0, 1.0)  # Orange for measurement lines
    
    collection.objects.link(obj)


def _create_measurement_label(measurement, measurement_index, collection, points_3d):
    """Create text label showing measurement value"""
    # Calculate center position for label
    center = tuple(sum(coords) / len(points_3d) for coords in zip(*points_3d))
    
    # Create text curve
    curve_data = bpy.data.curves.new(
        name=f"M{measurement_index}_{measurement.measurement_id}_Label",
        type='FONT'
    )
    curve_data.body = f"{measurement.label}\n{measurement.value:.2f} {measurement.unit}"
    curve_data.size = 0.01  # 10mm text size
    curve_data.align_x = 'CENTER'
    curve_data.align_y = 'CENTER'
    
    # Create object
    text_obj = bpy.data.objects.new(curve_data.name, curve_data)
    text_obj.location = center
    text_obj.show_in_front = True
    text_obj.color = (1.0, 0.5, 0.0, 1.0)  # Orange
    
    collection.objects.link(text_obj)


def clear_measurement_visualizations():
    """Remove all measurement visualization objects"""
    collection_name = "DICOM_Measurements"
    
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
        
        # Remove all objects in collection
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # Remove collection
        bpy.data.collections.remove(collection)
        
        log.info("Cleared measurement visualizations")


def update_measurement_visualization(measurement, measurement_index):
    """Update visualization for a specific measurement"""
    # Remove old objects for this measurement
    collection_name = "DICOM_Measurements"
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
        prefix = f"M{measurement_index}_{measurement.measurement_id}"
        
        for obj in list(collection.objects):
            if obj.name.startswith(prefix):
                bpy.data.objects.remove(obj, do_unlink=True)
    
    # Create new visualization
    create_measurement_visualization(measurement, measurement_index)
