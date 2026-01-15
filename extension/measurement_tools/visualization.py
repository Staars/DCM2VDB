"""Visualization helpers for measurements"""

import bpy
from ..utils import SimpleLogger

log = SimpleLogger()


def create_measurement_visualization(measurement, measurement_index):
    """Create visual objects for a measurement
    
    Args:
        measurement: DicomMeasurementProperty
        measurement_index: Index of measurement in collection
    """
    # Get or create collection for measurement objects
    collection_name = "DICOM_Measurements"
    if collection_name not in bpy.data.collections:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    else:
        collection = bpy.data.collections[collection_name]
    
    # Create empties for each point
    for idx, point in enumerate(measurement.points):
        # Convert mm to meters for Blender
        location = (point.x / 1000, point.y / 1000, point.z / 1000)
        
        # Create empty
        empty_name = f"M{measurement_index}_{measurement.measurement_id}_P{idx+1}"
        empty = bpy.data.objects.new(empty_name, None)
        empty.location = location
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.005  # 5mm radius
        
        # Color based on status
        if measurement.status == 'COMPLETED':
            empty.color = (0.0, 1.0, 0.0, 1.0)  # Green
        else:
            empty.color = (1.0, 1.0, 0.0, 1.0)  # Yellow
        
        collection.objects.link(empty)
    
    # Create lines for measurements
    if len(measurement.points) >= 2:
        _create_measurement_lines(measurement, measurement_index, collection)
    
    # Create text label for completed measurements
    if measurement.status == 'COMPLETED':
        _create_measurement_label(measurement, measurement_index, collection)
    
    log.debug(f"Created visualization for {measurement.label}")


def _create_measurement_lines(measurement, measurement_index, collection):
    """Create line objects connecting measurement points"""
    import bmesh
    
    # Convert points to Blender coordinates (mm to meters)
    points_3d = [(p.x / 1000, p.y / 1000, p.z / 1000) for p in measurement.points]
    
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
    
    # Color based on status
    if measurement.status == 'COMPLETED':
        obj.color = (0.0, 1.0, 0.0, 1.0)  # Green
    else:
        obj.color = (1.0, 1.0, 0.0, 1.0)  # Yellow
    
    collection.objects.link(obj)


def _create_measurement_label(measurement, measurement_index, collection):
    """Create text label showing measurement value"""
    # Calculate center position for label
    points_3d = [(p.x / 1000, p.y / 1000, p.z / 1000) for p in measurement.points]
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
    text_obj.color = (0.0, 1.0, 0.0, 1.0)  # Green
    
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
    """Update visualization for a specific measurement
    
    Removes old visualization and creates new one
    """
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
