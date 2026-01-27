"""Property definitions for measurement system"""

import bpy
from bpy.props import StringProperty, FloatProperty, IntProperty, CollectionProperty, EnumProperty, FloatVectorProperty
from bpy.types import PropertyGroup


class DicomLandmarkProperty(PropertyGroup):
    """Anatomical landmark with 3D position"""
    landmark_id: StringProperty(
        name="ID",
        description="Unique landmark identifier from template"
    )
    label: StringProperty(
        name="Label",
        description="Display label"
    )
    description: StringProperty(
        name="Description",
        description="Landmark description"
    )
    
    # Position in mm
    x: FloatProperty(name="X", description="X coordinate in mm")
    y: FloatProperty(name="Y", description="Y coordinate in mm")
    z: FloatProperty(name="Z", description="Z coordinate in mm")
    
    # Status
    is_placed: bpy.props.BoolProperty(
        name="Placed",
        description="Whether landmark has been placed",
        default=False
    )


class DicomMeasurementProperty(PropertyGroup):
    """Individual measurement in a protocol"""
    measurement_id: StringProperty(
        name="ID",
        description="Unique measurement identifier from template"
    )
    label: StringProperty(
        name="Label",
        description="Display label"
    )
    measurement_type: EnumProperty(
        name="Type",
        items=[
            ('distance_2d', "Distance 2D", "Distance between two points projected to plane"),
            ('distance_3d', "Distance 3D", "True 3D distance between two points"),
            ('distance_perpendicular_2d', "Distance Perpendicular 2D", "Distance between perpendiculars from two points to a reference line"),
            ('angle_2d', "Angle 2D", "Angle between two lines projected to plane"),
            ('angle_3d', "Angle 3D", "True 3D angle between two lines"),
            ('hu_value', "HU Value", "Hounsfield unit at point"),
        ],
        default='distance_2d'
    )
    projection_plane: EnumProperty(
        name="Projection Plane",
        items=[
            ('axial', "Axial", "XY plane (transverse)"),
            ('sagittal', "Sagittal", "YZ plane"),
            ('coronal', "Coronal", "XZ plane"),
        ],
        default='axial',
        description="Plane for 2D projections"
    )
    description: StringProperty(
        name="Description",
        description="Measurement description"
    )
    
    # Landmark IDs required for this measurement (stored as comma-separated string)
    landmark_ids: StringProperty(
        name="Landmark IDs",
        description="Comma-separated list of landmark IDs required",
        default=""
    )
    
    # Calculated result
    value: FloatProperty(
        name="Value",
        description="Calculated measurement value"
    )
    unit: StringProperty(
        name="Unit",
        description="Measurement unit (mm, degrees, HU)",
        default="mm"
    )
    
    # Status tracking
    status: EnumProperty(
        name="Status",
        items=[
            ('PENDING', "Pending", "Not all landmarks placed"),
            ('COMPLETED', "Completed", "All landmarks placed and calculated"),
        ],
        default='PENDING'
    )


def register_measurement_props():
    """Register measurement scene properties"""
    
    # Active measurement template
    bpy.types.Scene.dicom_measurement_template = StringProperty(
        name="Measurement Template",
        description="Active measurement protocol template",
        default=""
    )
    
    # Collection of landmarks
    bpy.types.Scene.dicom_landmarks = CollectionProperty(
        type=DicomLandmarkProperty,
        name="Landmarks"
    )
    
    # Collection of measurements
    bpy.types.Scene.dicom_measurements = CollectionProperty(
        type=DicomMeasurementProperty,
        name="Measurements"
    )
    
    # Active landmark index (for assignment)
    bpy.types.Scene.dicom_active_landmark_index = IntProperty(
        name="Active Landmark",
        description="Index of landmark to assign to cursor",
        default=-1
    )


def unregister_measurement_props():
    """Unregister measurement scene properties"""
    del bpy.types.Scene.dicom_measurement_template
    del bpy.types.Scene.dicom_landmarks
    del bpy.types.Scene.dicom_measurements
    del bpy.types.Scene.dicom_active_landmark_index
