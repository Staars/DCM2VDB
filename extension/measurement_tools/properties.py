"""Property definitions for measurement system"""

import bpy
from bpy.props import StringProperty, FloatProperty, IntProperty, CollectionProperty, EnumProperty, FloatVectorProperty
from bpy.types import PropertyGroup


class DicomMeasurementPointProperty(PropertyGroup):
    """3D point for measurements"""
    x: FloatProperty(name="X", description="X coordinate in mm")
    y: FloatProperty(name="Y", description="Y coordinate in mm")
    z: FloatProperty(name="Z", description="Z coordinate in mm")


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
            ('angle_2d', "Angle 2D", "Angle between two lines projected to plane"),
            ('angle_3d', "Angle 3D", "True 3D angle between two lines"),
            ('hu_value', "HU Value", "Hounsfield unit at point"),
        ],
        default='distance_2d'
    )
    points_required: IntProperty(
        name="Points Required",
        default=2,
        min=1,
        max=4
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
    
    # Captured points
    points: CollectionProperty(
        type=DicomMeasurementPointProperty,
        name="Points"
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
            ('PENDING', "Pending", "Not started"),
            ('IN_PROGRESS', "In Progress", "Capturing points"),
            ('COMPLETED', "Completed", "All points captured"),
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
    
    # Collection of measurements
    bpy.types.Scene.dicom_measurements = CollectionProperty(
        type=DicomMeasurementProperty,
        name="Measurements"
    )
    
    # Active measurement index (for point capture)
    bpy.types.Scene.dicom_active_measurement_index = IntProperty(
        name="Active Measurement",
        description="Index of measurement currently being captured",
        default=-1
    )


def unregister_measurement_props():
    """Unregister measurement scene properties"""
    del bpy.types.Scene.dicom_measurement_template
    del bpy.types.Scene.dicom_measurements
    del bpy.types.Scene.dicom_active_measurement_index
