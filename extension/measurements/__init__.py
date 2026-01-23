"""Measurement system for DICOM volumes"""

import bpy
from .properties import (
    DicomLandmarkProperty,
    DicomMeasurementProperty,
    register_measurement_props,
    unregister_measurement_props
)
from .operators import (
    DICOM_OT_load_measurement_template,
    DICOM_OT_assign_landmark,
    DICOM_OT_clear_landmark,
    DICOM_OT_clear_measurements,
    DICOM_OT_export_measurements_csv
)
from .panels import VIEW3D_PT_dicom_measurements

classes = (
    DicomLandmarkProperty,
    DicomMeasurementProperty,
    DICOM_OT_load_measurement_template,
    DICOM_OT_assign_landmark,
    DICOM_OT_clear_landmark,
    DICOM_OT_clear_measurements,
    DICOM_OT_export_measurements_csv,
    VIEW3D_PT_dicom_measurements,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    register_measurement_props()

def unregister():
    unregister_measurement_props()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
