"""Operator classes for DICOM import"""

import bpy

from .import_ops import (
    IMPORT_OT_dicom_load_patient,
    IMPORT_OT_dicom_scan,
    IMPORT_OT_dicom_import_series,
    IMPORT_OT_dicom_reload_patient,
)
from .preview_ops import (
    IMPORT_OT_dicom_preview,
    IMPORT_OT_dicom_preview_popup,
    IMPORT_OT_dicom_open_in_editor,
    IMPORT_OT_dicom_preview_slice,
    IMAGE_OT_dicom_set_cursor_3d,
    IMAGE_OT_dicom_scroll,
    IMAGE_OT_dicom_scroll_time_point,
    IMPORT_OT_dicom_preview_series,
    cleanup_preview_collections,
)
from .visualization_ops import (
    IMPORT_OT_dicom_visualize_series,
    IMPORT_OT_dicom_toggle_series_selection,
    IMPORT_OT_dicom_toggle_series_visibility,
    IMPORT_OT_dicom_toggle_display_mode,
    IMPORT_OT_dicom_calculate_volume,
    IMPORT_OT_dicom_bake_bone_mesh,
)
from .tool_ops import (
    IMPORT_OT_dicom_set_tool,
)
from ..compute.test import DICOM_OT_test_compute_backend

classes = (
    IMPORT_OT_dicom_load_patient,
    IMPORT_OT_dicom_visualize_series,
    IMPORT_OT_dicom_preview_series,
    IMPORT_OT_dicom_set_tool,
    IMPORT_OT_dicom_toggle_series_selection,
    IMPORT_OT_dicom_toggle_display_mode,
    IMPORT_OT_dicom_calculate_volume,
    IMPORT_OT_dicom_reload_patient,
    IMPORT_OT_dicom_bake_bone_mesh,
    IMPORT_OT_dicom_scan,
    IMPORT_OT_dicom_preview,
    IMPORT_OT_dicom_preview_popup,
    IMPORT_OT_dicom_open_in_editor,
    IMPORT_OT_dicom_preview_slice,
    IMAGE_OT_dicom_set_cursor_3d,
    IMAGE_OT_dicom_scroll,
    IMAGE_OT_dicom_scroll_time_point,
    IMPORT_OT_dicom_import_series,
    DICOM_OT_test_compute_backend,
)

addon_keymaps = []

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Add keymap for double-click in Image Editor
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Image', space_type='IMAGE_EDITOR')
        kmi = km.keymap_items.new('image.dicom_set_cursor_3d', 'LEFTMOUSE', 'DOUBLE_CLICK')
        addon_keymaps.append((km, kmi))

def unregister():
    # Remove keymaps
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()
    
    # Clean up preview collections
    cleanup_preview_collections()
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
