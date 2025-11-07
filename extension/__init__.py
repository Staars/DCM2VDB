# ##### BEGIN GPL LICENSE BLOCK #####
bl_info = {
    "name": "Import DICOM Volume (pydicom)",
    "author": "me",
    "version": (3, 2, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > DICOM | File > Import > DICOM Volume",
    "description": "Import CT/MR volumes with interactive preview",
    "category": "Import-Export",
}

import bpy

# Check for pydicom availability
try:
    from pydicom import dcmread
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

if "bpy" in locals():
    import importlib
    if "properties" in locals():
        importlib.reload(properties)
    if "operators" in locals():
        importlib.reload(operators)
    if "panels" in locals():
        importlib.reload(panels)
    if "dicom_io" in locals():
        importlib.reload(dicom_io)
    if "volume" in locals():
        importlib.reload(volume)
    if "preview" in locals():
        importlib.reload(preview)

from . import properties
from . import operators
from . import panels

# Keymap for mouse wheel scrolling in Image Editor
addon_keymaps = []

def register():
    # Register all classes
    properties.register()
    operators.register()
    panels.register()
    
    # Add keymap for scrolling
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Image', space_type='IMAGE_EDITOR')
        
        # Mouse wheel up = previous slice
        kmi = km.keymap_items.new(operators.IMAGE_OT_dicom_scroll.bl_idname, 'WHEELUPMOUSE', 'PRESS')
        kmi.properties.direction = -1
        addon_keymaps.append((km, kmi))
        
        # Mouse wheel down = next slice
        kmi = km.keymap_items.new(operators.IMAGE_OT_dicom_scroll.bl_idname, 'WHEELDOWNMOUSE', 'PRESS')
        kmi.properties.direction = 1
        addon_keymaps.append((km, kmi))

def unregister():
    # Remove keymaps
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()
    
    # Unregister all classes
    panels.unregister()
    operators.unregister()
    properties.unregister()

if __name__ == "__main__":
    register()