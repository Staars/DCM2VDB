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
import sys
import os
import subprocess
from .utils import SimpleLogger

# Get logger for this extension
log = SimpleLogger()

# Check for package availability
PYDICOM_AVAILABLE = False
SCIPY_AVAILABLE = False

try:
    from pydicom import dcmread
    PYDICOM_AVAILABLE = True
except ImportError:
    log.error("pydicom not available")

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    log.error("scipy not available")

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
    if "constants" in locals():
        importlib.reload(constants)
    if "volume_utils" in locals():
        importlib.reload(volume_utils)
    if "materials" in locals():
        importlib.reload(materials)
    if "geometry_nodes" in locals():
        importlib.reload(geometry_nodes)
    if "volume_creation" in locals():
        importlib.reload(volume_creation)
    if "volume" in locals():
        importlib.reload(volume)
    if "preview" in locals():
        importlib.reload(preview)
    if "measurements" in locals():
        importlib.reload(measurements)

from . import properties
from . import operators
from . import panels
from . import measurements

# Initialize compute backend and log info
try:
    from . import compute_backend
    backend_info = compute_backend.get_backend_info()
    log.info(f"Compute backend: {backend_info['name']}")
    if backend_info['gpu_accelerated']:
        log.info(f"  Device: {backend_info['device']}")
    else:
        log.info(f"  Running on CPU (install MLX or CuPy for GPU acceleration)")
except Exception as e:
    log.warning(f"Failed to initialize compute backend: {e}")

# Keymap for mouse wheel scrolling in Image Editor
addon_keymaps = []

def register():
    # Register all classes
    properties.register()
    operators.register()
    panels.register()
    measurements.register()
    
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
    measurements.unregister()
    panels.unregister()
    operators.unregister()
    properties.unregister()

if __name__ == "__main__":
    register()