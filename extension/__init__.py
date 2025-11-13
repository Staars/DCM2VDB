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

# Install bundled wheels if pydicom is not available
def ensure_pydicom():
    """Install pydicom from bundled wheel if not already available"""
    try:
        import pydicom
        return True
    except ImportError:
        pass
    
    # Get the wheels directory path
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    wheels_dir = os.path.join(addon_dir, "wheels")
    
    if not os.path.exists(wheels_dir):
        print(f"[DICOM] Wheels directory not found: {wheels_dir}")
        return False
    
    # Find the pydicom wheel
    wheel_files = [f for f in os.listdir(wheels_dir) if f.startswith("pydicom") and f.endswith(".whl")]
    
    if not wheel_files:
        print("[DICOM] No pydicom wheel found in wheels directory")
        return False
    
    wheel_path = os.path.join(wheels_dir, wheel_files[0])
    print(f"[DICOM] Installing pydicom from: {wheel_path}")
    
    # Use Blender's Python to install the wheel
    python_exe = sys.executable
    
    try:
        subprocess.check_call([python_exe, "-m", "pip", "install", "--no-deps", wheel_path])
        print("[DICOM] Successfully installed pydicom")
        
        # Try importing again
        import pydicom
        return True
    except subprocess.CalledProcessError as e:
        print(f"[DICOM] Failed to install pydicom: {e}")
        return False
    except ImportError:
        print("[DICOM] pydicom installed but import still failed")
        return False

# Install pydicom on module load
PYDICOM_AVAILABLE = ensure_pydicom()

# Check for pydicom availability after installation attempt
if PYDICOM_AVAILABLE:
    try:
        from pydicom import dcmread
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