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

# Install bundled wheels if packages are not available
def ensure_package(package_name, import_name=None):
    """Install package from bundled wheel if not already available"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        pass
    
    # Get the wheels directory path
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    wheels_dir = os.path.join(addon_dir, "wheels")
    
    if not os.path.exists(wheels_dir):
        log.warning(f"Wheels directory not found: {wheels_dir}")
        return False
    
    # Find the wheel for this package
    wheel_files = [f for f in os.listdir(wheels_dir) if f.startswith(package_name) and f.endswith(".whl")]
    
    if not wheel_files:
        log.warning(f"No {package_name} wheel found in wheels directory")
        return False
    
    wheel_path = os.path.join(wheels_dir, wheel_files[0])
    log.info(f"Installing {package_name} from: {wheel_path}")
    
    # Use Blender's Python to install the wheel
    python_exe = sys.executable
    
    try:
        subprocess.check_call([python_exe, "-m", "pip", "install", "--no-deps", wheel_path])
        log.info(f"Successfully installed {package_name}")
        
        # Try importing again
        __import__(import_name)
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to install {package_name}: {e}")
        return False
    except ImportError:
        log.error(f"{package_name} installed but import still failed")
        return False

# Install required packages on module load
PYDICOM_AVAILABLE = ensure_package("pydicom")
SCIPY_AVAILABLE = ensure_package("scipy")

# Check for package availability after installation attempt
if PYDICOM_AVAILABLE:
    try:
        from pydicom import dcmread
    except ImportError:
        PYDICOM_AVAILABLE = False

if SCIPY_AVAILABLE:
    try:
        from scipy import ndimage
    except ImportError:
        SCIPY_AVAILABLE = False
        log.warning("scipy installed but import failed")

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