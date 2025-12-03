"""Volume creation and OpenVDB handling - Backward compatibility wrapper"""

# Import from new modular structure
from .volume_creation import create_volume
from .materials import create_volume_material
from .geometry_nodes import create_tissue_mesh_geonodes
from .volume_utils import clean_old_volumes

# Re-export for backward compatibility
__all__ = [
    'create_volume',
    'create_volume_material',
    'create_mesh_material',
    'create_tissue_mesh_geonodes',
    'clean_temp_dir',
]
