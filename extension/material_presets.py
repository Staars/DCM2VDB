"""Material preset system for volume rendering"""

import json
import os
from .utils import SimpleLogger

log = SimpleLogger()


class MaterialPreset:
    """Represents a volume material preset"""
    
    def __init__(self, data):
        # Header section
        header = data.get("header", {})
        self.name = header.get("name", data.get("name", "Unknown"))
        self.description = header.get("description", data.get("description", ""))
        self.modality = header.get("modality", data.get("modality", "CT"))
        self.version = header.get("version", data.get("version", "1.0"))
        
        # Volume section
        volume = data.get("volume", data)  # Fallback to root for backward compatibility
        
        # HU range
        hu_range = volume.get("hu_range", {})
        self.hu_min = hu_range.get("min", -1024)
        self.hu_max = hu_range.get("max", 3071)
        
        # Rendering settings
        self.air_threshold = volume.get("air_threshold", -200)
        self.density_multiplier = volume.get("density_multiplier", 600)
        
        # Tissues (sorted by order)
        tissues_data = volume.get("tissues", [])
        self.tissues = sorted(tissues_data, key=lambda t: t.get("order", 0))
        
        # Mesh section
        self.meshes = data.get("mesh", [])
    
    def get_tissue(self, name):
        """Get tissue by name"""
        for tissue in self.tissues:
            if tissue["name"] == name:
                return tissue
        return None
    
    def get_mesh(self, name):
        """Get mesh definition by name"""
        for mesh in self.meshes:
            if mesh["name"] == name:
                return mesh
        return None
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "header": {
                "name": self.name,
                "description": self.description,
                "modality": self.modality,
                "version": self.version
            },
            "volume": {
                "hu_range": {
                    "min": self.hu_min,
                    "max": self.hu_max
                },
                "air_threshold": self.air_threshold,
                "density_multiplier": self.density_multiplier,
                "tissues": self.tissues
            },
            "mesh": self.meshes
        }


def get_preset_for_modality(modality, series_description=""):
    """Get appropriate preset name based on DICOM modality and series description
    
    Args:
        modality: DICOM modality (CT, MR, etc.)
        series_description: Series description for additional context
    
    Returns:
        Preset name string
    """
    modality = modality.upper() if modality else "CT"
    series_desc = series_description.upper() if series_description else ""
    
    if modality == "CT":
        # Check series description for brain/head scans
        if any(keyword in series_desc for keyword in ["BRAIN", "HEAD", "CRANIAL", "CEREBRAL", "SKULL"]):
            return "ct_brain"
        return "ct_standard"
    elif modality == "MR":
        # Check series description for T1/T2/etc.
        if "T1" in series_desc:
            return "mri_t1_brain"
        # Add more MRI presets here as they're created
        # elif "T2" in series_desc:
        #     return "mri_t2_brain"
        else:
            # Default to T1 for MR
            return "mri_t1_brain"
    else:
        # Unknown modality, default to CT
        log.info(f"Unknown modality '{modality}', defaulting to CT preset")
        return "ct_standard"

def load_preset(preset_name):
    """Load a material preset from JSON file"""
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    presets_dir = os.path.join(addon_dir, "presets", "tissue")
    preset_path = os.path.join(presets_dir, f"{preset_name}.json")
    
    if not os.path.exists(preset_path):
        log.warning(f"Preset not found: {preset_path}")
        return None
    
    try:
        with open(preset_path, 'r') as f:
            data = json.load(f)
        
        preset = MaterialPreset(data)
        log.info(f"Loaded preset: {preset.name} ({preset.modality})")
        return preset
    except Exception as e:
        log.error(f"Failed to load preset {preset_name}: {e}")
        return None


def list_presets():
    """List all available presets"""
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    presets_dir = os.path.join(addon_dir, "presets", "tissue")
    
    if not os.path.exists(presets_dir):
        return []
    
    presets = []
    for filename in os.listdir(presets_dir):
        if filename.endswith('.json'):
            preset_name = filename[:-5]  # Remove .json
            preset = load_preset(preset_name)
            if preset:
                presets.append((preset_name, preset.name, preset.description))
    
    return presets


def save_preset(preset, preset_name):
    """Save a material preset to JSON file"""
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    presets_dir = os.path.join(addon_dir, "presets", "tissue")
    
    # Create presets directory if it doesn't exist
    os.makedirs(presets_dir, exist_ok=True)
    
    preset_path = os.path.join(presets_dir, f"{preset_name}.json")
    
    try:
        with open(preset_path, 'w') as f:
            json.dump(preset.to_dict(), f, indent=2)
        
        log.info(f"Saved preset: {preset_path}")
        return True
    except Exception as e:
        log.error(f"Failed to save preset {preset_name}: {e}")
        return False
