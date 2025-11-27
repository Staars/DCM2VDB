"""Material preset system for volume rendering"""

import json
import os
from .dicom_io import log

class MaterialPreset:
    """Represents a volume material preset"""
    
    def __init__(self, data):
        self.name = data.get("name", "Unknown")
        self.description = data.get("description", "")
        self.modality = data.get("modality", "CT")
        self.version = data.get("version", "1.0")
        
        # HU range
        hu_range = data.get("hu_range", {})
        self.hu_min = hu_range.get("min", -1024)
        self.hu_max = hu_range.get("max", 3071)
        
        # Rendering settings
        self.air_threshold = data.get("air_threshold", -200)
        self.density_multiplier = data.get("density_multiplier", 600)
        
        # Tissues (sorted by order)
        tissues_data = data.get("tissues", [])
        self.tissues = sorted(tissues_data, key=lambda t: t.get("order", 0))
    
    def get_tissue(self, name):
        """Get tissue by name"""
        for tissue in self.tissues:
            if tissue["name"] == name:
                return tissue
        return None
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "modality": self.modality,
            "version": self.version,
            "hu_range": {
                "min": self.hu_min,
                "max": self.hu_max
            },
            "air_threshold": self.air_threshold,
            "density_multiplier": self.density_multiplier,
            "tissues": self.tissues
        }


def load_preset(preset_name):
    """Load a material preset from JSON file"""
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    presets_dir = os.path.join(addon_dir, "presets")
    preset_path = os.path.join(presets_dir, f"{preset_name}.json")
    
    if not os.path.exists(preset_path):
        log(f"Preset not found: {preset_path}")
        return None
    
    try:
        with open(preset_path, 'r') as f:
            data = json.load(f)
        
        preset = MaterialPreset(data)
        log(f"Loaded preset: {preset.name} ({preset.modality})")
        return preset
    except Exception as e:
        log(f"Failed to load preset {preset_name}: {e}")
        return None


def list_presets():
    """List all available presets"""
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    presets_dir = os.path.join(addon_dir, "presets")
    
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
    presets_dir = os.path.join(addon_dir, "presets")
    
    # Create presets directory if it doesn't exist
    os.makedirs(presets_dir, exist_ok=True)
    
    preset_path = os.path.join(presets_dir, f"{preset_name}.json")
    
    try:
        with open(preset_path, 'w') as f:
            json.dump(preset.to_dict(), f, indent=2)
        
        log(f"Saved preset: {preset_path}")
        return True
    except Exception as e:
        log(f"Failed to save preset {preset_name}: {e}")
        return False
