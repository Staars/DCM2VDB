"""Measurement template system for clinical protocols"""

import json
import os
from ..utils import SimpleLogger

log = SimpleLogger()


class MeasurementTemplate:
    """Represents a measurement protocol template"""
    
    def __init__(self, data):
        # Header section
        header = data.get("header", {})
        self.name = header.get("name", "unknown")
        self.label = header.get("label", "Unknown Template")
        self.description = header.get("description", "")
        self.modality = header.get("modality", "CT")
        self.version = header.get("version", "1.0")
        
        # Measurements
        self.measurements = data.get("measurements", [])
    
    def get_measurement(self, measurement_id):
        """Get measurement definition by ID"""
        for measurement in self.measurements:
            if measurement["id"] == measurement_id:
                return measurement
        return None
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "header": {
                "name": self.name,
                "label": self.label,
                "description": self.description,
                "modality": self.modality,
                "version": self.version
            },
            "measurements": self.measurements
        }


def load_measurement_template(template_name):
    """Load a measurement template from JSON file
    
    Args:
        template_name: Name of template (without .json extension)
    
    Returns:
        MeasurementTemplate object or None if not found
    """
    addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(addon_dir, "presets", "measurements")
    template_path = os.path.join(templates_dir, f"{template_name}.json")
    
    if not os.path.exists(template_path):
        log.warning(f"Measurement template not found: {template_path}")
        return None
    
    try:
        with open(template_path, 'r') as f:
            data = json.load(f)
        
        template = MeasurementTemplate(data)
        log.info(f"Loaded measurement template: {template.label} ({template.modality})")
        return template
    except Exception as e:
        log.error(f"Failed to load measurement template {template_name}: {e}")
        return None


def list_measurement_templates():
    """List all available measurement templates
    
    Returns:
        List of tuples: (template_name, label, description)
    """
    addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(addon_dir, "presets", "measurements")
    
    if not os.path.exists(templates_dir):
        return []
    
    templates = []
    for filename in os.listdir(templates_dir):
        if filename.endswith('.json'):
            template_name = filename[:-5]  # Remove .json
            template = load_measurement_template(template_name)
            if template:
                templates.append((template_name, template.label, template.description))
    
    return templates


def save_measurement_template(template, template_name):
    """Save a measurement template to JSON file
    
    Args:
        template: MeasurementTemplate object
        template_name: Name for the template file (without .json)
    
    Returns:
        bool: True if successful, False otherwise
    """
    addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(addon_dir, "presets", "measurements")
    
    # Create directory if it doesn't exist
    os.makedirs(templates_dir, exist_ok=True)
    
    template_path = os.path.join(templates_dir, f"{template_name}.json")
    
    try:
        with open(template_path, 'w') as f:
            json.dump(template.to_dict(), f, indent=2)
        
        log.info(f"Saved measurement template: {template_path}")
        return True
    except Exception as e:
        log.error(f"Failed to save measurement template {template_name}: {e}")
        return False
