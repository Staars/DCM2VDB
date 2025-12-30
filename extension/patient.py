"""
Patient data model for DICOM import.

This module defines the core data structures for representing a patient
and their imaging studies. Data is serializable to/from JSON for persistence
in .blend files.
"""

import json
from .utils import SimpleLogger
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Get logger for this extension
log = SimpleLogger()


@dataclass
class SeriesInfo:
    """
    Information about a single DICOM series (primary acquisitions only).
    """
    
    # Series Identification
    series_instance_uid: str
    series_number: int = 0
    series_description: str = ""
    
    # Acquisition Info
    modality: str = ""  # CT, MR, etc.
    image_type: List[str] = field(default_factory=list)  # ['ORIGINAL', 'PRIMARY', 'AXIAL']
    
    # Dimensions
    rows: int = 0
    cols: int = 0
    slice_count: int = 0
    
    # Spacing
    pixel_spacing: Tuple[float, float] = (1.0, 1.0)  # (row, col) in mm
    slice_thickness: float = 1.0  # in mm
    
    # Spatial Position (DICOM Patient Coordinate System)
    image_position_patient: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, z) in mm
    image_orientation_patient: Tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    frame_of_reference_uid: str = ""  # Links series in same spatial reference
    
    # Window/Level
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    
    # File References (relative to dicom_root_path)
    file_paths: List[str] = field(default_factory=list)
    slice_locations: List[float] = field(default_factory=list)
    
    # Visualization State
    is_loaded: bool = False  # Has VDB been generated?
    is_visible: bool = False  # Is volume visible in viewport?
    is_selected: bool = True  # Is series selected for processing? (default: True)
    show_volume: bool = True  # Show volume rendering (per-series)
    show_bone: bool = False  # Show bone mesh (per-series)
    
    # Per-series measurements (dynamic based on preset)
    # Dict mapping tissue name to volume in mL: {'fat': 123.45, 'liquid': 67.89, ...}
    tissue_volumes: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SeriesInfo':
        """Create from dictionary (JSON deserialization)."""
        # Convert lists back to tuples where needed
        if 'pixel_spacing' in data and isinstance(data['pixel_spacing'], list):
            data['pixel_spacing'] = tuple(data['pixel_spacing'])
        if 'image_position_patient' in data and isinstance(data['image_position_patient'], list):
            data['image_position_patient'] = tuple(data['image_position_patient'])
        if 'image_orientation_patient' in data and isinstance(data['image_orientation_patient'], list):
            data['image_orientation_patient'] = tuple(data['image_orientation_patient'])
        
        return cls(**data)


@dataclass
class Patient:
    """
    Central data structure representing a patient and their imaging studies.
    Serializable to/from JSON for .blend file persistence.
    """
    
    # Patient Metadata
    patient_id: str = ""
    patient_name: str = ""
    patient_birth_date: str = ""
    patient_sex: str = ""
    
    # Study Metadata
    study_instance_uid: str = ""
    study_date: str = ""
    study_description: str = ""
    
    # File System
    dicom_root_path: str = ""  # Root folder containing DICOM files
    
    # Series Data (list of primary series only)
    series: List[SeriesInfo] = field(default_factory=list)
    
    # Blender Object References
    volume_objects: Dict[str, str] = field(default_factory=dict)  # series_uid -> volume object name
    mesh_objects: Dict[str, str] = field(default_factory=dict)    # series_uid -> mesh object name
    
    # Statistics (not serialized, computed on load)
    primary_count: int = 0
    secondary_count: int = 0
    non_image_count: int = 0
    
    def to_json(self) -> str:
        """Serialize Patient to JSON string."""
        data = {
            'patient_id': self.patient_id,
            'patient_name': self.patient_name,
            'patient_birth_date': self.patient_birth_date,
            'patient_sex': self.patient_sex,
            'study_instance_uid': self.study_instance_uid,
            'study_date': self.study_date,
            'study_description': self.study_description,
            'dicom_root_path': self.dicom_root_path,
            'series': [s.to_dict() for s in self.series],
            'volume_objects': self.volume_objects,
            'mesh_objects': self.mesh_objects,
            'primary_count': self.primary_count,
            'secondary_count': self.secondary_count,
            'non_image_count': self.non_image_count,
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Patient':
        """Deserialize Patient from JSON string."""
        data = json.loads(json_str)
        
        # Convert series list
        series_data = data.pop('series', [])
        series = [SeriesInfo.from_dict(s) for s in series_data]
        
        # Create patient with remaining data
        patient = cls(**data)
        patient.series = series
        
        return patient
    
    def get_series_by_uid(self, series_uid: str) -> Optional[SeriesInfo]:
        """Get series by SeriesInstanceUID."""
        for s in self.series:
            if s.series_instance_uid == series_uid:
                return s
        return None
    
    def get_series_by_frame_of_reference(self) -> Dict[str, List[SeriesInfo]]:
        """Group series by FrameOfReferenceUID."""
        groups = {}
        for s in self.series:
            frame_uid = s.frame_of_reference_uid or "unknown"
            if frame_uid not in groups:
                groups[frame_uid] = []
            groups[frame_uid].append(s)
        return groups
    
    def get_loaded_series(self) -> List[SeriesInfo]:
        """Get all series that have been loaded (VDB generated)."""
        return [s for s in self.series if s.is_loaded]
    
    def get_visible_series(self) -> List[SeriesInfo]:
        """Get all series that are currently visible."""
        return [s for s in self.series if s.is_visible]
