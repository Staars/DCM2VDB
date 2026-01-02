"""Volume measurement and quantification tools"""

import numpy as np
import os
import json
from typing import Tuple, Optional, Any
from numpy.typing import NDArray
from .utils import SimpleLogger
from .constants import MM3_TO_ML

# Get logger for this extension
log = SimpleLogger()


def calculate_tissue_volume(
    vol_array: NDArray[np.float32], 
    hu_min: float, 
    hu_max: float, 
    pixel_spacing: Tuple[float, float], 
    slice_thickness: float
) -> float:
    """
    Calculate volume of tissue in HU range.
    
    Args:
        vol_array: Numpy array with HU values (3D volume)
        hu_min: Minimum HU threshold
        hu_max: Maximum HU threshold
        pixel_spacing: (row_spacing, col_spacing) in mm
        slice_thickness: Slice thickness in mm
    
    Returns:
        Volume in milliliters (mL)
    """
    # Create mask for HU range
    mask = (vol_array >= hu_min) & (vol_array <= hu_max)
    
    # Count voxels in range
    voxel_count = np.sum(mask)
    
    # Calculate voxel volume in mm³
    voxel_volume_mm3 = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
    
    # Total volume in mm³
    total_volume_mm3 = voxel_count * voxel_volume_mm3
    
    # Convert to mL (1 mL = MM3_TO_ML mm³)
    volume_ml = total_volume_mm3 / MM3_TO_ML
    
    log.debug(f"Tissue volume calculation:")
    log.debug(f"  HU range: {hu_min} to {hu_max}")
    log.debug(f"  Voxels in range: {voxel_count:,}")
    log.debug(f"  Voxel volume: {voxel_volume_mm3:.3f} mm³")
    log.debug(f"  Total volume: {volume_ml:.2f} mL")
    
    return volume_ml


def calculate_and_store_tissue_volumes(context: Any, series: Any) -> bool:
    """
    Calculate all tissue volumes for a series and store in series object.
    
    This function:
    1. Loads the volume data from disk
    2. Parses spacing information
    3. Gets tissue thresholds from the active material preset
    4. Calculates volume for each tissue type
    5. Stores results in series.tissue_volumes dictionary
    
    Args:
        context: Blender context (provides access to scene properties)
        series: Series object to calculate and store volumes for
        
    Returns:
        bool: True if calculation succeeded, False if failed or no data available
    """
    scn = context.scene
    
    # Check if volume data is available
    if not scn.dicom_volume_data_path or not os.path.exists(scn.dicom_volume_data_path):
        log.debug("No volume data available for measurement calculation")
        return False
    
    try:
        # Load volume data
        vol_array = np.load(scn.dicom_volume_data_path)
        
        # Parse spacing: [X, Y, Z] in mm
        spacing = json.loads(scn.dicom_volume_spacing)
        pixel_spacing = (spacing[1], spacing[0])  # (row, col) = (Y, X)
        slice_thickness = spacing[2]  # Z
        
        # Get tissue thresholds from active material preset
        from .properties import get_tissue_thresholds_from_preset
        thresholds = get_tissue_thresholds_from_preset(scn.dicom_active_material_preset)
        
        # Clear previous measurements
        series.tissue_volumes = {}
        
        # Calculate volume for each tissue defined in preset
        for tissue_name, tissue_range in thresholds.items():
            volume_ml = calculate_tissue_volume(
                vol_array,
                tissue_range.get('min', 0),
                tissue_range.get('max', 0),
                pixel_spacing, 
                slice_thickness
            )
            if volume_ml > 0:
                series.tissue_volumes[tissue_name] = volume_ml
        
        log.debug(f"Tissue volumes calculated for series {series.series_number}")
        return True
        
    except Exception as e:
        log.error(f"Failed to calculate tissue volumes: {e}")
        return False
