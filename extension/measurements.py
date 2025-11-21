"""Volume measurement and quantification tools"""

import numpy as np
from .dicom_io import log


def calculate_tissue_volume(vol_array, hu_min, hu_max, pixel_spacing, slice_thickness):
    """
    Calculate volume of tissue in HU range.
    
    Args:
        vol_array: numpy array with HU values
        hu_min: minimum HU threshold
        hu_max: maximum HU threshold
        pixel_spacing: (row_spacing, col_spacing) in mm
        slice_thickness: slice thickness in mm
    
    Returns:
        volume in milliliters (mL)
    """
    # Create mask for HU range
    mask = (vol_array >= hu_min) & (vol_array <= hu_max)
    
    # Count voxels in range
    voxel_count = np.sum(mask)
    
    # Calculate voxel volume in mm³
    voxel_volume_mm3 = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
    
    # Total volume in mm³
    total_volume_mm3 = voxel_count * voxel_volume_mm3
    
    # Convert to mL (1 mL = 1000 mm³)
    volume_ml = total_volume_mm3 / 1000.0
    
    log(f"Tissue volume calculation:")
    log(f"  HU range: {hu_min} to {hu_max}")
    log(f"  Voxels in range: {voxel_count:,}")
    log(f"  Voxel volume: {voxel_volume_mm3:.3f} mm³")
    log(f"  Total volume: {volume_ml:.2f} mL")
    
    return volume_ml
