"""CSV export functionality for measurements"""

import csv
from ..utils import SimpleLogger

log = SimpleLogger()


def export_measurements_to_csv(context, filepath):
    """Export measurements to CSV file
    
    Args:
        context: Blender context
        filepath: Output CSV file path
    
    Returns:
        bool: True if successful, False otherwise
    """
    scn = context.scene
    
    if len(scn.dicom_measurements) == 0:
        log.warning("No measurements to export")
        return False
    
    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow(['Measurement', 'Type', 'Value', 'Unit', 'Status', 'Points'])
            
            # Data rows
            for measurement in scn.dicom_measurements:
                points_str = "; ".join([
                    f"({p.x:.2f}, {p.y:.2f}, {p.z:.2f})"
                    for p in measurement.points
                ])
                
                writer.writerow([
                    measurement.label,
                    measurement.measurement_type,
                    f"{measurement.value:.2f}" if measurement.status == 'COMPLETED' else "",
                    measurement.unit,
                    measurement.status,
                    points_str
                ])
        
        log.info(f"Exported {len(scn.dicom_measurements)} measurements to {filepath}")
        return True
        
    except Exception as e:
        log.error(f"Failed to export measurements: {e}")
        return False
