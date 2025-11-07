"""DICOM file I/O operations"""

import os
import numpy as np

try:
    from pydicom import dcmread
    try:
        from pydicom.fileset import FileSet
        HAS_DICOMDIR = True
    except ImportError:
        HAS_DICOMDIR = False
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    HAS_DICOMDIR = False

def log(msg): 
    """Print log message"""
    print(f"[DICOM] {msg}")

def gather_dicom_files(root_dir):
    """Gather all DICOM files from directory"""
    candidates = set()
    dicomdir_path = os.path.join(root_dir, "DICOMDIR")
    
    if HAS_DICOMDIR and os.path.isfile(dicomdir_path):
        try:
            fs = FileSet(dicomdir_path)
            for instance in fs:
                abs_path = os.path.normpath(os.path.join(root_dir, instance.path))
                if os.path.isfile(abs_path):
                    candidates.add(abs_path)
            if candidates:
                return list(candidates)
        except Exception as e:
            log(f"DICOMDIR parsing failed: {e}")
    
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            full_path = os.path.join(dirpath, f)
            try:
                if f.upper().endswith(('.DCM', '.DICOM', '')) and os.path.getsize(full_path) > 1000:
                    candidates.add(full_path)
            except:
                pass
    return list(candidates)

def organize_by_series(file_paths):
    """Organize DICOM files by SeriesInstanceUID"""
    series_dict = {}
    
    for path in file_paths:
        try:
            ds = dcmread(path, stop_before_pixels=True, force=True)
            series_uid = getattr(ds, 'SeriesInstanceUID', 'UNKNOWN')
            
            if series_uid not in series_dict:
                # Get image dimensions from first file
                try:
                    ds_full = dcmread(path, force=True)
                    if hasattr(ds_full, 'pixel_array'):
                        rows, cols = ds_full.pixel_array.shape
                    else:
                        rows = getattr(ds, 'Rows', 0)
                        cols = getattr(ds, 'Columns', 0)
                except:
                    rows = getattr(ds, 'Rows', 0)
                    cols = getattr(ds, 'Columns', 0)
                
                series_dict[series_uid] = {
                    'uid': series_uid,
                    'description': getattr(ds, 'SeriesDescription', 'No Description'),
                    'modality': getattr(ds, 'Modality', 'Unknown'),
                    'number': getattr(ds, 'SeriesNumber', 0),
                    'files': [],
                    'instance_count': 0,
                    'rows': rows,
                    'cols': cols,
                    'window_center': float(getattr(ds, 'WindowCenter', 0)) if hasattr(ds, 'WindowCenter') else None,
                    'window_width': float(getattr(ds, 'WindowWidth', 0)) if hasattr(ds, 'WindowWidth') else None,
                    'slice_locations': [],
                }
            
            series_dict[series_uid]['files'].append(path)
            series_dict[series_uid]['slice_locations'].append(
                float(getattr(ds, 'SliceLocation', getattr(ds, 'InstanceNumber', 0)))
            )
            series_dict[series_uid]['instance_count'] = len(series_dict[series_uid]['files'])
        except Exception as e:
            log(f"Failed to read {path}: {e}")
    
    # Sort files within each series by slice location
    for series in series_dict.values():
        paired = list(zip(series['files'], series['slice_locations']))
        paired.sort(key=lambda x: x[1])
        series['files'] = [p[0] for p in paired]
        series['slice_locations'] = sorted(series['slice_locations'])
    
    # Convert to list and sort by series number
    series_list = list(series_dict.values())
    series_list.sort(key=lambda s: s['number'])
    
    return series_list

def load_slice(path):
    """Load a single DICOM slice with proper calibration"""
    ds = dcmread(path, force=True)
    if not hasattr(ds, 'pixel_array'): 
        raise ValueError("No pixel_array")
    
    pixels = ds.pixel_array.astype(np.float32)
    
    # Apply rescale slope and intercept to get Hounsfield units (for CT) or real values
    rescale_slope = float(getattr(ds, 'RescaleSlope', 1.0))
    rescale_intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    
    # Convert to real units (Hounsfield for CT, or calibrated values for MR)
    pixels = pixels * rescale_slope + rescale_intercept
    
    spacing = [float(x) for x in (getattr(ds, 'PixelSpacing', [1.0, 1.0]))][:2]
    
    return {
        "pixels": pixels,
        "pixel_spacing": spacing,
        "slice_thickness": float(getattr(ds, 'SliceThickness', 1.0)),
        "slice_location": float(getattr(ds, 'SliceLocation', 0.0)),
        "instance_number": int(getattr(ds, 'InstanceNumber', 0)),
        "window_center": float(getattr(ds, 'WindowCenter', 0.0)) if hasattr(ds, 'WindowCenter') else None,
        "window_width": float(getattr(ds, 'WindowWidth', 0.0)) if hasattr(ds, 'WindowWidth') else None,
        "rescale_slope": rescale_slope,
        "rescale_intercept": rescale_intercept,
        "ds": ds,
        "path": path
    }