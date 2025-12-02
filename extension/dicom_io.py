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

def classify_dicom_file(filepath):
    """
    Classify a DICOM file into primary/secondary/non-image/invalid.
    
    Returns:
        'primary': Primary image (load)
        'secondary': Secondary/derived image (ignore)
        'non_image': Non-image DICOM (ignore)
        'invalid': Not a valid DICOM file (ignore)
    """
    try:
        ds = dcmread(filepath, stop_before_pixels=True, force=True)
        
        # Check if it has pixel data
        if not hasattr(ds, 'Rows') or not hasattr(ds, 'Columns'):
            return 'non_image'
        
        # Check SOP Class for Secondary Capture
        if hasattr(ds, 'SOPClassUID'):
            if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.7':
                return 'secondary'
        
        # Check ImageType
        if hasattr(ds, 'ImageType'):
            image_type = ds.ImageType
            if len(image_type) > 1:
                if image_type[1] == 'PRIMARY':
                    return 'primary'
                elif image_type[1] == 'SECONDARY':
                    return 'secondary'
                elif image_type[0] == 'DERIVED':
                    return 'secondary'
        
        # Default to primary if can't determine
        return 'primary'
        
    except Exception as e:
        log(f"Failed to classify {filepath}: {e}")
        return 'invalid'

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
                
                # Get spacing info
                pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
                slice_thickness = float(getattr(ds, 'SliceThickness', 1.0))
                
                series_dict[series_uid] = {
                    'uid': series_uid,
                    'description': getattr(ds, 'SeriesDescription', 'No Description'),
                    'modality': getattr(ds, 'Modality', 'Unknown'),
                    'number': getattr(ds, 'SeriesNumber', 0),
                    'files': [],
                    'instance_count': 0,
                    'rows': rows,
                    'cols': cols,
                    'pixel_spacing': [float(x) for x in pixel_spacing],
                    'slice_thickness': slice_thickness,
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
    
    # Convert to list and sort by series number (handle None values)
    series_list = list(series_dict.values())
    series_list.sort(key=lambda s: s['number'] if s['number'] is not None else 0)
    
    return series_list

def load_slice(path):
    """
    Load a single DICOM slice with proper calibration.
    
    Returns:
        dict with slice data, or None if slice cannot be loaded
    """
    try:
        ds = dcmread(path, force=True)
    except Exception as e:
        log(f"Failed to read DICOM file {path}: {e}")
        return None
    
    # Try to access pixel_array (may fail for compressed or incomplete files)
    try:
        pixels = ds.pixel_array.astype(np.float32)
    except Exception as e:
        log(f"No pixel data in {path}: {e}")
        return None
    
    # Apply rescale slope and intercept to get Hounsfield units (for CT) or real values
    rescale_slope = float(getattr(ds, 'RescaleSlope', 1.0))
    rescale_intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    
    # Convert to real units (Hounsfield for CT, or calibrated values for MR)
    pixels = pixels * rescale_slope + rescale_intercept
    
    spacing = [float(x) for x in (getattr(ds, 'PixelSpacing', [1.0, 1.0]))][:2]
    
    # Get spatial information for 3D positioning
    position = [float(x) for x in getattr(ds, 'ImagePositionPatient', [0.0, 0.0, 0.0])]
    orientation = [float(x) for x in getattr(ds, 'ImageOrientationPatient', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])]
    
    return {
        "pixels": pixels,
        "pixel_spacing": spacing,
        "slice_thickness": float(getattr(ds, 'SliceThickness', 1.0)),
        "slice_location": float(getattr(ds, 'SliceLocation', 0.0)),
        "instance_number": int(getattr(ds, 'InstanceNumber', 0)),
        "position": position,  # ImagePositionPatient
        "orientation": orientation,  # ImageOrientationPatient
        "window_center": float(getattr(ds, 'WindowCenter', 0.0)) if hasattr(ds, 'WindowCenter') else None,
        "window_width": float(getattr(ds, 'WindowWidth', 0.0)) if hasattr(ds, 'WindowWidth') else None,
        "rescale_slope": rescale_slope,
        "rescale_intercept": rescale_intercept,
        "ds": ds,
        "path": path
    }


def load_patient_from_folder(root_dir):
    """
    Load patient data from DICOM folder.
    Scans all files, classifies them, and creates Patient object with primary series.
    
    Returns:
        Patient object with all primary series loaded
    """
    from .patient import Patient, SeriesInfo
    
    log(f"Loading patient from: {root_dir}")
    
    # Gather all DICOM files
    all_files = gather_dicom_files(root_dir)
    log(f"Found {len(all_files)} potential DICOM files")
    
    # Classify files
    primary_files = []
    secondary_count = 0
    non_image_count = 0
    invalid_count = 0
    
    for filepath in all_files:
        classification = classify_dicom_file(filepath)
        if classification == 'primary':
            primary_files.append(filepath)
        elif classification == 'secondary':
            secondary_count += 1
        elif classification == 'non_image':
            non_image_count += 1
        else:  # invalid
            invalid_count += 1
    
    log(f"Classification: {len(primary_files)} primary, {secondary_count} secondary, {non_image_count} non-image, {invalid_count} invalid")
    
    if not primary_files:
        raise ValueError("No primary DICOM images found")
    
    # Organize primary files by series
    series_list = organize_by_series(primary_files)
    log(f"Organized into {len(series_list)} primary series")
    
    # Create Patient object
    patient = Patient()
    patient.dicom_root_path = root_dir
    patient.primary_count = len(primary_files)
    patient.secondary_count = secondary_count
    patient.non_image_count = non_image_count
    
    # Extract patient and study metadata from first file
    if primary_files:
        try:
            ds = dcmread(primary_files[0], stop_before_pixels=True, force=True)
            patient.patient_id = str(getattr(ds, 'PatientID', ''))
            patient.patient_name = str(getattr(ds, 'PatientName', ''))
            patient.patient_birth_date = str(getattr(ds, 'PatientBirthDate', ''))
            patient.patient_sex = str(getattr(ds, 'PatientSex', ''))
            patient.study_instance_uid = str(getattr(ds, 'StudyInstanceUID', ''))
            patient.study_date = str(getattr(ds, 'StudyDate', ''))
            patient.study_description = str(getattr(ds, 'StudyDescription', ''))
        except Exception as e:
            log(f"Failed to extract patient metadata: {e}")
    
    # Convert series_list to SeriesInfo objects
    for series_dict in series_list:
        # Get spatial metadata from first file
        try:
            ds = dcmread(series_dict['files'][0], stop_before_pixels=True, force=True)
            
            # ImagePositionPatient
            ipp = getattr(ds, 'ImagePositionPatient', [0.0, 0.0, 0.0])
            image_position_patient = tuple(float(x) for x in ipp)
            
            # ImageOrientationPatient
            iop = getattr(ds, 'ImageOrientationPatient', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            image_orientation_patient = tuple(float(x) for x in iop)
            
            # FrameOfReferenceUID
            frame_of_reference_uid = str(getattr(ds, 'FrameOfReferenceUID', ''))
            
            # ImageType
            image_type = list(getattr(ds, 'ImageType', []))
            
        except Exception as e:
            log(f"Failed to extract spatial metadata: {e}")
            image_position_patient = (0.0, 0.0, 0.0)
            image_orientation_patient = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            frame_of_reference_uid = ""
            image_type = []
        
        # Make file paths relative to root_dir
        relative_paths = []
        for fpath in series_dict['files']:
            try:
                rel_path = os.path.relpath(fpath, root_dir)
                relative_paths.append(rel_path)
            except:
                relative_paths.append(fpath)
        
        series_info = SeriesInfo(
            series_instance_uid=series_dict['uid'],
            series_number=series_dict['number'] if series_dict['number'] is not None else 0,
            series_description=series_dict['description'],
            modality=series_dict['modality'],
            image_type=image_type,
            rows=series_dict['rows'],
            cols=series_dict['cols'],
            slice_count=series_dict['instance_count'],
            pixel_spacing=(series_dict['pixel_spacing'][0], series_dict['pixel_spacing'][1]) if 'pixel_spacing' in series_dict else (1.0, 1.0),
            slice_thickness=series_dict.get('slice_thickness', 1.0),
            image_position_patient=image_position_patient,
            image_orientation_patient=image_orientation_patient,
            frame_of_reference_uid=frame_of_reference_uid,
            window_center=series_dict.get('window_center'),
            window_width=series_dict.get('window_width'),
            file_paths=relative_paths,
            slice_locations=series_dict['slice_locations'],
        )
        
        patient.series.append(series_info)
    
    log(f"Patient loaded: {patient.patient_name} ({patient.patient_id})")
    log(f"  Study: {patient.study_description} ({patient.study_date})")
    log(f"  {len(patient.series)} primary series")
    
    return patient
