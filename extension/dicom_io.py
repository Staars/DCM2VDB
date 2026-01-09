"""DICOM file I/O operations"""

import os
import numpy as np
from .utils import SimpleLogger
from .constants import MIN_DICOM_FILE_SIZE, SOP_CLASS_SECONDARY_CAPTURE

# Get logger for this extension
log = SimpleLogger()

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
            if ds.SOPClassUID == SOP_CLASS_SECONDARY_CAPTURE:
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
        log.error(f"Failed to classify {filepath}: {e}")
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
            log.warning(f"DICOMDIR parsing failed: {e}")
    
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            full_path = os.path.join(dirpath, f)
            try:
                if f.upper().endswith(('.DCM', '.DICOM', '')) and os.path.getsize(full_path) > MIN_DICOM_FILE_SIZE:
                    candidates.add(full_path)
            except:
                pass
    return list(candidates)

def analyze_series_for_4d(file_paths):
    """
    Analyze DICOM files to detect 4D series based on AcquisitionNumber.
    
    Returns:
        dict: {series_uid: {
            'is_4d': bool,
            'acquisition_numbers': [list],
            'files_by_acquisition': {acq_num: [file_paths]},
            'metadata': {...}
        }}
    """
    series_data = {}
    
    for path in file_paths:
        try:
            ds = dcmread(path, stop_before_pixels=True, force=True)
            series_uid = getattr(ds, 'SeriesInstanceUID', 'UNKNOWN')
            acquisition_num = getattr(ds, 'AcquisitionNumber', None)
            instance_num = getattr(ds, 'InstanceNumber', None)
            
            # Initialize series data
            if series_uid not in series_data:
                series_data[series_uid] = {
                    'acquisition_numbers': set(),
                    'files_by_acquisition': {},
                    'metadata': {
                        'description': getattr(ds, 'SeriesDescription', 'No Description'),
                        'modality': getattr(ds, 'Modality', 'Unknown'),
                        'series_number': getattr(ds, 'SeriesNumber', 0),
                    }
                }
            
            # Track acquisition numbers
            if acquisition_num is not None:
                series_data[series_uid]['acquisition_numbers'].add(acquisition_num)
                
                # Group files by acquisition
                if acquisition_num not in series_data[series_uid]['files_by_acquisition']:
                    series_data[series_uid]['files_by_acquisition'][acquisition_num] = []
                
                series_data[series_uid]['files_by_acquisition'][acquisition_num].append({
                    'path': path,
                    'instance_num': instance_num,
                    'ds': ds
                })
            else:
                # No acquisition number - treat as single acquisition
                if 0 not in series_data[series_uid]['files_by_acquisition']:
                    series_data[series_uid]['files_by_acquisition'][0] = []
                
                series_data[series_uid]['files_by_acquisition'][0].append({
                    'path': path,
                    'instance_num': instance_num,
                    'ds': ds
                })
                
        except Exception as e:
            log.error(f"Failed to analyze {path}: {e}")
    
    # Determine if each series is 4D
    for series_uid, data in series_data.items():
        num_acquisitions = len(data['files_by_acquisition'])
        data['is_4d'] = num_acquisitions > 1
        data['acquisition_numbers'] = sorted(data['acquisition_numbers'])
    
    return series_data

def organize_by_series(file_paths):
    """Organize DICOM files by SeriesInstanceUID, detecting 4D series"""
    
    # Analyze for 4D series
    log.info("Analyzing DICOM files for 4D series...")
    series_analysis = analyze_series_for_4d(file_paths)
    
    series_dict = {}
    
    # Process each series
    for series_uid, analysis in series_analysis.items():
        metadata = analysis['metadata']
        
        if analysis['is_4d']:
            # 4D series detected - create ONE series entry with all time points
            num_time_points = len(analysis['acquisition_numbers'])
            log.info(f"4D series detected: {metadata['description']}")
            log.info(f"  Time points: {num_time_points} (acquisitions: {analysis['acquisition_numbers']})")
            
            # Get metadata from first time point
            first_acq = analysis['acquisition_numbers'][0]
            first_files = analysis['files_by_acquisition'][first_acq]
            first_ds = first_files[0]['ds']
            first_path = first_files[0]['path']
            
            # Get image dimensions
            try:
                ds_full = dcmread(first_path, force=True)
                if hasattr(ds_full, 'pixel_array'):
                    rows, cols = ds_full.pixel_array.shape
                else:
                    rows = getattr(first_ds, 'Rows', 0)
                    cols = getattr(first_ds, 'Columns', 0)
            except:
                rows = getattr(first_ds, 'Rows', 0)
                cols = getattr(first_ds, 'Columns', 0)
            
            pixel_spacing = getattr(first_ds, 'PixelSpacing', [1.0, 1.0])
            slice_thickness = getattr(first_ds, 'SliceThickness', 1.0)
            slice_thickness = float(slice_thickness) if slice_thickness is not None else 1.0
            
            # Organize time points data
            time_points_data = []
            for acq_num in sorted(analysis['acquisition_numbers']):
                files_info = analysis['files_by_acquisition'][acq_num]
                files_info.sort(key=lambda x: x['instance_num'] if x['instance_num'] else 0)
                
                time_points_data.append({
                    'acquisition_number': acq_num,
                    'files': [f['path'] for f in files_info],
                    'slice_count': len(files_info),
                })
            
            # Collect ALL files from all time points for preview
            all_files = []
            for tp_data in time_points_data:
                all_files.extend(tp_data['files'])
            
            # Create single series entry for 4D data
            series_dict[series_uid] = {
                'uid': series_uid,
                'description': f"{metadata['description']} [4D - {num_time_points} timepoints]",
                'modality': metadata['modality'],
                'number': metadata['series_number'],
                'files': all_files,  # ALL files from all time points for preview
                'instance_count': len(all_files),
                'rows': rows,
                'cols': cols,
                'pixel_spacing': [float(x) for x in pixel_spacing],
                'slice_thickness': slice_thickness,
                'window_center': float(getattr(first_ds, 'WindowCenter', 0)) if hasattr(first_ds, 'WindowCenter') else None,
                'window_width': float(getattr(first_ds, 'WindowWidth', 0)) if hasattr(first_ds, 'WindowWidth') else None,
                'slice_locations': [float(getattr(f['ds'], 'SliceLocation', f['instance_num'] or 0)) for f in first_files],
                'is_4d': True,
                'time_points': time_points_data,
                'num_time_points': num_time_points,
            }
            
            log.info(f"  Created 4D series with {num_time_points} time points")
        
        else:
            # Regular series
            # Get files from the single acquisition (key 0 or the only key)
            acq_key = list(analysis['files_by_acquisition'].keys())[0]
            files_info = analysis['files_by_acquisition'][acq_key]
            
            # Sort by instance number
            files_info.sort(key=lambda x: x['instance_num'] if x['instance_num'] else 0)
            
            # Get metadata from first file
            first_ds = files_info[0]['ds']
            first_path = files_info[0]['path']
            
            # Get image dimensions
            try:
                ds_full = dcmread(first_path, force=True)
                if hasattr(ds_full, 'pixel_array'):
                    rows, cols = ds_full.pixel_array.shape
                else:
                    rows = getattr(first_ds, 'Rows', 0)
                    cols = getattr(first_ds, 'Columns', 0)
            except:
                rows = getattr(first_ds, 'Rows', 0)
                cols = getattr(first_ds, 'Columns', 0)
            
            pixel_spacing = getattr(first_ds, 'PixelSpacing', [1.0, 1.0])
            slice_thickness = getattr(first_ds, 'SliceThickness', 1.0)
            slice_thickness = float(slice_thickness) if slice_thickness is not None else 1.0
            
            series_dict[series_uid] = {
                'uid': series_uid,
                'description': metadata['description'],
                'modality': metadata['modality'],
                'number': metadata['series_number'],
                'files': [f['path'] for f in files_info],
                'instance_count': len(files_info),
                'rows': rows,
                'cols': cols,
                'pixel_spacing': [float(x) for x in pixel_spacing],
                'slice_thickness': slice_thickness,
                'window_center': float(getattr(first_ds, 'WindowCenter', 0)) if hasattr(first_ds, 'WindowCenter') else None,
                'window_width': float(getattr(first_ds, 'WindowWidth', 0)) if hasattr(first_ds, 'WindowWidth') else None,
                'slice_locations': [float(getattr(f['ds'], 'SliceLocation', f['instance_num'] or 0)) for f in files_info],
            }
    
    # Convert to list and sort by series number
    series_list = list(series_dict.values())
    series_list.sort(key=lambda s: (s['number'] if s['number'] is not None else 0, s['description']))
    
    log.info(f"Organized into {len(series_list)} series")
    
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
        log.error(f"Failed to read DICOM file {path}: {e}")
        return None
    
    # Try to access pixel_array (may fail for compressed or incomplete files)
    try:
        pixels = ds.pixel_array.astype(np.float32)
    except Exception as e:
        log.warning(f"No pixel data in {path}: {e}")
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
    
    slice_thickness = getattr(ds, 'SliceThickness', 1.0)
    slice_thickness = float(slice_thickness) if slice_thickness is not None else 1.0
    
    slice_location = getattr(ds, 'SliceLocation', 0.0)
    slice_location = float(slice_location) if slice_location is not None else 0.0
    
    return {
        "pixels": pixels,
        "pixel_spacing": spacing,
        "slice_thickness": slice_thickness,
        "slice_location": slice_location,
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
    
    log.info(f"Loading patient from: {root_dir}")
    
    # Gather all DICOM files
    all_files = gather_dicom_files(root_dir)
    log.info(f"Found {len(all_files)} potential DICOM files")
    
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
    
    log.info(f"Classification: {len(primary_files)} primary, {secondary_count} secondary, {non_image_count} non-image, {invalid_count} invalid")
    
    if not primary_files:
        raise ValueError("No primary DICOM images found")
    
    # Organize primary files by series
    series_list = organize_by_series(primary_files)
    log.info(f"Organized into {len(series_list)} primary series")
    
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
            log.warning(f"Failed to extract patient metadata: {e}")
    
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
            log.warning(f"Failed to extract spatial metadata: {e}")
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
            is_4d=series_dict.get('is_4d', False),
            time_points=series_dict.get('time_points', []),
            num_time_points=series_dict.get('num_time_points', 0),
        )
        
        patient.series.append(series_info)
    
    log.info(f"Patient loaded: {patient.patient_name} ({patient.patient_id})")
    log.info(f"  Study: {patient.study_description} ({patient.study_date})")
    log.info(f"  {len(patient.series)} primary series")
    
    return patient
