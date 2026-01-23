"""Measurement calculation functions"""

import numpy as np
from typing import Tuple
from ..utils import SimpleLogger

log = SimpleLogger()


def project_point_to_plane(point: Tuple[float, float, float], plane: str) -> Tuple[float, float]:
    """Project 3D point to 2D plane
    
    Args:
        point: (x, y, z) coordinates in mm
        plane: 'axial', 'sagittal', or 'coronal'
    
    Returns:
        (u, v) 2D coordinates in the projection plane
    """
    x, y, z = point
    
    if plane == 'axial':  # XY plane (transverse)
        return (x, y)
    elif plane == 'sagittal':  # YZ plane
        return (y, z)
    elif plane == 'coronal':  # XZ plane
        return (x, z)
    else:
        log.warning(f"Unknown plane '{plane}', defaulting to axial")
        return (x, y)


def calculate_distance_2d(point1: Tuple[float, float, float],
                         point2: Tuple[float, float, float],
                         plane: str = 'axial') -> float:
    """Calculate 2D distance between two points projected to plane
    
    Args:
        point1: (x, y, z) coordinates in mm
        point2: (x, y, z) coordinates in mm
        plane: 'axial', 'sagittal', or 'coronal'
    
    Returns:
        Distance in mm
    """
    p1_2d = np.array(project_point_to_plane(point1, plane))
    p2_2d = np.array(project_point_to_plane(point2, plane))
    
    distance = np.linalg.norm(p2_2d - p1_2d)
    
    log.debug(f"Distance 2D ({plane}): {point1} to {point2} = {distance:.2f} mm")
    return float(distance)


def calculate_distance_3d(point1: Tuple[float, float, float],
                         point2: Tuple[float, float, float]) -> float:
    """Calculate true 3D Euclidean distance between two points
    
    Args:
        point1: (x, y, z) coordinates in mm
        point2: (x, y, z) coordinates in mm
    
    Returns:
        Distance in mm
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    distance = np.linalg.norm(p2 - p1)
    
    log.debug(f"Distance 3D: {point1} to {point2} = {distance:.2f} mm")
    return float(distance)


def calculate_angle_2d(point1: Tuple[float, float, float],
                      point2: Tuple[float, float, float],
                      point3: Tuple[float, float, float],
                      point4: Tuple[float, float, float],
                      plane: str = 'axial') -> float:
    """Calculate angle between two lines projected to plane
    
    Line 1: point1 -> point2
    Line 2: point3 -> point4
    
    Args:
        point1: First point of line 1 (x, y, z) in mm
        point2: Second point of line 1 (x, y, z) in mm
        point3: First point of line 2 (x, y, z) in mm
        point4: Second point of line 2 (x, y, z) in mm
        plane: 'axial', 'sagittal', or 'coronal'
    
    Returns:
        Angle in degrees (0-180)
    """
    # Project all points to 2D plane
    p1_2d = np.array(project_point_to_plane(point1, plane))
    p2_2d = np.array(project_point_to_plane(point2, plane))
    p3_2d = np.array(project_point_to_plane(point3, plane))
    p4_2d = np.array(project_point_to_plane(point4, plane))
    
    # Create direction vectors for both lines
    v1 = p2_2d - p1_2d
    v2 = p4_2d - p3_2d
    
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1_norm, v2_norm)
    
    # Clamp to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(np.abs(cos_angle))  # Use abs to get acute angle
    angle_deg = np.degrees(angle_rad)
    
    log.debug(f"Angle 2D ({plane}): line({point1}-{point2}) to line({point3}-{point4}) = {angle_deg:.2f}°")
    return float(angle_deg)


def calculate_angle_3d(point1: Tuple[float, float, float],
                      point2: Tuple[float, float, float],
                      point3: Tuple[float, float, float],
                      point4: Tuple[float, float, float]) -> float:
    """Calculate true 3D angle between two lines
    
    Line 1: point1 -> point2
    Line 2: point3 -> point4
    
    Args:
        point1: First point of line 1 (x, y, z) in mm
        point2: Second point of line 1 (x, y, z) in mm
        point3: First point of line 2 (x, y, z) in mm
        point4: Second point of line 2 (x, y, z) in mm
    
    Returns:
        Angle in degrees (0-180)
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    p4 = np.array(point4)
    
    # Create direction vectors for both lines
    v1 = p2 - p1
    v2 = p4 - p3
    
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1_norm, v2_norm)
    
    # Clamp to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(np.abs(cos_angle))  # Use abs to get acute angle
    angle_deg = np.degrees(angle_rad)
    
    log.debug(f"Angle 3D: line({point1}-{point2}) to line({point3}-{point4}) = {angle_deg:.2f}°")
    return float(angle_deg)


def sample_hu_value(point: Tuple[float, float, float],
                   volume_data: np.ndarray,
                   volume_origin: Tuple[float, float, float],
                   voxel_spacing: Tuple[float, float, float]) -> float:
    """Sample HU value at a point in volume data
    
    Args:
        point: (x, y, z) coordinates in mm (world space)
        volume_data: 3D numpy array with HU values
        volume_origin: (x, y, z) origin of volume in mm
        voxel_spacing: (x, y, z) voxel spacing in mm
    
    Returns:
        HU value at point (interpolated if between voxels)
    """
    # Convert world coordinates to voxel indices
    point_arr = np.array(point)
    origin_arr = np.array(volume_origin)
    spacing_arr = np.array(voxel_spacing)
    
    voxel_coords = (point_arr - origin_arr) / spacing_arr
    
    # Round to nearest voxel
    voxel_idx = np.round(voxel_coords).astype(int)
    
    # Check bounds
    shape = volume_data.shape
    if (voxel_idx[0] < 0 or voxel_idx[0] >= shape[0] or
        voxel_idx[1] < 0 or voxel_idx[1] >= shape[1] or
        voxel_idx[2] < 0 or voxel_idx[2] >= shape[2]):
        log.warning(f"Point {point} is outside volume bounds")
        return 0.0
    
    hu_value = volume_data[voxel_idx[0], voxel_idx[1], voxel_idx[2]]
    
    log.debug(f"HU sampling at {point}: {hu_value:.1f} HU")
    return float(hu_value)


def calculate_distance_perpendicular_2d(ref_point1: Tuple[float, float, float],
                                       ref_point2: Tuple[float, float, float],
                                       point_a: Tuple[float, float, float],
                                       point_b: Tuple[float, float, float],
                                       plane: str = 'axial') -> float:
    """Calculate distance between perpendiculars from two points to a reference line
    
    This is used for TT-TG distance:
    - Reference line: posterior condylar line (ref_point1 to ref_point2)
    - Point A: tibial tuberosity
    - Point B: trochlear groove
    
    The function:
    1. Projects all points to the specified plane
    2. Drops perpendiculars from point_a and point_b to the reference line
    3. Calculates distance between the two perpendicular intersection points
    
    Args:
        ref_point1: First point of reference line (x, y, z) in mm
        ref_point2: Second point of reference line (x, y, z) in mm
        point_a: First point to drop perpendicular from (x, y, z) in mm
        point_b: Second point to drop perpendicular from (x, y, z) in mm
        plane: 'axial', 'sagittal', or 'coronal'
    
    Returns:
        Distance in mm between the two perpendicular intersection points
    """
    # Project all points to 2D plane
    ref1_2d = np.array(project_point_to_plane(ref_point1, plane))
    ref2_2d = np.array(project_point_to_plane(ref_point2, plane))
    pa_2d = np.array(project_point_to_plane(point_a, plane))
    pb_2d = np.array(project_point_to_plane(point_b, plane))
    
    # Reference line direction vector
    ref_vec = ref2_2d - ref1_2d
    ref_length = np.linalg.norm(ref_vec)
    ref_unit = ref_vec / ref_length
    
    # Find perpendicular intersection point for point_a
    # Vector from ref_point1 to point_a
    vec_to_a = pa_2d - ref1_2d
    # Project onto reference line
    proj_length_a = np.dot(vec_to_a, ref_unit)
    intersection_a = ref1_2d + proj_length_a * ref_unit
    
    # Find perpendicular intersection point for point_b
    vec_to_b = pb_2d - ref1_2d
    proj_length_b = np.dot(vec_to_b, ref_unit)
    intersection_b = ref1_2d + proj_length_b * ref_unit
    
    # Distance between the two intersection points
    distance = np.linalg.norm(intersection_b - intersection_a)
    
    log.debug(f"Distance Perpendicular 2D ({plane}):")
    log.debug(f"  Reference line: {ref_point1} to {ref_point2}")
    log.debug(f"  Point A: {point_a} -> intersection at {intersection_a}")
    log.debug(f"  Point B: {point_b} -> intersection at {intersection_b}")
    log.debug(f"  Distance: {distance:.2f} mm")
    
    return float(distance)
