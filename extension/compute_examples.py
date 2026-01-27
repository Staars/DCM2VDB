"""Examples of using the compute backend for GPU-accelerated operations

This file demonstrates how to replace numpy operations with GPU-accelerated
equivalents using the backend abstraction layer.
"""

from .compute_backend import xp, backend_name, to_numpy, from_numpy
import numpy as np


def example_volume_threshold(volume_data, threshold_min, threshold_max):
    """GPU-accelerated volume thresholding
    
    Args:
        volume_data: numpy array (Z, Y, X)
        threshold_min: Minimum HU value
        threshold_max: Maximum HU value
    
    Returns:
        numpy array: Binary mask
    """
    # Convert to backend array
    data = from_numpy(volume_data)
    
    # GPU-accelerated operations
    mask = (data >= threshold_min) & (data <= threshold_max)
    
    # Convert back to numpy
    return to_numpy(mask)


def example_volume_resample(volume_data, scale_factor):
    """GPU-accelerated volume resampling (simple nearest neighbor)
    
    Args:
        volume_data: numpy array (Z, Y, X)
        scale_factor: float, scaling factor
    
    Returns:
        numpy array: Resampled volume
    """
    # For simple operations, can work directly with backend
    data = from_numpy(volume_data)
    
    # Example: downsample by taking every Nth voxel
    step = int(1.0 / scale_factor)
    if step > 1:
        resampled = data[::step, ::step, ::step]
        return to_numpy(resampled)
    
    return volume_data


def example_morphological_operations(mask, iterations=1):
    """GPU-accelerated morphological operations
    
    Args:
        mask: numpy array, binary mask
        iterations: int, number of iterations
    
    Returns:
        numpy array: Processed mask
    """
    data = from_numpy(mask.astype(np.float32))
    
    # Simple dilation using convolution-like operation
    # (This is a simplified example - real morphology needs proper kernels)
    for _ in range(iterations):
        # Shift and combine
        dilated = data.copy()
        # This is where you'd implement proper morphological operations
        # using GPU-accelerated convolutions
    
    return to_numpy(dilated > 0.5)


def example_component_labeling_prep(volume_data, threshold):
    """GPU-accelerated preprocessing for component labeling
    
    Args:
        volume_data: numpy array (Z, Y, X)
        threshold: HU threshold value
    
    Returns:
        numpy array: Binary mask ready for labeling
    """
    # This is the slow part that can be GPU-accelerated
    data = from_numpy(volume_data)
    
    # Threshold
    mask = data > threshold
    
    # Optional: Remove small noise with simple filtering
    # (Real implementation would use proper morphological operations)
    
    return to_numpy(mask)


def example_statistics(volume_data, mask=None):
    """GPU-accelerated volume statistics
    
    Args:
        volume_data: numpy array (Z, Y, X)
        mask: optional numpy array, region of interest
    
    Returns:
        dict: Statistics (mean, std, min, max, etc.)
    """
    data = from_numpy(volume_data)
    
    if mask is not None:
        mask_gpu = from_numpy(mask)
        data = data[mask_gpu > 0]
    
    # GPU-accelerated statistics
    stats = {
        'mean': float(to_numpy(xp.mean(data))),
        'std': float(to_numpy(xp.std(data))),
        'min': float(to_numpy(xp.min(data))),
        'max': float(to_numpy(xp.max(data))),
        'median': float(to_numpy(xp.median(data))),
    }
    
    return stats


def example_distance_calculation(points_a, points_b):
    """GPU-accelerated pairwise distance calculation
    
    Args:
        points_a: numpy array (N, 3)
        points_b: numpy array (M, 3)
    
    Returns:
        numpy array (N, M): Distance matrix
    """
    a = from_numpy(points_a)
    b = from_numpy(points_b)
    
    # Compute pairwise distances using broadcasting
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    a_sq = xp.sum(a**2, axis=1, keepdims=True)  # (N, 1)
    b_sq = xp.sum(b**2, axis=1, keepdims=True)  # (M, 1)
    ab = xp.dot(a, b.T)  # (N, M)
    
    distances = xp.sqrt(a_sq + b_sq.T - 2*ab)
    
    return to_numpy(distances)


# Usage example in your existing code:
"""
# In volume_creation.py or similar:

from .compute_backend import xp, backend_name, to_numpy, from_numpy

def process_volume_gpu(volume_data, threshold):
    # Convert to GPU
    data = from_numpy(volume_data)
    
    # GPU operations
    mask = data > threshold
    result = xp.sum(mask)
    
    # Convert back
    return to_numpy(result)
"""
