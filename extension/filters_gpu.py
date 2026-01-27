"""GPU-accelerated image filters using MLX

Replaces scipy.ndimage filters with GPU-accelerated MLX implementations.
"""

import numpy as np
from .compute_backend import xp, backend_name, to_numpy, from_numpy
from .utils import SimpleLogger

log = SimpleLogger()


def create_gaussian_kernel_2d(sigma, size=None):
    """Create 2D Gaussian kernel
    
    Args:
        sigma: Standard deviation
        size: Kernel size (if None, auto-calculate from sigma)
    
    Returns:
        2D kernel array
    """
    if size is None:
        # Auto-size: 6*sigma covers 99.7% of distribution
        size = int(np.ceil(sigma * 6))
        if size % 2 == 0:
            size += 1  # Make odd
    
    # Create coordinate grids
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    
    # Gaussian formula
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    
    # Normalize
    kernel = kernel / np.sum(kernel)
    
    return kernel


def gaussian_filter_gpu(image, sigma):
    """GPU-accelerated 2D Gaussian filter
    
    Args:
        image: 2D numpy array
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        Filtered 2D numpy array
    """
    if backend_name == 'numpy':
        # Fallback to scipy if no GPU
        from scipy import ndimage
        return ndimage.gaussian_filter(image, sigma=sigma)
    
    try:
        if backend_name == 'mlx':
            import mlx.core as mx
            
            # Create 1D Gaussian kernel for separable convolution
            size_1d = int(np.ceil(sigma * 6))
            if size_1d % 2 == 0:
                size_1d += 1
            
            ax = np.arange(-size_1d // 2 + 1., size_1d // 2 + 1.)
            kernel_1d = np.exp(-(ax**2) / (2. * sigma**2))
            kernel_1d = kernel_1d / np.sum(kernel_1d)
            
            # Transfer to GPU
            image_gpu = from_numpy(image.astype(np.float32))
            kernel_1d_gpu = from_numpy(kernel_1d.astype(np.float32))
            
            # Reshape image: (H, W) -> (1, H, W, 1) for conv2d
            # MLX conv2d expects: input (N, H, W, C_in)
            image_4d = mx.reshape(image_gpu, (1, image_gpu.shape[0], image_gpu.shape[1], 1))
            
            # Horizontal pass
            # MLX weight format: (C_out, KH, KW, C_in)
            # For horizontal 1D: (1, 1, K, 1) - 1 output channel, height=1, width=K, 1 input channel
            kernel_h = mx.reshape(kernel_1d_gpu, (1, 1, size_1d, 1))
            result_h = mx.conv2d(image_4d, kernel_h, padding=(0, size_1d//2))
            
            # Vertical pass
            # For vertical 1D: (1, K, 1, 1) - 1 output channel, height=K, width=1, 1 input channel
            kernel_v = mx.reshape(kernel_1d_gpu, (1, size_1d, 1, 1))
            result_4d = mx.conv2d(result_h, kernel_v, padding=(size_1d//2, 0))
            
            # Force evaluation
            mx.eval(result_4d)
            
            # Extract 2D result: (1, H, W, 1) -> (H, W)
            result_2d = mx.squeeze(result_4d)
            
            # Ensure 2D
            if result_2d.ndim != 2:
                raise ValueError(f"Unexpected shape after separable conv: {result_4d.shape} -> {result_2d.shape}")
            
            return to_numpy(result_2d)
        
        elif backend_name == 'cupy':
            # CuPy has GPU-accelerated scipy.ndimage functions
            from cupyx.scipy import ndimage as cp_ndimage
            
            # Transfer to GPU
            image_gpu = from_numpy(image.astype(np.float32))
            
            # Apply Gaussian filter on GPU
            result_gpu = cp_ndimage.gaussian_filter(image_gpu, sigma=sigma)
            
            return to_numpy(result_gpu)
        
        else:
            # Unknown backend, fallback
            from scipy import ndimage
            return ndimage.gaussian_filter(image, sigma=sigma)
        
    except Exception as e:
        log.warning(f"GPU Gaussian filter failed, falling back to scipy: {e}")
        from scipy import ndimage
        return ndimage.gaussian_filter(image, sigma=sigma)


def gaussian_filter_3d_gpu(volume, sigma):
    """GPU-accelerated 3D Gaussian filter
    
    Uses separable convolution with conv3d: 3 passes (X, Y, Z) instead of full 3D kernel.
    This is much more efficient: O(3*n*k) vs O(n*k^3)
    
    Args:
        volume: 3D numpy array (Z, Y, X)
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        Filtered 3D numpy array
    """
    if backend_name == 'numpy':
        # Fallback to scipy if no GPU
        from scipy import ndimage
        return ndimage.gaussian_filter(volume, sigma=sigma)
    
    try:
        if backend_name == 'mlx':
            import mlx.core as mx
            
            # Create 1D Gaussian kernel
            size_1d = int(np.ceil(sigma * 6))
            if size_1d % 2 == 0:
                size_1d += 1
            
            ax = np.arange(-size_1d // 2 + 1., size_1d // 2 + 1.)
            kernel_1d = np.exp(-(ax**2) / (2. * sigma**2))
            kernel_1d = kernel_1d / np.sum(kernel_1d)
            
            # Transfer to GPU
            volume_gpu = from_numpy(volume.astype(np.float32))
            kernel_1d_gpu = from_numpy(kernel_1d.astype(np.float32))
            
            Z, Y, X = volume_gpu.shape
            
            # Reshape volume for conv3d: (Z, Y, X) -> (1, Z, Y, X, 1)
            # MLX conv3d expects: input (N, D, H, W, C_in)
            volume_5d = mx.reshape(volume_gpu, (1, Z, Y, X, 1))
            
            # Pass 1: Filter along X axis (width)
            # MLX weight format: (C_out, KD, KH, KW, C_in)
            # For X-axis 1D: (1, 1, 1, K, 1) - 1 output, depth=1, height=1, width=K, 1 input
            kernel_x = mx.reshape(kernel_1d_gpu, (1, 1, 1, size_1d, 1))
            result_x = mx.conv3d(volume_5d, kernel_x, padding=(0, 0, size_1d//2))
            
            # Pass 2: Filter along Y axis (height)
            # For Y-axis 1D: (1, 1, K, 1, 1) - 1 output, depth=1, height=K, width=1, 1 input
            kernel_y = mx.reshape(kernel_1d_gpu, (1, 1, size_1d, 1, 1))
            result_y = mx.conv3d(result_x, kernel_y, padding=(0, size_1d//2, 0))
            
            # Pass 3: Filter along Z axis (depth)
            # For Z-axis 1D: (1, K, 1, 1, 1) - 1 output, depth=K, height=1, width=1, 1 input
            kernel_z = mx.reshape(kernel_1d_gpu, (1, size_1d, 1, 1, 1))
            result_5d = mx.conv3d(result_y, kernel_z, padding=(size_1d//2, 0, 0))
            
            # Force evaluation
            mx.eval(result_5d)
            
            # Extract 3D result: (1, Z, Y, X, 1) -> (Z, Y, X)
            result_3d = mx.squeeze(result_5d)
            
            # Ensure 3D
            if result_3d.ndim != 3:
                raise ValueError(f"Unexpected shape after 3D separable conv: {result_5d.shape} -> {result_3d.shape}")
            
            return to_numpy(result_3d)
        
        elif backend_name == 'cupy':
            # CuPy has GPU-accelerated scipy.ndimage functions
            from cupyx.scipy import ndimage as cp_ndimage
            
            # Transfer to GPU
            volume_gpu = from_numpy(volume.astype(np.float32))
            
            # Apply 3D Gaussian filter on GPU
            result_gpu = cp_ndimage.gaussian_filter(volume_gpu, sigma=sigma)
            
            return to_numpy(result_gpu)
        
        else:
            # Unknown backend, fallback
            from scipy import ndimage
            return ndimage.gaussian_filter(volume, sigma=sigma)
        
    except Exception as e:
        log.warning(f"GPU 3D Gaussian filter failed, falling back to scipy: {e}")
        from scipy import ndimage
        return ndimage.gaussian_filter(volume, sigma=sigma)


def percentile_filter_gpu(image, percentile, size=3):
    """GPU-accelerated percentile filter
    
    Args:
        image: 2D numpy array
        percentile: Percentile value (0-100)
        size: Filter window size (odd number)
    
    Returns:
        Filtered 2D numpy array
    """
    if backend_name == 'numpy':
        # Fallback to scipy
        from scipy import ndimage
        return ndimage.percentile_filter(image, percentile=percentile, size=size)
    
    try:
        if backend_name == 'mlx':
            import mlx.core as mx
            
            # Transfer to GPU
            image_gpu = from_numpy(image.astype(np.float32))
            H, W = image_gpu.shape
            
            # Pad image with edge values
            pad = size // 2
            image_padded = mx.pad(image_gpu, pad, mode='edge')
            
            # Create sliding windows by extracting shifted views
            windows = []
            for dy in range(size):
                for dx in range(size):
                    window = image_padded[dy:dy+H, dx:dx+W]
                    windows.append(window)
            
            # Stack all windows: (size*size, H, W)
            windows_stack = mx.stack(windows, axis=0)
            
            # Calculate the index for the desired percentile
            # For 9 values (3×3): 25th percentile ≈ index 2, 75th ≈ index 6
            num_values = size * size
            kth = int(np.round(percentile / 100.0 * (num_values - 1)))
            kth = max(0, min(kth, num_values - 1))  # Clamp to valid range
            
            # Partition along the window dimension (axis=0)
            # This puts the kth smallest element at index kth
            partitioned = mx.partition(windows_stack, kth, axis=0)
            
            # Extract the kth element (the percentile value)
            result = partitioned[kth, :, :]
            
            # Force evaluation
            mx.eval(result)
            
            return to_numpy(result)
        
        elif backend_name == 'cupy':
            # CuPy has GPU-accelerated scipy.ndimage functions
            from cupyx.scipy import ndimage as cp_ndimage
            
            # Transfer to GPU
            image_gpu = from_numpy(image.astype(np.float32))
            
            # Apply percentile filter on GPU
            result_gpu = cp_ndimage.percentile_filter(image_gpu, percentile=percentile, size=size)
            
            return to_numpy(result_gpu)
        
        else:
            # Unknown backend, fallback
            from scipy import ndimage
            return ndimage.percentile_filter(image, percentile=percentile, size=size)
        
    except Exception as e:
        log.warning(f"GPU percentile filter failed, falling back to scipy: {e}")
        from scipy import ndimage
        return ndimage.percentile_filter(image, percentile=percentile, size=size)


def median_filter_gpu(image, size=3):
    """GPU-accelerated median filter (50th percentile)
    
    Args:
        image: 2D numpy array
        size: Filter window size (odd number)
    
    Returns:
        Filtered 2D numpy array
    """
    if backend_name == 'numpy':
        # Fallback to scipy
        from scipy import ndimage
        return ndimage.median_filter(image, size=size)
    
    try:
        if backend_name == 'mlx':
            import mlx.core as mx
            
            # Transfer to GPU
            image_gpu = from_numpy(image.astype(np.float32))
            H, W = image_gpu.shape
            
            # Pad image with edge values
            pad = size // 2
            image_padded = mx.pad(image_gpu, pad, mode='edge')
            
            # Create sliding windows by extracting shifted views
            # For a 3×3 kernel, we need 9 shifted copies
            windows = []
            for dy in range(size):
                for dx in range(size):
                    # Extract shifted view: [dy:dy+H, dx:dx+W]
                    window = image_padded[dy:dy+H, dx:dx+W]
                    windows.append(window)
            
            # Stack all windows: (size*size, H, W)
            windows_stack = mx.stack(windows, axis=0)
            
            # Compute median along the window dimension (axis=0)
            result = mx.median(windows_stack, axis=0)
            
            # Force evaluation
            mx.eval(result)
            
            return to_numpy(result)
        
        elif backend_name == 'cupy':
            # CuPy has GPU-accelerated scipy.ndimage functions
            from cupyx.scipy import ndimage as cp_ndimage
            
            # Transfer to GPU
            image_gpu = from_numpy(image.astype(np.float32))
            
            # Apply median filter on GPU
            result_gpu = cp_ndimage.median_filter(image_gpu, size=size)
            
            return to_numpy(result_gpu)
        
        else:
            # Unknown backend, fallback
            from scipy import ndimage
            return ndimage.median_filter(image, size=size)
        
    except Exception as e:
        log.warning(f"GPU median filter failed, falling back to scipy: {e}")
        from scipy import ndimage
        return ndimage.median_filter(image, size=size)


def denoise_slice_gpu(
    slice_array,
    method='GAUSSIAN',
    strength=1.0
):
    """GPU-accelerated denoising for a single 2D slice
    
    Replaces scipy.ndimage filters with MLX GPU implementations.
    
    Args:
        slice_array: 2D numpy array (single slice)
        method: Denoising method ('GAUSSIAN', 'PERCENTILE_25', 'PERCENTILE_75', 'MEDIAN')
        strength: Filter strength (0.01-1.0)
    
    Returns:
        Denoised 2D numpy array
    """
    from .constants import (
        DENOISING_PERCENTILE_BLEND_MULTIPLIER,
        DENOISING_MEDIAN_KERNEL_SIZE
    )
    
    if method == 'GAUSSIAN':
        # Gaussian filter - sigma scales with strength
        result = gaussian_filter_gpu(slice_array, sigma=strength)
    
    elif method == 'PERCENTILE_25':
        # 25th percentile filter
        size = DENOISING_MEDIAN_KERNEL_SIZE
        filtered = percentile_filter_gpu(slice_array, percentile=25, size=size)
        
        # Blend with original
        blend_factor = strength * DENOISING_PERCENTILE_BLEND_MULTIPLIER
        blend_factor = min(blend_factor, 1.0)
        result = (1.0 - blend_factor) * slice_array + blend_factor * filtered
        log.info(f"  GPU Percentile 25% filter with {blend_factor*100:.1f}% blend")
    
    elif method == 'PERCENTILE_75':
        # 75th percentile filter
        size = DENOISING_MEDIAN_KERNEL_SIZE
        filtered = percentile_filter_gpu(slice_array, percentile=75, size=size)
        
        # Blend with original
        blend_factor = strength * DENOISING_PERCENTILE_BLEND_MULTIPLIER
        blend_factor = min(blend_factor, 1.0)
        result = (1.0 - blend_factor) * slice_array + blend_factor * filtered
        log.info(f"  GPU Percentile 75% filter with {blend_factor*100:.1f}% blend")
    
    elif method == 'MEDIAN':
        # Median filter
        size = DENOISING_MEDIAN_KERNEL_SIZE
        filtered = median_filter_gpu(slice_array, size=size)
        
        # Blend with original
        blend_factor = strength * DENOISING_PERCENTILE_BLEND_MULTIPLIER
        blend_factor = min(blend_factor, 1.0)
        result = (1.0 - blend_factor) * slice_array + blend_factor * filtered
        log.info(f"  GPU Median filter with {blend_factor*100:.1f}% blend")
    
    else:
        result = slice_array
    
    return result.astype(slice_array.dtype)


def denoise_volume_batch_gpu(volume, method='GAUSSIAN', strength=1.0):
    """GPU-accelerated batch denoising for entire 3D volume
    
    Processes all slices at once on GPU for maximum performance.
    
    Args:
        volume: 3D numpy array (Z, Y, X)
        method: Denoising method
        strength: Filter strength
    
    Returns:
        Denoised 3D numpy array
    """
    if backend_name == 'numpy':
        # Fallback to slice-by-slice scipy
        from .volume_utils import denoise_slice_scipy
        result = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            result[i] = denoise_slice_scipy(volume[i], method=method, strength=strength)
        return result
    
    log.info(f"GPU batch denoising: {volume.shape}, method={method}, backend={backend_name}")
    
    # Process all slices on GPU
    result = np.zeros_like(volume)
    
    for i in range(volume.shape[0]):
        result[i] = denoise_slice_gpu(volume[i], method=method, strength=strength)
        
        # Log progress
        if (i + 1) % 36 == 0 or i == volume.shape[0] - 1:
            progress = int((i + 1) / volume.shape[0] * 100)
            log.info(f"  GPU denoising: {progress}% ({i+1}/{volume.shape[0]} slices)")
    
    return result
