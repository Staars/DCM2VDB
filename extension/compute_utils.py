"""GPU-accelerated volume processing utilities"""

from .compute_backend import xp, backend_name, to_numpy, from_numpy
from .utils import SimpleLogger

log = SimpleLogger()


def threshold_volume_gpu(volume_data, min_val, max_val):
    """GPU-accelerated volume thresholding
    
    Args:
        volume_data: numpy array (Z, Y, X)
        min_val: minimum threshold value
        max_val: maximum threshold value
    
    Returns:
        numpy array: thresholded volume (values outside range set to 0)
    """
    try:
        # GPU path
        log.debug(f"GPU thresholding: {volume_data.shape}, backend={backend_name}")
        
        data_gpu = from_numpy(volume_data)
        mask_gpu = (data_gpu >= min_val) & (data_gpu <= max_val)
        result_gpu = data_gpu * mask_gpu
        
        # Force evaluation for MLX lazy execution
        if backend_name == 'mlx':
            import mlx.core as mx
            mx.eval(result_gpu)
        
        return to_numpy(result_gpu)
        
    except Exception as e:
        log.warning(f"GPU thresholding failed, falling back to CPU: {e}")
        # Fallback to CPU
        mask = (volume_data >= min_val) & (volume_data <= max_val)
        return volume_data * mask


def calculate_volume_statistics_gpu(volume_data, mask=None):
    """GPU-accelerated volume statistics
    
    Args:
        volume_data: numpy array
        mask: optional boolean mask (same shape as volume_data)
    
    Returns:
        dict: statistics (mean, std, min, max, count)
    """
    try:
        data_gpu = from_numpy(volume_data)
        
        if mask is not None:
            mask_gpu = from_numpy(mask)
            
            if backend_name == 'mlx':
                # MLX doesn't support boolean indexing, use where
                masked = xp.where(mask_gpu, data_gpu, 0)
                count = xp.sum(mask_gpu)
                sum_val = xp.sum(masked)
                mean_val = sum_val / xp.maximum(count, 1)
                
                # Std calculation
                diff = xp.where(mask_gpu, data_gpu - mean_val, 0)
                variance = xp.sum(diff * diff) / xp.maximum(count, 1)
                std_val = xp.sqrt(variance)
                
                # Min/max with masking (use large sentinels)
                min_val = xp.min(xp.where(mask_gpu, data_gpu, xp.array(1e10)))
                max_val = xp.max(xp.where(mask_gpu, data_gpu, xp.array(-1e10)))
                
                # Force evaluation
                import mlx.core as mx
                mx.eval(mean_val, std_val, min_val, max_val, count)
            else:
                # CuPy/NumPy support boolean indexing
                masked_data = data_gpu[mask_gpu]
                mean_val = xp.mean(masked_data)
                std_val = xp.std(masked_data)
                min_val = xp.min(masked_data)
                max_val = xp.max(masked_data)
                count = xp.sum(mask_gpu)
        else:
            # No mask - simple statistics
            mean_val = xp.mean(data_gpu)
            std_val = xp.std(data_gpu)
            min_val = xp.min(data_gpu)
            max_val = xp.max(data_gpu)
            count = data_gpu.size
            
            if backend_name == 'mlx':
                import mlx.core as mx
                mx.eval(mean_val, std_val, min_val, max_val)
        
        return {
            'mean': float(to_numpy(mean_val)),
            'std': float(to_numpy(std_val)),
            'min': float(to_numpy(min_val)),
            'max': float(to_numpy(max_val)),
            'count': int(to_numpy(count))
        }
        
    except Exception as e:
        log.warning(f"GPU statistics failed, falling back to CPU: {e}")
        # Fallback to CPU
        import numpy as np
        if mask is not None:
            masked_data = volume_data[mask]
            return {
                'mean': float(np.mean(masked_data)),
                'std': float(np.std(masked_data)),
                'min': float(np.min(masked_data)),
                'max': float(np.max(masked_data)),
                'count': int(np.sum(mask))
            }
        else:
            return {
                'mean': float(np.mean(volume_data)),
                'std': float(np.std(volume_data)),
                'min': float(np.min(volume_data)),
                'max': float(np.max(volume_data)),
                'count': int(volume_data.size)
            }
