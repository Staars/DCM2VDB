"""GPU-accelerated compute backend abstraction layer

Automatically selects the best available backend:
- MLX for Apple Silicon (macOS ARM64)
- CuPy for NVIDIA CUDA
- NumPy as CPU fallback

Usage:
    from .compute_backend import xp, backend_name
    
    # Use xp like numpy
    array = xp.array([1, 2, 3])
    result = xp.sum(array)
    
    # Convert back to numpy if needed
    numpy_array = to_numpy(result)
"""

import sys
import platform
from .utils import SimpleLogger

log = SimpleLogger()

# Backend selection
_backend = None
_backend_name = 'numpy'
xp = None  # Will be set to the selected backend module


def _detect_backend():
    """Detect and initialize the best available compute backend"""
    global _backend, _backend_name, xp
    
    system = platform.system()
    machine = platform.machine()
    
    # Try MLX for Apple Silicon
    if system == 'Darwin' and machine == 'arm64':
        try:
            import mlx.core as mx
            _backend = mx
            _backend_name = 'mlx'
            xp = mx
            log.info(f"✓ Using MLX backend (Apple Silicon GPU acceleration)")
            return
        except ImportError as e:
            log.warning(f"MLX not available: {e}")
    
    # Try CuPy for NVIDIA CUDA (only on Windows/Linux)
    if system in ['Windows', 'Linux']:
        try:
            import cupy as cp
            # Test if CUDA is actually available
            try:
                cp.cuda.Device(0).compute_capability
                _backend = cp
                _backend_name = 'cupy'
                xp = cp
                log.info(f"✓ Using CuPy backend (NVIDIA CUDA GPU acceleration)")
                return
            except cp.cuda.runtime.CUDARuntimeError:
                log.warning("CuPy installed but no CUDA device found")
        except ImportError:
            pass
    
    # Fallback to NumPy
    import numpy as np
    _backend = np
    _backend_name = 'numpy'
    xp = np
    log.info(f"Using NumPy backend (CPU only)")


def to_numpy(array):
    """Convert backend array to numpy array
    
    Args:
        array: Array from any backend (MLX, CuPy, or NumPy)
    
    Returns:
        numpy.ndarray
    """
    if _backend_name == 'numpy':
        return array
    elif _backend_name == 'mlx':
        import numpy as np
        return np.array(array)
    elif _backend_name == 'cupy':
        return array.get()
    else:
        # Unknown backend, try to convert
        import numpy as np
        return np.asarray(array)


def from_numpy(array):
    """Convert numpy array to backend array
    
    Args:
        array: numpy.ndarray
    
    Returns:
        Backend array (MLX, CuPy, or NumPy)
    """
    if _backend_name == 'numpy':
        return array
    else:
        return xp.array(array)


def get_backend_info():
    """Get information about the current backend
    
    Returns:
        dict: Backend information including name, device, and capabilities
    """
    info = {
        'name': _backend_name,
        'module': str(_backend),
        'gpu_accelerated': _backend_name in ['mlx', 'cupy']
    }
    
    if _backend_name == 'mlx':
        try:
            import mlx.core as mx
            device_info = mx.metal.device_info()
            
            info['device'] = device_info.get('device_name', 'Apple Silicon GPU')
            info['platform'] = f"{platform.system()} {platform.machine()}"
            info['metal_available'] = mx.metal.is_available()
            
            # Add detailed Metal device info
            if 'architecture' in device_info:
                info['architecture'] = device_info['architecture']
            if 'max_buffer_length' in device_info:
                info['max_buffer_size'] = f"{device_info['max_buffer_length'] / 1024**3:.1f} GB"
            if 'max_recommended_working_set_size' in device_info:
                info['recommended_memory'] = f"{device_info['max_recommended_working_set_size'] / 1024**3:.1f} GB"
            if 'memory_size' in device_info:
                info['total_memory'] = f"{device_info['memory_size'] / 1024**3:.1f} GB"
        except Exception as e:
            info['device'] = 'Apple Silicon GPU'
            info['error'] = str(e)
    elif _backend_name == 'cupy':
        import cupy as cp
        try:
            device = cp.cuda.Device()
            info['device'] = device.name.decode('utf-8')
            info['compute_capability'] = device.compute_capability
            info['memory_total'] = f"{device.mem_info[1] / 1024**3:.1f} GB"
            info['memory_free'] = f"{device.mem_info[0] / 1024**3:.1f} GB"
        except:
            info['device'] = 'CUDA device (details unavailable)'
    else:
        info['device'] = 'CPU'
        info['platform'] = f"{platform.system()} {platform.machine()}"
    
    return info


# Initialize backend on module import
_detect_backend()

# Export backend name for conditional code
backend_name = _backend_name
