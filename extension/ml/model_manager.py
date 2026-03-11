"""
Model manager - handles loading and platform detection
"""

import sys
import os
import platform
from pathlib import Path


def get_model_path():
    """Get the path to the models directory"""
    # Models are bundled in extension/ml/ (medsam2_mlx/ and medsam2_onnx/)
    extension_dir = Path(__file__).parent
    models_dir = extension_dir  # ml directory itself contains the model folders
    
    return models_dir


def _setup_onnx_dll_dirs():
    """Register DLL search directories for ONNX Runtime on Windows.
    
    Python 3.8+ on Windows restricts DLL search paths. We add the
    onnxruntime package directory so its bundled DLLs can be found,
    and also add any VC++ runtime directories that may be present
    in site-packages.
    """
    if platform.system() != 'Windows':
        return
    
    try:
        import importlib.util
        spec = importlib.util.find_spec('onnxruntime')
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                ort_dir = Path(loc)
                # onnxruntime ships DLLs in capi/ subdirectory
                capi_dir = ort_dir / 'capi'
                for d in [ort_dir, capi_dir]:
                    if d.is_dir():
                        os.add_dll_directory(str(d))
    except Exception:
        pass


def detect_platform():
    """
    Detect the best available backend
    
    Returns:
        str: 'mlx' or 'onnx'
    """
    # Platform-specific backends:
    # - macOS ARM64: MLX (Apple Silicon Metal GPU)
    # - Windows/Linux: ONNX Runtime
    
    system = platform.system()
    machine = platform.machine()
    
    # macOS ARM64 -> MLX
    if system == 'Darwin' and machine == 'arm64':
        try:
            import mlx.core
            return "mlx"
        except ImportError:
            raise RuntimeError(
                "MLX not available on macOS ARM64. "
                "Make sure mlx is installed."
            )
    
    # Windows/Linux -> ONNX
    else:
        _setup_onnx_dll_dirs()
        try:
            import onnxruntime
            return "onnx"
        except ImportError as e:
            msg = f"ONNX Runtime not available: {e}"
            if system == 'Windows' and 'DLL' in str(e):
                msg += (
                    "\n\nThis is usually caused by missing Visual C++ "
                    "Redistributable. Install it from:\n"
                    "https://aka.ms/vs/17/release/vc_redist.x64.exe"
                )
            raise RuntimeError(msg)


def get_predictor():
    """
    Get the predictor for the current platform
    
    Returns:
        Predictor instance (MLXPredictor or ONNXPredictor)
    """
    backend = detect_platform()
    models_dir = get_model_path()
    
    if backend == "mlx":
        try:
            from .inference_mlx import MLXPredictor
            return MLXPredictor(models_dir)
        except ImportError as e:
            raise RuntimeError(
                f"MLX backend not available: {e}\n"
                "Make sure mlx is installed."
            )
    
    elif backend == "onnx":
        try:
            from .inference_onnx import ONNXPredictor
            return ONNXPredictor(models_dir)
        except ImportError as e:
            raise RuntimeError(
                f"ONNX backend not available: {e}\n"
                "Make sure onnxruntime is installed."
            )
    
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")
