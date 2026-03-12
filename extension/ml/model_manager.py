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
    onnxruntime package directory so its bundled DLLs can be found.
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


def _diagnose_dll_error():
    """Try to identify exactly which DLL is missing on Windows.
    
    Returns a diagnostic string with details about which VC++ runtime
    DLLs are present/missing.
    """
    import ctypes
    
    lines = []
    # Check the VC++ runtime DLLs that onnxruntime depends on
    for dll_name in ['vcruntime140.dll', 'vcruntime140_1.dll',
                     'msvcp140.dll', 'concrt140.dll']:
        try:
            ctypes.WinDLL(dll_name)
            lines.append(f"  ✓ {dll_name} found")
        except OSError:
            lines.append(f"  ✗ {dll_name} MISSING")
    
    # Check where onnxruntime's own DLLs live
    try:
        import importlib.util
        spec = importlib.util.find_spec('onnxruntime')
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                capi = os.path.join(loc, 'capi')
                if os.path.isdir(capi):
                    dlls = [f for f in os.listdir(capi)
                            if f.endswith(('.dll', '.pyd'))]
                    lines.append(f"  onnxruntime capi dir: {capi}")
                    lines.append(f"  DLLs found: {', '.join(dlls[:10])}")
    except Exception:
        pass
    
    return '\n'.join(lines)


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
        except (ImportError, OSError) as e:
            msg = f"ONNX Runtime not available: {e}"
            if system == 'Windows':
                diag = _diagnose_dll_error()
                msg += f"\n\nDLL diagnostics:\n{diag}"
                msg += (
                    "\n\nThis is usually caused by missing Visual C++ "
                    "Redistributable (2015-2022).\n"
                    "Download: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                    "Or ask IT to install 'Microsoft Visual C++ Redistributable'."
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
