"""
Model manager - handles loading and platform detection
"""

import sys
import platform
from pathlib import Path


def get_model_path():
    """Get the path to the models directory"""
    # Models are bundled in extension/ml/ (medsam2_mlx/ and medsam2_onnx/)
    extension_dir = Path(__file__).parent
    models_dir = extension_dir  # ml directory itself contains the model folders
    
    return models_dir


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
        try:
            import onnxruntime
            return "onnx"
        except ImportError:
            raise RuntimeError(
                "ONNX Runtime not available. "
                "Make sure onnxruntime is installed."
            )


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
