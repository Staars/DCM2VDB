"""
ML-powered segmentation for medical images

This module provides AI segmentation using MedSAM2:
- MLX backend for macOS (Apple Silicon)
- ONNX backend for Windows/Linux

Usage:
    from . import ml
    predictor = ml.get_predictor()  # Auto-detects platform
    mask = predictor.segment(image, points)
"""

from .model_manager import get_predictor

__all__ = ['get_predictor']
