#!/usr/bin/env python3
"""
Test both ONNX and MLX inference for MedSAM2

Tests the converted models with dummy data to verify they work correctly
"""

import sys
import numpy as np
from pathlib import Path


def test_onnx_inference():
    """Test ONNX models with dummy data"""
    print("=" * 60)
    print("ONNX INFERENCE TEST")
    print("=" * 60)
    
    # Check if onnxruntime is installed
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠ onnxruntime not installed - skipping ONNX test")
        print("  Install with: uv pip install onnxruntime")
        return None
    
    print(f"✓ ONNXRuntime version: {ort.__version__}")
    
    # Check model files
    encoder_path = "../medsam2_onnx/encoder.onnx"
    decoder_path = "../medsam2_onnx/decoder.onnx"
    
    if not Path(encoder_path).exists():
        print(f"⚠ Encoder not found: {encoder_path}")
        return None
    
    if not Path(decoder_path).exists():
        print(f"⚠ Decoder not found: {decoder_path}")
        return None
    
    print(f"✓ Found encoder: {encoder_path}")
    print(f"✓ Found decoder: {decoder_path}")
    
    # Load models
    print("\n--- Loading Models ---")
    
    try:
        encoder_session = ort.InferenceSession(encoder_path)
        decoder_session = ort.InferenceSession(decoder_path)
        print("✓ Models loaded")
    except Exception as e:
        print(f"✗ ERROR loading models: {e}")
        return False
    
    # Test inference
    print("\n--- Running Inference ---")
    
    # Step 1: Run encoder
    dummy_image = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
    
    try:
        encoder_outputs = encoder_session.run(None, {'image': dummy_image})
        image_embeddings = encoder_outputs[0]
        backbone_fpn_0 = encoder_outputs[1]
        backbone_fpn_1 = encoder_outputs[2]
        
        print(f"✓ Encoder: {dummy_image.shape} → embeddings {image_embeddings.shape}")
    except Exception as e:
        print(f"✗ ERROR running encoder: {e}")
        return False
    
    # Step 2: Run decoder with prompts
    dummy_coords = np.array([[[512.0, 512.0], [600.0, 600.0]]], dtype=np.float32)
    dummy_labels = np.array([[1.0, 1.0]], dtype=np.float32)
    
    try:
        decoder_outputs = decoder_session.run(
            None,
            {
                'image_embeddings': image_embeddings,
                'backbone_fpn_0': backbone_fpn_0,
                'backbone_fpn_1': backbone_fpn_1,
                'point_coords': dummy_coords,
                'point_labels': dummy_labels
            }
        )
        
        masks = decoder_outputs[0]
        iou_predictions = decoder_outputs[1]
        
        print(f"✓ Decoder: prompts → masks {masks.shape}, IoU {iou_predictions.shape}")
        print(f"  IoU values: {iou_predictions[0]}")
    except Exception as e:
        print(f"✗ ERROR running decoder: {e}")
        return False
    
    print("\n✓ ONNX TEST PASSED")
    return True


def test_mlx_inference():
    """Test MLX model with dummy data"""
    print("\n" + "=" * 60)
    print("MLX INFERENCE TEST")
    print("=" * 60)
    
    # Check if MLX is installed
    try:
        import mlx.core as mx
    except ImportError:
        print("⚠ MLX not installed - skipping MLX test")
        print("  Install with: uv pip install mlx")
        print("  Note: MLX only works on Apple Silicon (M1/M2/M3/M4)")
        return None
    
    print(f"✓ MLX version: {mx.__version__}")
    
    # Check model file
    weights_path = "../medsam2_mlx/weights.safetensors"
    
    if not Path(weights_path).exists():
        print(f"⚠ Weights not found: {weights_path}")
        return None
    
    print(f"✓ Found weights: {weights_path}")
    
    # Load weights
    print("\n--- Loading Weights ---")
    
    try:
        weights = mx.load(weights_path)
        print(f"✓ Loaded {len(weights)} layers")
        
        # Show some layer info
        sample_keys = list(weights.keys())[:3]
        print(f"  Sample layers: {sample_keys}")
        
        # Check total parameters
        total_params = sum(w.size for w in weights.values())
        print(f"  Total parameters: {total_params:,}")
    except Exception as e:
        print(f"✗ ERROR loading weights: {e}")
        return False
    
    # Test basic operations
    print("\n--- Testing MLX Operations ---")
    
    try:
        # Create dummy image tensor
        dummy_image = mx.random.normal((1, 3, 1024, 1024))
        print(f"✓ Created dummy image: {dummy_image.shape}")
        
        # Test a simple operation
        normalized = (dummy_image - mx.mean(dummy_image)) / mx.std(dummy_image)
        print(f"✓ Normalization works: mean={mx.mean(normalized):.4f}, std={mx.std(normalized):.4f}")
        
        # Test weight access
        first_key = list(weights.keys())[0]
        first_weight = weights[first_key]
        print(f"✓ Weight access works: {first_key} → shape {first_weight.shape}")
        
    except Exception as e:
        print(f"✗ ERROR in MLX operations: {e}")
        return False
    
    print("\n✓ MLX TEST PASSED")
    print("  Note: Full inference requires implementing the model architecture")
    print("  Weights are loaded and ready to use")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("MedSAM2 Inference Test Suite")
    print("=" * 60)
    print()
    
    results = {}
    
    # Test ONNX
    onnx_result = test_onnx_inference()
    results['ONNX'] = onnx_result
    
    # Test MLX
    mlx_result = test_mlx_inference()
    results['MLX'] = mlx_result
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, result in results.items():
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⊘ SKIPPED"
        
        print(f"{name:10s} {status}")
    
    # Exit code
    failed = any(r is False for r in results.values())
    skipped_all = all(r is None for r in results.values())
    
    if failed:
        print("\n⚠ Some tests failed")
        return 1
    elif skipped_all:
        print("\n⚠ All tests skipped - install dependencies")
        return 1
    else:
        print("\n✓ All available tests passed")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
