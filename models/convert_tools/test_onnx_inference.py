#!/usr/bin/env python3
"""
Test ONNX inference for MedSAM2

Tests the split encoder/decoder ONNX models with a dummy image
"""

import numpy as np
from pathlib import Path


def test_onnx_inference():
    """Test ONNX models with dummy data"""
    print("=" * 60)
    print("MedSAM2 ONNX Inference Test")
    print("=" * 60)
    
    # Check if onnxruntime is installed
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed")
        print("Install with: uv pip install onnxruntime")
        return False
    
    print(f"✓ ONNXRuntime version: {ort.__version__}")
    
    # Check model files
    encoder_path = "../medsam2_onnx/encoder.onnx"
    decoder_path = "../medsam2_onnx/decoder.onnx"
    
    if not Path(encoder_path).exists():
        print(f"ERROR: Encoder not found: {encoder_path}")
        return False
    
    if not Path(decoder_path).exists():
        print(f"ERROR: Decoder not found: {decoder_path}")
        return False
    
    print(f"✓ Found encoder: {encoder_path}")
    print(f"✓ Found decoder: {decoder_path}")
    
    # Load models
    print("\n" + "=" * 60)
    print("Loading ONNX Models")
    print("=" * 60)
    
    try:
        encoder_session = ort.InferenceSession(encoder_path)
        print("✓ Encoder loaded")
        
        decoder_session = ort.InferenceSession(decoder_path)
        print("✓ Decoder loaded")
    except Exception as e:
        print(f"ERROR loading models: {e}")
        return False
    
    # Print model info
    print("\n--- Encoder Info ---")
    print("Inputs:")
    for inp in encoder_session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    print("Outputs:")
    for out in encoder_session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")
    
    print("\n--- Decoder Info ---")
    print("Inputs:")
    for inp in decoder_session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    print("Outputs:")
    for out in decoder_session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")
    
    # Test inference
    print("\n" + "=" * 60)
    print("Running Inference Test")
    print("=" * 60)
    
    # Step 1: Run encoder
    print("\n--- Step 1: Encode Image ---")
    dummy_image = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
    print(f"Input image shape: {dummy_image.shape}")
    
    try:
        encoder_outputs = encoder_session.run(None, {'image': dummy_image})
        
        # Get the main outputs (first 3)
        image_embeddings = encoder_outputs[0]
        backbone_fpn_0 = encoder_outputs[1]
        backbone_fpn_1 = encoder_outputs[2]
        
        print(f"✓ Image embeddings: {image_embeddings.shape}")
        print(f"✓ Backbone FPN 0: {backbone_fpn_0.shape}")
        print(f"✓ Backbone FPN 1: {backbone_fpn_1.shape}")
    except Exception as e:
        print(f"ERROR running encoder: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Run decoder with prompts
    print("\n--- Step 2: Decode with Prompts ---")
    
    # Create dummy prompts (2 points)
    dummy_coords = np.array([[[512.0, 512.0], [600.0, 600.0]]], dtype=np.float32)  # Center + offset
    dummy_labels = np.array([[1.0, 1.0]], dtype=np.float32)  # Both positive points
    
    print(f"Point coords: {dummy_coords.shape}")
    print(f"Point labels: {dummy_labels.shape}")
    
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
        
        print(f"✓ Masks: {masks.shape}")
        print(f"✓ IoU predictions: {iou_predictions.shape}")
        print(f"  IoU values: {iou_predictions}")
    except Exception as e:
        print(f"ERROR running decoder: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
    print("✓ Encoder works correctly")
    print("✓ Decoder works correctly")
    print("✓ Full pipeline functional")
    print("\nOutput shapes:")
    print(f"  Masks: {masks.shape}")
    print(f"  IoU: {iou_predictions.shape}")
    
    return True


if __name__ == "__main__":
    success = test_onnx_inference()
    exit(0 if success else 1)
