#!/usr/bin/env python3
"""
Convert MedSAM2 to ONNX format (split encoder/decoder)

This script exports MedSAM2 as two separate ONNX models:
1. Image Encoder: image → embeddings
2. Mask Decoder: embeddings + prompts → mask
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path


def export_medsam2_split(checkpoint_path, output_dir="../medsam2_onnx"):
    """Export MedSAM2 as split ONNX components"""
    print("=" * 60)
    print("MedSAM2 Split ONNX Export")
    print("=" * 60)
    
    # Import SAM2
    try:
        from sam2.build_sam import build_sam2
    except ImportError:
        print("ERROR: SAM2 not installed")
        print("Run: python convert_to_onnx.py first to install SAM2")
        sys.exit(1)
    
    print("\n--- Loading Model ---")
    
    # Use CPU for ONNX export (most reliable)
    device = "cpu"
    print(f"Using device: {device}")
    
    # Load model
    config = "sam2_hiera_t.yaml"  # MedSAM2 uses Tiny
    model = build_sam2(config, ckpt_path=None, device=device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    print("✓ Model loaded")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # --- PART A: IMAGE ENCODER ---
    print("\n" + "=" * 60)
    print("PART A: Exporting Image Encoder")
    print("=" * 60)
    print("Converts 1024x1024 RGB image → embeddings")
    
    try:
        # Wrap encoder to get correct outputs
        class EncoderWrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
            
            def forward(self, x):
                # Run encoder and get all outputs
                features = self.encoder(x)
                
                # features is a dict with 'vision_features' and 'backbone_fpn'
                image_embeddings = features['vision_features']
                backbone_fpn = features.get('backbone_fpn', [])
                
                # Return embeddings and FPN features
                if len(backbone_fpn) >= 2:
                    return image_embeddings, backbone_fpn[0], backbone_fpn[1]
                else:
                    # Fallback if no FPN features
                    return image_embeddings, image_embeddings, image_embeddings
        
        encoder_wrapper = EncoderWrapper(model.image_encoder)
        dummy_img = torch.randn(1, 3, 1024, 1024)
        
        encoder_path = output_path / "medsam2_encoder.onnx"
        
        with torch.no_grad():
            torch.onnx.export(
                encoder_wrapper,
                dummy_img,
                str(encoder_path),
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['image'],
                output_names=['image_embeddings', 'backbone_fpn_0', 'backbone_fpn_1'],
                dynamic_axes={'image': {0: 'batch_size'}},
                dynamo=False  # Use classic exporter, not Dynamo
            )
        
        encoder_size = os.path.getsize(encoder_path) / (1024 * 1024)
        print(f"✓ Encoder exported: {encoder_path}")
        print(f"  Size: {encoder_size:.1f} MB")
    
    except Exception as e:
        print(f"ERROR exporting encoder: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # --- PART B: MASK DECODER ---
    print("\n" + "=" * 60)
    print("PART B: Exporting Mask Decoder")
    print("=" * 60)
    print("Takes embeddings + prompts (points/boxes) → mask")
    
    try:
        # Create wrapper for decoder
        class Sam2DecoderWrapper(nn.Module):
            def __init__(self, sam2_model):
                super().__init__()
                self.decoder = sam2_model.sam_mask_decoder
                self.prompt_encoder = sam2_model.sam_prompt_encoder

            def forward(self, image_embeddings, backbone_fpn_0, backbone_fpn_1, point_coords, point_labels):
                # Collect high-res features in the list format expected by decoder
                high_res_features = [backbone_fpn_0, backbone_fpn_1]
                
                # Compute prompt embeddings (points/boxes)
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                )
                
                # Compute mask - pass high_res_features to fix the error
                low_res_masks, iou_predictions, _, _ = self.decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features  # FIX HERE
                )
                
                return low_res_masks, iou_predictions

        decoder_model = Sam2DecoderWrapper(model)
        
        # Dummy inputs must match encoder outputs (all 256 channels)
        dummy_embeds = torch.randn(1, 256, 64, 64)       # Main embeddings
        dummy_fpn0 = torch.randn(1, 256, 256, 256)       # High-res Scale 0 (256 channels!)
        dummy_fpn1 = torch.randn(1, 256, 128, 128)       # High-res Scale 1 (256 channels!)
        dummy_coords = torch.randn(1, 2, 2)              # [batch, points, xy]
        dummy_labels = torch.ones(1, 2)                  # [batch, labels]

        decoder_path = output_path / "medsam2_decoder.onnx"
        
        with torch.no_grad():
            torch.onnx.export(
                decoder_model,
                (dummy_embeds, dummy_fpn0, dummy_fpn1, dummy_coords, dummy_labels),
                str(decoder_path),
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['image_embeddings', 'backbone_fpn_0', 'backbone_fpn_1', 'point_coords', 'point_labels'],
                output_names=['masks', 'iou_predictions'],
                dynamic_axes={
                    'point_coords': {1: 'num_points'},
                    'point_labels': {1: 'num_points'}
                },
                dynamo=False  # Use classic exporter, not Dynamo
            )
        
        decoder_size = os.path.getsize(decoder_path) / (1024 * 1024)
        print(f"✓ Decoder exported: {decoder_path}")
        print(f"  Size: {decoder_size:.1f} MB")
        
        total_size = encoder_size + decoder_size
        print(f"\n✓ Total ONNX size: {total_size:.1f} MB")
        
        return str(encoder_path), str(decoder_path)
    
    except Exception as e:
        print(f"ERROR exporting decoder: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_onnx(onnx_path):
    """Verify ONNX model"""
    try:
        import onnx
    except ImportError:
        print("WARNING: onnx not installed, skipping verification")
        return False
    
    print(f"\nVerifying: {onnx_path}")
    
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        print("✓ ONNX model is valid")
        print(f"  Inputs: {[i.name for i in model.graph.input]}")
        print(f"  Outputs: {[o.name for o in model.graph.output]}")
        
        return True
    
    except Exception as e:
        print(f"ERROR verifying ONNX: {e}")
        return False


def main():
    """Main export pipeline"""
    checkpoint_path = "../MedSAM2_latest.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Run convert_medsam2.py first to download the model")
        sys.exit(1)
    
    # Export split ONNX models
    result = export_medsam2_split(checkpoint_path)
    
    if result:
        encoder_path, decoder_path = result
        
        # Verify both models
        print("\n" + "=" * 60)
        print("Verifying ONNX Models")
        print("=" * 60)
        verify_onnx(encoder_path)
        verify_onnx(decoder_path)
        
        # Summary
        print("\n" + "=" * 60)
        print("EXPORT COMPLETE!")
        print("=" * 60)
        print(f"✓ Encoder: {encoder_path}")
        print(f"✓ Decoder: {decoder_path}")
        print("\nUsage:")
        print("1. Run encoder on image → get embeddings")
        print("2. Run decoder with embeddings + prompts → get mask")
        print("3. Can cache embeddings and run decoder multiple times!")
        print("\nNext steps:")
        print("1. Test ONNX inference with onnxruntime")
        print("2. Compare with MLX performance")
        print("3. Integrate into Blender extension")
    else:
        print("\n" + "=" * 60)
        print("EXPORT FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
