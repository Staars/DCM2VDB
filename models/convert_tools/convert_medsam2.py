#!/usr/bin/env python3
"""
Convert MedSAM2 from PyTorch to MLX and ONNX formats

This script:
1. Downloads MedSAM2 weights from HuggingFace
2. Converts to MLX format (for macOS Apple Silicon)
3. Converts to ONNX format (encoder + decoder for Windows/Linux)

Requirements:
    pip install torch huggingface-hub mlx onnx
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path


def download_medsam2(model_name="MedSAM2_latest.pt", output_dir="../.cache"):
    """Download MedSAM2 weights from HuggingFace"""
    print("=" * 60)
    print("STEP 1: Downloading MedSAM2 from HuggingFace")
    print("=" * 60)
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed")
        print("Install with: pip install huggingface-hub")
        sys.exit(1)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_name} to cache folder...")
    print("This may take a few minutes (~180 MB)...")
    
    try:
        model_path = hf_hub_download(
            repo_id="wanglab/MedSAM2",
            filename=model_name,
            local_dir=str(output_path),
            local_dir_use_symlinks=False
        )
        
        print(f"✓ Downloaded to: {model_path}")
        return model_path
    
    except Exception as e:
        print(f"ERROR downloading model: {e}")
        sys.exit(1)


def load_pytorch_checkpoint(checkpoint_path):
    """Load MedSAM2 PyTorch checkpoint"""
    print("\n" + "=" * 60)
    print("STEP 2: Loading PyTorch Checkpoint")
    print("=" * 60)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"✓ Checkpoint loaded")
        
        # Check if it's a state dict or full model
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("  Found 'model' key in checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("  Found 'state_dict' key in checkpoint")
        else:
            state_dict = checkpoint
            print("  Using checkpoint as state_dict directly")
        
        # Print some info
        total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        print(f"  Total parameters: {total_params:,}")
        print(f"  Number of layers: {len(state_dict)}")
        
        return state_dict
    
    except Exception as e:
        print(f"ERROR loading PyTorch checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def convert_to_mlx(state_dict, output_path="../medsam2_mlx"):
    """Convert PyTorch state dict to MLX format"""
    print("\n" + "=" * 60)
    print("STEP 3: Converting to MLX Format")
    print("=" * 60)
    
    try:
        import mlx.core as mx
        import numpy as np
    except ImportError:
        print("ERROR: MLX not installed")
        print("Install with: pip install mlx")
        return None
    
    print("Converting PyTorch tensors to MLX arrays...")
    
    try:
        mlx_weights = {}
        
        for key, value in state_dict.items():
            if hasattr(value, 'numpy'):
                # Convert PyTorch tensor to numpy, then to MLX
                numpy_array = value.cpu().numpy()
                mlx_array = mx.array(numpy_array)
                mlx_weights[key] = mlx_array
                
                if len(mlx_weights) % 100 == 0:
                    print(f"  Converted {len(mlx_weights)} layers...")
        
        print(f"✓ Converted {len(mlx_weights)} layers to MLX format")
        
        # Save MLX weights
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving MLX weights to: {output_dir}")
        mx.save_safetensors(str(output_dir / "weights.safetensors"), mlx_weights)
        
        print(f"✓ MLX weights saved")
        
        # Calculate size
        size_mb = os.path.getsize(output_dir / "weights.safetensors") / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")
        
        return str(output_dir)
    
    except Exception as e:
        print(f"ERROR converting to MLX: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_to_onnx(checkpoint_path, output_dir="../medsam2_onnx"):
    """Convert MedSAM2 to ONNX format (split encoder/decoder)"""
    print("\n" + "=" * 60)
    print("STEP 4: Converting to ONNX Format")
    print("=" * 60)
    
    # Import SAM2
    try:
        sys.path.insert(0, '../.cache/sam2_repo')
        from sam2.build_sam import build_sam2
    except ImportError:
        print("ERROR: SAM2 not installed")
        print("SAM2 repository must be at ../.cache/sam2_repo")
        print("Run setup_conversion_env.sh to clone it")
        return None
    
    print("Loading SAM2 model architecture...")
    
    # Use CPU for ONNX export (most reliable)
    device = "cpu"
    
    try:
        # Load model
        config = "sam2_hiera_t.yaml"  # MedSAM2 uses Tiny
        model = build_sam2(config, ckpt_path=None, device=device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        
        print("✓ Model loaded")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # --- PART A: IMAGE ENCODER ---
    print("\n--- Exporting Image Encoder ---")
    print("Converts 1024x1024 RGB image → embeddings")
    
    try:
        # Wrap encoder to get correct outputs
        class EncoderWrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
            
            def forward(self, x):
                # 1. Den Standard-Encoder-Lauf machen
                features = self.encoder(x)
                
                # 2. image_embeddings (die 256er für den Decoder-Hals)
                image_embeddings = features["vision_features"]
                
                # 3. Die korrekten High-Res Features extrahieren
                # In SAM2/MedSAM2 liegen die originalen (unprojizierten) Features 
                # oft in 'backbone_fpn' ODER wir müssen sie direkt aus den 
                # 'vision_stack' Ausgängen nehmen.
                
                # Versuchen wir den sichersten Weg für das Tiny-Modell:
                # Wir nehmen die ersten zwei Skalen der FPN-Features.
                # Falls diese 256 sind, müssen wir sie auf 32/64 reduzieren:
                
                fpn0 = features["backbone_fpn"][0]
                fpn1 = features["backbone_fpn"][1]
                
                # Falls der MedSAM2-Backbone sie bereits auf 256 aufgeblasen hat, 
                # schneiden wir sie einfach hart ab, da der Decoder-Checkpoint 
                # (den wir nicht ändern können) nur die ersten 32 bzw. 64 Gewichte nutzt:
                if fpn0.shape[1] == 256:
                    fpn0 = fpn0[:, :32, :, :]
                if fpn1.shape[1] == 256:
                    fpn1 = fpn1[:, :64, :, :]
                    
                return image_embeddings, fpn0, fpn1

        encoder_wrapper = EncoderWrapper(model.image_encoder)
        dummy_img = torch.randn(1, 3, 1024, 1024)
        
        encoder_path = output_path / "encoder.onnx"
        
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
                dynamo=False
            )
        
        encoder_size = os.path.getsize(encoder_path) / (1024 * 1024)
        print(f"✓ Encoder exported: {encoder_path} ({encoder_size:.1f} MB)")
    
    except Exception as e:
        print(f"ERROR exporting encoder: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # --- PART B: MASK DECODER ---
    print("\n--- Exporting Mask Decoder ---")
    print("Takes embeddings + prompts → mask")
    
    # --- PART B: MASK DECODER (Fortsetzung) ---
    try:
        class DecoderWrapper(nn.Module):
            def __init__(self, sam2_model):
                super().__init__()
                self.decoder = sam2_model.sam_mask_decoder
                self.prompt_encoder = sam2_model.sam_prompt_encoder

            def forward(self, image_embeddings, backbone_fpn_0, backbone_fpn_1, point_coords, point_labels):
                # Erstellt die Liste der High-Res Features (Skip-Connections)
                # Wichtig: MedSAM2 Tiny erwartet hier exakt 32 und 64 Kanäle
                high_res_features = [backbone_fpn_0, backbone_fpn_1]
                
                # Berechnet die Prompt-Embeddings aus Klicks/Boxen
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                )
                
                # Berechnet die eigentliche Maske
                low_res_masks, iou_predictions, _, _ = self.decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features
                )
                
                return low_res_masks, iou_predictions

        decoder_model = DecoderWrapper(model)
        
        # Dummy Inputs mit den korrekten Tiny-Dimensionen (32/64)
        dummy_embeds = torch.randn(1, 256, 64, 64)
        dummy_fpn0 = torch.randn(1, 32, 256, 256)   # Matches Tiny Scale 0
        dummy_fpn1 = torch.randn(1, 64, 128, 128)   # Matches Tiny Scale 1
        dummy_coords = torch.randn(1, 2, 2)         # Batch, Points, XY
        dummy_labels = torch.ones(1, 2)             # Batch, Labels (1=Pos, 0=Neg)

        decoder_path = output_path / "decoder.onnx"
        
        print("Exporting Decoder to ONNX...")
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
                dynamo=False
            )
        
        decoder_size = os.path.getsize(decoder_path) / (1024 * 1024)
        print(f"✓ Decoder exported: {decoder_path} ({decoder_size:.1f} MB)")
        return str(output_path)

    except Exception as e:
        print(f"ERROR exporting decoder: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Vollständige Pipeline ausführen"""
    print("MedSAM2 All-in-One Converter (MLX & ONNX)")
    print("=" * 60)
    
    # 1. Download
    ckpt_path = download_medsam2()
    
    # 2. State Dict laden (für MLX)
    state_dict = load_pytorch_checkpoint(ckpt_path)
    
    # 3. MLX Konvertierung
    convert_to_mlx(state_dict)
    
    # 4. ONNX Konvertierung (erfordert sam2_repo)
    convert_to_onnx(ckpt_path)
    
    print("\n" + "=" * 60)
    print("ALL CONVERSIONS FINISHED")
    print("=" * 60)
    print("Files created:")
    print("  ../medsam2_mlx/weights.safetensors  -> Use for macOS Apple Silicon")
    print("  ../medsam2_onnx/encoder.onnx        -> Use for Windows/NVIDIA (Heavy)")
    print("  ../medsam2_onnx/decoder.onnx        -> Use for Windows/NVIDIA (Light)")


if __name__ == "__main__":
    main()

