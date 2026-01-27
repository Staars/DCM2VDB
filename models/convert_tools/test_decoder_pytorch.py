#!/usr/bin/env python3
"""Test decoder with PyTorch to understand expected inputs"""

import torch
import sys

sys.path.insert(0, '../.cache/sam2_repo')
from sam2.build_sam import build_sam2

# Load model
model = build_sam2("sam2_hiera_t.yaml", ckpt_path=None, device="cpu")
checkpoint = torch.load("models/MedSAM2_latest.pt", map_location='cpu')
model.load_state_dict(checkpoint, strict=False)
model.eval()

# Run encoder
dummy_img = torch.randn(1, 3, 1024, 1024)
with torch.no_grad():
    features = model.image_encoder(dummy_img)

print("Encoder outputs:")
print(f"  vision_features: {features['vision_features'].shape}")
print(f"  backbone_fpn[0]: {features['backbone_fpn'][0].shape}")
print(f"  backbone_fpn[1]: {features['backbone_fpn'][1].shape}")

# Try decoder
print("\nTesting decoder...")
dummy_coords = torch.randn(1, 2, 2)
dummy_labels = torch.ones(1, 2)

# Encode prompts
sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
    points=(dummy_coords, dummy_labels),
    boxes=None,
    masks=None,
)

print(f"  sparse_embeddings: {sparse_embeddings.shape}")
print(f"  dense_embeddings: {dense_embeddings.shape}")

# Run decoder
try:
    with torch.no_grad():
        low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
            image_embeddings=features['vision_features'],
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=features['backbone_fpn'][:2]  # First 2 FPN features
        )
    
    print(f"\n✓ Decoder works!")
    print(f"  masks: {low_res_masks.shape}")
    print(f"  iou: {iou_predictions.shape}")
    
except Exception as e:
    print(f"\n✗ Decoder failed: {e}")
    import traceback
    traceback.print_exc()
