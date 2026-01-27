#!/usr/bin/env python3
"""Inspect SAM2 encoder output structure"""

import torch
import sys
from pathlib import Path

# Add sam2_repo to path
sys.path.insert(0, '../.cache/sam2_repo')

from sam2.build_sam import build_sam2

# Load model
config = "sam2_hiera_t.yaml"
model = build_sam2(config, ckpt_path=None, device="cpu")
checkpoint = torch.load("models/MedSAM2_latest.pt", map_location='cpu')
model.load_state_dict(checkpoint, strict=False)
model.eval()

# Test encoder
dummy_img = torch.randn(1, 3, 1024, 1024)

with torch.no_grad():
    features = model.image_encoder(dummy_img)

print("Encoder output type:", type(features))
print("\nEncoder output structure:")

if isinstance(features, dict):
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list with {len(value)} items")
            for i, item in enumerate(value):
                if isinstance(item, torch.Tensor):
                    print(f"    [{i}]: {item.shape}")
        else:
            print(f"  {key}: {type(value)}")
elif isinstance(features, torch.Tensor):
    print(f"  Single tensor: {features.shape}")
else:
    print(f"  Unknown type: {type(features)}")
