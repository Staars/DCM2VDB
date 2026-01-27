#!/usr/bin/env python3
"""Inspect SAM2 decoder structure"""

import torch
import sys

sys.path.insert(0, '../.cache/sam2_repo')
from sam2.build_sam import build_sam2

# Load model
model = build_sam2("sam2_hiera_t.yaml", ckpt_path=None, device="cpu")
checkpoint = torch.load("models/MedSAM2_latest.pt", map_location='cpu')
model.load_state_dict(checkpoint, strict=False)
model.eval()

print("Mask Decoder structure:")
print(model.sam_mask_decoder)

print("\n\nLooking for FPN projection layers...")
for name, module in model.sam_mask_decoder.named_modules():
    if 'conv' in name.lower() or 'proj' in name.lower() or 'down' in name.lower():
        print(f"{name}: {module}")
