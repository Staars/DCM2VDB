#!/usr/bin/env python3
"""Check MedSAM2 checkpoint layer sizes"""

import torch

checkpoint = torch.load("models/MedSAM2_latest.pt", map_location='cpu')

print("Looking for conv_s0 and conv_s1 in checkpoint:")
for key in sorted(checkpoint.keys()):
    if 'conv_s' in key:
        value = checkpoint[key]
        print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")

print("\nAll decoder-related keys:")
for key in sorted(checkpoint.keys()):
    if 'mask_decoder' in key or 'sam_mask_decoder' in key:
        value = checkpoint[key]
        print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
