#!/usr/bin/env python3
"""
Convert MedSAM2 from PyTorch to MLX and ONNX formats — CI/GH Actions version.

Outputs directly into extension/ml/medsam2_mlx/ and extension/ml/medsam2_onnx/.
No CoreML conversion (not used in the extension).

Usage (from repo root):
    python convert_medsam2_ci.py
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path


# ── Repo root (this script lives at repo root) ─────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / ".cache"
SAM2_REPO = CACHE_DIR / "sam2_repo"
MLX_OUT   = REPO_ROOT / "extension" / "ml" / "medsam2_mlx"
ONNX_OUT  = REPO_ROOT / "extension" / "ml" / "medsam2_onnx"


# ── Weight key sets for transposition ───────────────────────────────────────
CONVT_KEYS = {
    "sam_mask_decoder.output_upscaling.0.weight",
    "sam_mask_decoder.output_upscaling.3.weight",
}

POS_EMBED_NCHW_KEYS = {
    "image_encoder.trunk.pos_embed",
    "image_encoder.trunk.pos_embed_window",
}


def download_medsam2(model_name="MedSAM2_latest.pt"):
    print("=" * 60)
    print("STEP 1: Downloading MedSAM2 from HuggingFace")
    print("=" * 60)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed"); sys.exit(1)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_file = CACHE_DIR / model_name
    if target_file.exists():
        print(f"✓ Model already exists: {target_file}")
        return str(target_file)
    print(f"Downloading {model_name} (~180 MB)...")
    try:
        downloaded_path = hf_hub_download(repo_id="wanglab/MedSAM2", filename=model_name)
        import shutil
        shutil.copy2(downloaded_path, target_file)
        print(f"✓ Downloaded to: {target_file}")
        return str(target_file)
    except Exception as e:
        print(f"ERROR downloading model: {e}"); sys.exit(1)


def load_pytorch_checkpoint(checkpoint_path):
    print("\n" + "=" * 60)
    print("STEP 2: Loading PyTorch Checkpoint")
    print("=" * 60)
    print(f"Loading: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        total_params = sum(p.numel() for p in state_dict.values()
                           if isinstance(p, torch.Tensor))
        print(f"✓ Loaded {len(state_dict)} keys, {total_params:,} params")
        return state_dict
    except Exception as e:
        print(f"ERROR: {e}"); import traceback; traceback.print_exc(); sys.exit(1)


def _compute_hiera_block_configs(state_dict, block_prefix):
    stages            = [1, 2, 7, 2]
    q_pool            = 3
    q_stride          = (2, 2)
    window_spec       = (8, 4, 14, 7)
    global_att_blocks = (5, 7, 9)
    dim_mul           = 2.0
    head_mul          = 2.0

    depth       = sum(stages)
    stage_ends  = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
    q_pool_blks = set([x + 1 for x in stage_ends[:-1]][:q_pool])

    actual_depth = 0
    while (block_prefix + f"{actual_depth}.norm1.weight") in state_dict:
        actual_depth += 1
    if actual_depth != depth:
        print(f"  ⚠️  yaml depth={depth} but weights have {actual_depth} blocks — "
              f"using weights to drive config")
        depth = actual_depth

    blocks_cfg = []
    cur_stage  = 1
    cur_heads  = 1

    for i in range(depth):
        prefix    = block_prefix + f"{i}."
        norm1_key = prefix + "norm1.weight"
        norm2_key = prefix + "norm2.weight"
        qkv_key   = prefix + "attn.qkv.weight"
        proj_key  = prefix + "proj.weight"

        embed_dim     = int(state_dict[norm1_key].shape[0])
        dim_out       = int(state_dict[norm2_key].shape[0])
        qkv_shape     = list(state_dict[qkv_key].shape)

        window_size = window_spec[cur_stage - 1]
        if i in global_att_blocks:
            window_size = 0

        if i - 1 in stage_ends:
            cur_heads = int(cur_heads * head_mul)
            cur_stage += 1

        has_q_stride = i in q_pool_blks
        has_proj     = proj_key in state_dict

        blocks_cfg.append({
            "idx":         i,
            "embed_dim":   embed_dim,
            "dim_out":     dim_out,
            "num_heads":   cur_heads,
            "qkv_shape":   qkv_shape,
            "window_size": window_size,
            "q_stride":    list(q_stride) if has_q_stride else None,
            "global_attn": window_size == 0,
            "has_proj":    has_proj,
        })

    return blocks_cfg


def extract_model_config(state_dict, mlx_weights):
    cfg      = {}
    all_keys = list(state_dict.keys())

    patch_candidates = [k for k in all_keys if "patch_embed" in k and "proj.weight" in k]
    cfg["patch_embed_prefix"] = patch_candidates[0].replace("proj.weight", "") \
        if patch_candidates else "image_encoder.trunk.patch_embed."

    pe_candidates = [k for k in all_keys if "pos_embed" in k
                     and "window" not in k and "trunk" in k]
    cfg["pos_embed_key"] = pe_candidates[0] if pe_candidates \
        else "image_encoder.trunk.pos_embed"

    pe_win_candidates = [k for k in all_keys if "pos_embed_window" in k and "trunk" in k]
    cfg["pos_embed_win_key"] = pe_win_candidates[0] if pe_win_candidates \
        else "image_encoder.trunk.pos_embed_window"

    neck_candidates = [k for k in all_keys if "neck" in k and "convs.0" in k]
    cfg["neck_prefix"] = neck_candidates[0].split("convs.0")[0] if neck_candidates \
        else "image_encoder.neck."

    block_candidates = [k for k in all_keys if "trunk.blocks.0." in k]
    cfg["block_prefix"] = block_candidates[0].split("blocks.0.")[0] + "blocks." \
        if block_candidates else "image_encoder.trunk.blocks."

    blocks_cfg = _compute_hiera_block_configs(state_dict, cfg["block_prefix"])
    cfg["hiera_blocks"] = blocks_cfg
    cfg["num_blocks"]   = len(blocks_cfg)

    neck_in_dims = []
    j = 0
    while True:
        nk = cfg["neck_prefix"] + f"convs.{j}.conv.weight"
        if nk in state_dict:
            neck_in_dims.append(int(state_dict[nk].shape[1]))
        else:
            break
        j += 1
    cfg["fpn_in_dims"] = list(reversed(neck_in_dims)) if neck_in_dims else [96, 192, 384, 768]
    cfg["fpn_out_dim"] = 256
    cfg["fpn_scalp"]   = 1
    cfg["fpn_top_down_levels"] = [2, 3]

    p0 = "sam_mask_decoder.transformer.layers.0."
    cfg["decoder_mlp_style"] = "lin1_lin2" \
        if p0 + "mlp.lin1.weight" in state_dict else "layers_list"

    uc0 = mlx_weights.get("sam_mask_decoder.output_upscaling.0.weight")
    uc3 = mlx_weights.get("sam_mask_decoder.output_upscaling.3.weight")
    if uc0 is not None:
        s = list(uc0.shape)
        cfg["upscale_conv0"] = {"C_out": s[0], "C_in": s[3]}
    if uc3 is not None:
        s = list(uc3.shape)
        cfg["upscale_conv1"] = {"C_out": s[0], "C_in": s[3]}

    n_layers = 0
    while f"sam_mask_decoder.transformer.layers.{n_layers}.self_attn.q_proj.weight" \
            in state_dict:
        n_layers += 1
    cfg["decoder_transformer_layers"] = max(n_layers, 2)

    pe_w = mlx_weights.get(cfg["pos_embed_key"])
    if pe_w is not None:
        cfg["pos_embed_shape"] = list(pe_w.shape)
    pe_win_w = mlx_weights.get(cfg["pos_embed_win_key"])
    if pe_win_w is not None:
        cfg["pos_embed_win_shape"] = list(pe_win_w.shape)

    return cfg


def convert_to_mlx(state_dict, output_path=None):
    import mlx.core as mx
    if output_path is None:
        output_path = str(MLX_OUT)
    stats = dict(conv2d=0, convt=0, pos_embed=0, qkv_raw=0, linear=0, skipped=0)
    mlx_weights = {}

    for key, value in state_dict.items():
        if not hasattr(value, "detach"):
            stats["skipped"] += 1
            continue

        arr = value.detach().cpu().numpy().astype(np.float32)

        if key in CONVT_KEYS and arr.ndim == 4:
            arr = arr.transpose(1, 2, 3, 0)
            stats["convt"] += 1
        elif key in POS_EMBED_NCHW_KEYS and arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)
            stats["pos_embed"] += 1
        elif key.endswith(".weight") and arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)
            stats["conv2d"] += 1
        elif arr.ndim == 2 and ".attn.qkv." in key:
            stats["qkv_raw"] += 1

        mlx_weights[key] = mx.array(arr)

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_file = str(out_dir / "weights.safetensors")
    mx.save_safetensors(weights_file, mlx_weights)

    print(f"\nMLX conversion stats:")
    print(f"  Conv2d transposed  : {stats['conv2d']}")
    print(f"  ConvTranspose2d    : {stats['convt']}")
    print(f"  pos_embed NHWC     : {stats['pos_embed']}")
    print(f"  QKV raw            : {stats['qkv_raw']}")
    print(f"  Skipped            : {stats['skipped']}")
    print(f"  Total output keys  : {len(mlx_weights)}")
    print(f"  Saved: {weights_file}")

    print("\nExtracting model config...")
    cfg = extract_model_config(state_dict, mlx_weights)
    config_file = str(out_dir / "model_config.json")
    with open(config_file, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"✓ model_config.json saved: {config_file}")

    return str(out_dir), cfg


def convert_to_onnx(checkpoint_path, output_dir=None):
    import torch.nn as torch_nn
    if output_dir is None:
        output_dir = str(ONNX_OUT)
    print("\n" + "=" * 60)
    print("STEP 4: Converting to ONNX")
    print("=" * 60)
    try:
        sys.path.insert(0, str(SAM2_REPO))
        from sam2.build_sam import build_sam2
    except ImportError:
        print(f"ERROR: SAM2 not installed at {SAM2_REPO}"); return None

    device = "cpu"
    try:
        model      = build_sam2("sam2_hiera_t.yaml", ckpt_path=None, device=device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("✓ Model loaded")
    except Exception as e:
        print(f"ERROR loading model: {e}"); import traceback; traceback.print_exc(); return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n--- Exporting Image Encoder ---")
    try:
        class EncoderWrapper(torch_nn.Module):
            def __init__(self, encoder, sam2_model):
                super().__init__()
                self.encoder = encoder
                self.proj_0  = sam2_model.sam_mask_decoder.conv_s0
                self.proj_1  = sam2_model.sam_mask_decoder.conv_s1
            def forward(self, x):
                features         = self.encoder(x)
                image_embeddings = features["vision_features"]
                fpn              = features["backbone_fpn"]
                return image_embeddings, self.proj_0(fpn[0]), self.proj_1(fpn[1])

        enc        = EncoderWrapper(model.image_encoder, model).eval()
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1) / 255.0
        pixel_std  = torch.tensor([ 58.395,  57.12, 57.375]).view(1, 3, 1, 1) / 255.0
        dummy_img  = (torch.ones(1, 3, 1024, 1024) * (127.5 / 255.0) - pixel_mean) / pixel_std
        encoder_path = output_path / "encoder.onnx"
        with torch.no_grad():
            test_out = enc(dummy_img)
            print(f"  Test OK: {[list(o.shape) for o in test_out]}")
        torch.onnx.export(enc, dummy_img, str(encoder_path),
                          export_params=True, opset_version=18,
                          do_constant_folding=True,
                          input_names=["image"],
                          output_names=["image_embeddings", "backbone_fpn_0", "backbone_fpn_1"],
                          dynamic_axes={"image": {0: "batch_size"}},
                          dynamo=False)
        print(f"  ✓ Encoder: {encoder_path} ({os.path.getsize(encoder_path)/1024/1024:.1f} MB)")
    except Exception as e:
        print(f"ERROR encoder: {e}"); import traceback; traceback.print_exc(); return None

    print("\n--- Exporting Mask Decoder ---")
    try:
        class DecoderWrapper(torch_nn.Module):
            def __init__(self, sam2_model):
                super().__init__()
                self.decoder        = sam2_model.sam_mask_decoder
                self.prompt_encoder = sam2_model.sam_prompt_encoder
            def forward(self, image_embeddings, backbone_fpn_0, backbone_fpn_1,
                        point_coords, point_labels):
                sparse_emb, dense_emb = self.prompt_encoder(
                    points=(point_coords, point_labels), boxes=None, masks=None)
                low_res_masks, iou_predictions, _, _ = self.decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False, repeat_image=False,
                    high_res_features=[backbone_fpn_0, backbone_fpn_1])
                return low_res_masks, iou_predictions

        dec    = DecoderWrapper(model).eval()
        d_emb  = torch.randn(1, 256, 64, 64)  * 0.1
        d_fpn0 = torch.randn(1,  32, 256, 256) * 0.1
        d_fpn1 = torch.randn(1,  64, 128, 128) * 0.1
        d_pts  = torch.randint(0, 1024, (1, 2, 2)).float()
        d_lbl  = torch.ones(1, 2).float()
        decoder_path = output_path / "decoder.onnx"
        with torch.no_grad():
            masks, iou = dec(d_emb, d_fpn0, d_fpn1, d_pts, d_lbl)
            print(f"  Test OK: masks={list(masks.shape)} iou={list(iou.shape)}")
        torch.onnx.export(dec, (d_emb, d_fpn0, d_fpn1, d_pts, d_lbl),
                          str(decoder_path),
                          export_params=True, opset_version=17,
                          do_constant_folding=True,
                          input_names=["image_embeddings", "backbone_fpn_0", "backbone_fpn_1",
                                       "point_coords", "point_labels"],
                          output_names=["masks", "iou_predictions"],
                          dynamic_axes={"point_coords": {1: "num_points"},
                                        "point_labels": {1: "num_points"}},
                          dynamo=False)
        print(f"  ✓ Decoder: {decoder_path} ({os.path.getsize(decoder_path)/1024/1024:.1f} MB)")
        return str(output_path)
    except Exception as e:
        print(f"ERROR decoder: {e}"); import traceback; traceback.print_exc(); return None


def main():
    print("MedSAM2 CI Converter (MLX & ONNX)")
    print("=" * 60)
    print(f"  Cache dir : {CACHE_DIR}")
    print(f"  SAM2 repo : {SAM2_REPO}")
    print(f"  MLX output: {MLX_OUT}")
    print(f"  ONNX output: {ONNX_OUT}")
    print("=" * 60)

    # Clone SAM2 repo if needed (for ONNX conversion)
    if not SAM2_REPO.exists():
        print(f"\nCloning SAM2 repository to {SAM2_REPO}...")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        import subprocess
        subprocess.run(["git", "clone", "https://github.com/facebookresearch/segment-anything-2.git",
                        str(SAM2_REPO)], check=True)
        print("✓ SAM2 repository cloned")

    ckpt_path  = download_medsam2()
    state_dict = load_pytorch_checkpoint(ckpt_path)

    print("\n" + "=" * 60)
    print("STEP 3: Converting to MLX + Config")
    print("=" * 60)
    mlx_dir, cfg = convert_to_mlx(state_dict)
    if mlx_dir is None:
        print("ERROR: MLX conversion failed"); sys.exit(1)

    onnx_dir = convert_to_onnx(ckpt_path)
    if onnx_dir is None:
        print("ERROR: ONNX conversion failed"); sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)
    print(f"  {MLX_OUT}/weights.safetensors")
    print(f"  {MLX_OUT}/model_config.json")
    print(f"  {ONNX_OUT}/encoder.onnx")
    print(f"  {ONNX_OUT}/decoder.onnx")


if __name__ == "__main__":
    main()
