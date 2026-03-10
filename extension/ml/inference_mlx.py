#!/usr/bin/env python3
"""
MLX-based inference for MedSAM2 on macOS Apple Silicon

Complete SAM2 architecture implementation in MLX for efficient
inference on Apple Silicon Metal GPU.

- Exact SAM2 token order: [obj_score_token, iou_token, mask_tokens..., sparse]
- PIL-free: all resize ops use nn.Upsample with correct MLX mode names
  ('nearest', 'linear'=bilinear, 'cubic'=bicubic)
"""

import json, math, warnings
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Resize helper (PIL-free, correct MLX mode names)
# ---------------------------------------------------------------------------

def _upsample(x, h_out, w_out, mode="linear"):
    """x: (N, H, W, C) → (N, h_out, w_out, C).
    mode: 'nearest' | 'linear' (bilinear for 2-D) | 'cubic' (bicubic for 2-D)
    """
    h_in, w_in = x.shape[1], x.shape[2]
    return nn.Upsample(scale_factor=(h_out / h_in, w_out / w_in), mode=mode)(x)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _w(weights, key):
    if key not in weights:
        similar = [k for k in weights if key.split(".")[-2] in k][:4]
        raise KeyError(f"Weight not found: '{key}'\nSimilar: {similar}")
    return weights[key]

def _wopt(weights, key):
    return weights.get(key, None)

def _chk(x, tag, expected=None):
    mx.eval(x)
    s = list(x.shape)
    if expected is not None and s != list(expected):
        raise AssertionError(f"[{tag}] shape {s} != expected {list(expected)}")
    print(f"  [chk] {tag}: {s}")
    return x


# ---------------------------------------------------------------------------
# Window / pooling helpers
# ---------------------------------------------------------------------------

def window_partition(x, window_size):
    N, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = mx.pad(x, [(0,0),(0,pad_h),(0,pad_w),(0,0)])
    Hp, Wp = H+pad_h, W+pad_w
    x = x.reshape(N, Hp//window_size, window_size, Wp//window_size, window_size, C)
    x = x.transpose(0,1,3,2,4,5).reshape(-1, window_size, window_size, C)
    return x, (Hp, Wp)

def window_unpartition(x, window_size, pad_hw, hw):
    Hp, Wp = pad_hw; H, W = hw
    N = x.shape[0] // ((Hp//window_size)*(Wp//window_size))
    x = x.reshape(N, Hp//window_size, Wp//window_size, window_size, window_size, -1)
    x = x.transpose(0,1,3,2,4,5).reshape(N, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x

def maxpool2d_nhwc(x, stride=2):
    N, H, W, C = x.shape
    return x.reshape(N, H//stride, stride, W//stride, stride, C).max(axis=(2,4))


# ---------------------------------------------------------------------------
# LayerNorm2d
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias   = mx.zeros((num_channels,))
        self.eps    = eps

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var  = ((x - mean)**2).mean(axis=-1, keepdims=True)
        return self.weight * (x - mean) / mx.sqrt(var + self.eps) + self.bias

    def load(self, weights, prefix):
        w = _wopt(weights, prefix+"weight")
        b = _wopt(weights, prefix+"bias")
        if w is not None: self.weight = w
        if b is not None: self.bias   = b


# ---------------------------------------------------------------------------
# MultiScaleAttention / MultiScaleBlock
# ---------------------------------------------------------------------------

class MultiScaleAttention(nn.Module):
    def __init__(self, dim, dim_out, num_heads, q_stride=None):
        super().__init__()
        self.dim_out   = dim_out
        self.num_heads = num_heads
        self.head_dim  = dim_out // num_heads
        self.scale     = self.head_dim ** -0.5
        self.q_stride  = q_stride
        self.qkv  = nn.Linear(dim,     dim_out*3, bias=True)
        self.proj = nn.Linear(dim_out, dim_out,   bias=True)

    def __call__(self, x):
        N, H, W, C = x.shape
        qkv = self.qkv(x.reshape(N, H*W, C)).reshape(N, H*W, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        if self.q_stride is not None:
            q_reshaped = q.reshape(N, H, W, self.num_heads*self.head_dim)
            q_reshaped = maxpool2d_nhwc(q_reshaped, stride=self.q_stride[0])  # NHWC, no transpose needed
            H_q = H // self.q_stride[0]; W_q = W // self.q_stride[0]
            q   = q_reshaped.reshape(N, H_q*W_q, self.num_heads, self.head_dim)
        else:
            H_q, W_q = H, W
        q = q.transpose(0,2,1,3); k = k.transpose(0,2,1,3); v = v.transpose(0,2,1,3)
        attn = mx.softmax((q @ k.transpose(0,1,3,2)) * self.scale, axis=-1)
        out  = (attn @ v).transpose(0,2,1,3).reshape(N, H_q*W_q, self.dim_out)
        return self.proj(out).reshape(N, H_q, W_q, self.dim_out)

    def load(self, weights, prefix):
        self.qkv.weight  = _w(weights, prefix+"attn.qkv.weight")
        self.qkv.bias    = _w(weights, prefix+"attn.qkv.bias")
        self.proj.weight = _w(weights, prefix+"attn.proj.weight")
        self.proj.bias   = _w(weights, prefix+"attn.proj.bias")


class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dim_out, num_heads, window_size, q_stride=None):
        super().__init__()
        self.window_size = window_size
        self.q_stride    = q_stride
        self.norm1  = nn.LayerNorm(dim)
        self.attn   = MultiScaleAttention(dim, dim_out, num_heads, q_stride=q_stride)
        self.norm2  = nn.LayerNorm(dim_out)
        self.mlp_l0 = nn.Linear(dim_out, dim_out*4)
        self.mlp_l1 = nn.Linear(dim_out*4, dim_out)
        self.proj   = nn.Linear(dim, dim_out, bias=True) if dim != dim_out else None

    def __call__(self, x):
        N, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.proj is not None:
            shortcut = self.proj(x)
            if self.q_stride is not None:
                shortcut = maxpool2d_nhwc(shortcut, stride=self.q_stride[0])
        elif self.q_stride is not None:
            shortcut = maxpool2d_nhwc(shortcut, stride=self.q_stride[0])
        
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)
        
        x = self.attn(x)
        
        if self.window_size > 0:
            if self.q_stride is not None:
                # After attention with q_stride, window spatial dims are reduced
                ws_out = self.window_size // self.q_stride[0]
                H_out, W_out = shortcut.shape[1:3]  # pooled dims
                C_out = x.shape[-1]  # channel dim from attention output
                
                # Original number of windows (before attention)
                nH_orig = (H + (self.window_size - H % self.window_size) % self.window_size) // self.window_size
                nW_orig = (W + (self.window_size - W % self.window_size) % self.window_size) // self.window_size
                
                # Reshape: (N*nH_orig*nW_orig, ws_out, ws_out, C_out) -> (N, nH_orig, nW_orig, ws_out, ws_out, C_out)
                x = x.reshape(N, nH_orig, nW_orig, ws_out, ws_out, C_out)
                # Rearrange to (N, Hp, Wp, C_out) where Hp = nH_orig * ws_out, Wp = nW_orig * ws_out
                Hp = nH_orig * ws_out
                Wp = nW_orig * ws_out
                x = x.transpose(0, 1, 3, 2, 4, 5).reshape(N, Hp, Wp, C_out)
                
                # Crop to original pooled dimensions
                if Hp > H_out or Wp > W_out:
                    x = x[:, :H_out, :W_out, :]
            else:
                x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        
        x = shortcut + x
        x = x + self.mlp_l1(nn.gelu(self.mlp_l0(self.norm2(x))))
        return x

    def load(self, weights, prefix):
        self.norm1.weight  = _w(weights, prefix+"norm1.weight")
        self.norm1.bias    = _w(weights, prefix+"norm1.bias")
        self.norm2.weight  = _w(weights, prefix+"norm2.weight")
        self.norm2.bias    = _w(weights, prefix+"norm2.bias")
        self.attn.load(weights, prefix)
        # Handle multiple naming conventions for MLP weights
        mlp_prefix = prefix + "mlp."
        # Try proj_in/proj_out first (HF format)
        w0 = _wopt(weights, mlp_prefix+"proj_in.weight")
        b0 = _wopt(weights, mlp_prefix+"proj_in.bias")
        w1 = _wopt(weights, mlp_prefix+"proj_out.weight")
        b1 = _wopt(weights, mlp_prefix+"proj_out.bias")
        
        # If not found, try layers.0/layers.1 (MedSAM2)
        if w0 is None or w1 is None:
            w0 = _wopt(weights, mlp_prefix+"layers.0.weight")
            b0 = _wopt(weights, mlp_prefix+"layers.0.bias")
            w1 = _wopt(weights, mlp_prefix+"layers.1.weight")
            b1 = _wopt(weights, mlp_prefix+"layers.1.bias")
        
        # Raise error if still not found
        if w0 is None:
            raise KeyError(f"MLP weight not found: tried {mlp_prefix}proj_in.weight, {mlp_prefix}layers.0.weight")
        if w1 is None:
            raise KeyError(f"MLP weight not found: tried {mlp_prefix}proj_out.weight, {mlp_prefix}layers.1.weight")
        
        self.mlp_l0.weight = w0
        if b0 is not None: self.mlp_l0.bias = b0
        self.mlp_l1.weight = w1
        if b1 is not None: self.mlp_l1.bias = b1
        if self.proj is not None:
            self.proj.weight = _w(weights, prefix+"proj.weight")
            self.proj.bias   = _w(weights, prefix+"proj.bias")


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, kernel_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def __call__(self, x):
        return self.proj(x)

    def load(self, weights, prefix):
        self.proj.weight = _w(weights, prefix+"proj.weight")
        b = _wopt(weights, prefix+"proj.bias")
        if b is not None: self.proj.bias = b


# ---------------------------------------------------------------------------
# FPNNeck  — bilinear upsample via _upsample(..., mode='linear')
# ---------------------------------------------------------------------------

class FPNNeck(nn.Module):
    def __init__(self, backbone_channel_list, d_model=256, scalp=1, fpn_top_down_levels=None):
        super().__init__()
        self.scalp = scalp
        n = len(backbone_channel_list)
        self.convs = [nn.Conv2d(backbone_channel_list[n-1-i], d_model, kernel_size=1)
                      for i in range(n)]
        # Default to all levels if not specified (PyTorch default)
        self.fpn_top_down_levels = fpn_top_down_levels or list(range(len(backbone_channel_list)))
        
    def __call__(self, xs):
        n = len(self.convs)
        out = [None] * n
        prev = None
        
        # Iterate from deepest to shallowest (reverse order)
        for i in range(n-1, -1, -1):
            # PyTorch uses self.convs[n - i] where n = len(convs) - 1
            # Our convs are stored in reverse order: convs[0] is for the deepest level
            conv_idx = n - 1 - i  # convs are stored in reverse: convs[0] is for deepest level
            lateral = self.convs[conv_idx](xs[i])
            
            if i in self.fpn_top_down_levels and prev is not None:
                # Use 2x nearest neighbor upsampling as in PyTorch
                # First try 2x upsampling
                h, w = lateral.shape[1:3]
                prev_upsampled = nn.Upsample(scale_factor=2.0, mode='nearest')(prev)
                
                # If sizes don't match after 2x upsampling, use exact size matching
                if prev_upsampled.shape[1:3] != (h, w):
                    # Fallback to exact size matching
                    prev_upsampled = _upsample(prev, h, w, mode="nearest")
                
                prev = lateral + prev_upsampled
            else:
                prev = lateral
            
            out[i] = prev
        
        if self.scalp > 0:
            out = out[:-self.scalp]
        return out

    def load(self, weights, prefix):
        for i, conv in enumerate(self.convs):
            conv.weight = _w(weights, f"{prefix}convs.{i}.conv.weight")
            b = _wopt(weights, f"{prefix}convs.{i}.conv.bias")
            if b is not None:
                conv.bias = b


# ---------------------------------------------------------------------------
# HieraEncoder  — pos embed upsample via _upsample(..., mode='cubic')
# ---------------------------------------------------------------------------

class HieraEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._patch_prefix = cfg["patch_embed_prefix"]
        self._block_prefix = cfg["block_prefix"]
        self._neck_prefix  = cfg["neck_prefix"]
        self._pe_key       = cfg["pos_embed_key"]
        self._pe_win_key   = cfg["pos_embed_win_key"]
        blocks_cfg  = cfg["hiera_blocks"]
        first_embed = blocks_cfg[0]["embed_dim"]
        self.patch_embed = PatchEmbed(embed_dim=first_embed)
        self.blocks = [MultiScaleBlock(
            dim=b["embed_dim"], dim_out=b["dim_out"],
            num_heads=b["num_heads"], window_size=b["window_size"],
            q_stride=tuple(b["q_stride"]) if b["q_stride"] else None,
        ) for b in blocks_cfg]
        self.pos_embed     = mx.zeros(cfg.get("pos_embed_shape",     [1,7,7,first_embed]))
        self.pos_embed_win = mx.zeros(cfg.get("pos_embed_win_shape", [1,8,8,first_embed]))
        backbone_ch = cfg.get("fpn_in_dims", [96,192,384,768])
        self.neck = FPNNeck(backbone_channel_list=backbone_ch,
                            d_model=cfg.get("fpn_out_dim", 256),
                            scalp=cfg.get("fpn_scalp", 1),
                            fpn_top_down_levels=cfg.get("fpn_top_down_levels", [2, 3]))
        self._stage_indices = set(self._find_stage_indices(blocks_cfg))

    @staticmethod
    def _find_stage_indices(blocks_cfg):
        indices = []
        for i, b in enumerate(blocks_cfg):
            is_last = (i == len(blocks_cfg)-1)
            next_ds = (not is_last and blocks_cfg[i+1]["q_stride"] is not None)
            if next_ds or is_last:
                indices.append(i)
        return indices

    def _apply_pos_embed(self, x):
        N, H, W, C = x.shape
        # pos_embed is (1, pH, pW, C) mx.array — cubic upsample to (1, H, W, C)
        # MLX uses 'cubic' mode (bicubic for 2-D)
        pe_up = _upsample(self.pos_embed, H, W, mode="cubic")

        # window pos embed: tile then crop — pure MLX, no NumPy roundtrip
        pe_win = self.pos_embed_win[0]          # (wH, wW, C)
        wH, wW = pe_win.shape[0], pe_win.shape[1]
        rh = math.ceil(H / wH); rw = math.ceil(W / wW)
        # tile via repeat + reshape
        pe_win_up = mx.concatenate([pe_win] * rh, axis=0)   # (rh*wH, wW, C)
        pe_win_up = mx.concatenate([pe_win_up] * rw, axis=1) # (rh*wH, rw*wW, C)
        pe_win_up = pe_win_up[:H, :W, :][None]               # (1, H, W, C)

        return x + pe_up + pe_win_up

    def __call__(self, x):
        x = x.transpose(0,2,3,1)
        x = self.patch_embed(x)
        x = self._apply_pos_embed(x)
        stage_features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self._stage_indices:
                stage_features.append(x)
        fpn_out  = self.neck(stage_features)
        fpn_nchw = [f.transpose(0,3,1,2) for f in fpn_out]
        # Apply conv_s0/conv_s1 projections here — matching ONNX/CoreML EncoderWrapper.
        # PyTorch high_res_feats are already post-projection (32ch, 64ch).
        # fpn_nchw[0]: (1,256,256,256) → conv_s0 → (1,32,256,256)
        # fpn_nchw[1]: (1,256,128,128) → conv_s1 → (1,64,128,128)
        dec = self._decoder_ref
        hr0 = dec.conv_s0(fpn_nchw[0].transpose(0,2,3,1)).transpose(0,3,1,2)
        hr1 = dec.conv_s1(fpn_nchw[1].transpose(0,2,3,1)).transpose(0,3,1,2)
        return {"vision_features": fpn_nchw[-1], "backbone_fpn": [hr0, hr1, fpn_nchw[-1]]}

    def load(self, weights):
        self.patch_embed.load(weights, self._patch_prefix)
        pe = _wopt(weights, self._pe_key)
        if pe is not None: self.pos_embed = pe
        pe_win = _wopt(weights, self._pe_win_key)
        if pe_win is not None: self.pos_embed_win = pe_win
        for i, block in enumerate(self.blocks):
            block.load(weights, self._block_prefix+f"{i}.")
        self.neck.load(weights, self._neck_prefix)


# ---------------------------------------------------------------------------
# PositionEmbeddingRandom / PromptEncoder
# ---------------------------------------------------------------------------

class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.positional_encoding_gaussian_matrix = mx.zeros((2, num_pos_feats))

    def _pe_encoding(self, coords):
        coords = 2*coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return mx.concatenate([mx.sin(coords), mx.cos(coords)], axis=-1)

    def forward_with_coords(self, coords, image_size):
        cn = mx.stack([
            coords[...,0].astype(mx.float32) / image_size[1],
            coords[...,1].astype(mx.float32) / image_size[0],
        ], axis=-1)
        return self._pe_encoding(cn)

    def __call__(self, size):
        H, W = size
        y    = (mx.arange(H, dtype=mx.float32) + 0.5) / H
        x    = (mx.arange(W, dtype=mx.float32) + 0.5) / W
        grid = mx.stack([mx.broadcast_to(x[None,:], (H,W)),
                         mx.broadcast_to(y[:,None], (H,W))], axis=-1)
        return self._pe_encoding(grid).transpose(2,0,1)

    def load(self, weights, prefix):
        m = _wopt(weights, prefix+"positional_encoding_gaussian_matrix")
        if m is not None: self.positional_encoding_gaussian_matrix = m


class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=256, image_embed_size=(64,64), input_image_size=(1024,1024)):
        super().__init__()
        self.embed_dim         = embed_dim
        self.image_embed_size  = image_embed_size
        self.input_image_size  = input_image_size
        self.pe_layer          = PositionEmbeddingRandom(embed_dim//2)
        self.point_embeddings  = [mx.zeros((embed_dim,)) for _ in range(4)]
        self.not_a_point_embed = mx.zeros((embed_dim,))
        self.no_mask_embed     = mx.zeros((embed_dim,))

    def _embed_points(self, coords, labels, pad=True):
        coords = coords + 0.5
        if pad:
            N = coords.shape[0]
            coords = mx.concatenate([coords, mx.zeros((N,1,2), dtype=mx.float32)], axis=1)
            labels = mx.concatenate([labels, mx.full((N,1), -1, dtype=mx.float32)], axis=1)

        pe = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        lbl = labels[..., None]

        # padding points (label==-1): PE must be ZEROED, only not_a_point_embed added
        # all other unlabeled points: pe + not_a_point_embed (default)
        out = mx.where(lbl == -1,
                    self.not_a_point_embed.reshape(1,1,-1),
                    pe + self.not_a_point_embed.reshape(1,1,-1))
        out = mx.where(lbl == 0, pe + self.point_embeddings[0].reshape(1,1,-1), out)
        out = mx.where(lbl == 1, pe + self.point_embeddings[1].reshape(1,1,-1), out)
        out = mx.where(lbl == 2, pe + self.point_embeddings[2].reshape(1,1,-1), out)
        out = mx.where(lbl == 3, pe + self.point_embeddings[3].reshape(1,1,-1), out)
        return out


    def get_dense_pe(self):
        return self.pe_layer(self.image_embed_size)[None, ...]

    def __call__(self, points=None, boxes=None, masks=None):
        if points is not None:
            coords, labels = points
            sparse = self._embed_points(coords, labels, pad=(boxes is None))
        else:
            sparse = mx.zeros((1, 0, self.embed_dim))
        h, w  = self.image_embed_size
        dense = mx.broadcast_to(self.no_mask_embed.reshape(1, self.embed_dim, 1, 1),
                                 (sparse.shape[0], self.embed_dim, h, w))
        return sparse, dense

    def load(self, weights, prefix):
        # Verify prefix exists at all — catches wrong prefix silently
        found = [k for k in weights if k.startswith(prefix)]
        if not found:
            raise KeyError(f"No weights found with prefix '{prefix}'. "
                           f"Available prefixes: {sorted(set(k.split('.')[0]+'.'+k.split('.')[1] for k in weights))}")
        self.pe_layer.load(weights, prefix+"pe_layer.")
        for i in range(4):
            k = _wopt(weights, prefix+f"point_embeddings.{i}.weight")
            if k is not None: self.point_embeddings[i] = k.reshape(-1)
        nap = _wopt(weights, prefix+"not_a_point_embed.weight")
        if nap is not None: self.not_a_point_embed = nap.reshape(-1)
        nme = _wopt(weights, prefix+"no_mask_embed.weight")
        if nme is not None: self.no_mask_embed = nme.reshape(-1)
        # Verify something actually loaded
        loaded = sum([
            any(_wopt(weights, prefix+f"point_embeddings.{i}.weight") is not None for i in range(4)),
            _wopt(weights, prefix+"pe_layer.positional_encoding_gaussian_matrix") is not None,
        ])
        if loaded == 0:
            raise KeyError(f"PromptEncoder: found prefix '{prefix}' but no expected sub-keys loaded. "
                           f"Sample keys: {found[:5]}")


# ---------------------------------------------------------------------------
# SAM2Attention / TwoWayAttentionBlock / MLP
# ---------------------------------------------------------------------------

class SAM2Attention(nn.Module):
    def __init__(self, embedding_dim, num_heads, downsample_rate=1, kv_in_dim=None):
        super().__init__()
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads    = num_heads
        self.head_dim     = self.internal_dim // num_heads
        kv_dim    = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.q_proj   = nn.Linear(embedding_dim,     self.internal_dim)
        self.k_proj   = nn.Linear(kv_dim,            self.internal_dim)
        self.v_proj   = nn.Linear(kv_dim,            self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim,  embedding_dim)

    def _sep(self, x, B, L):
        return x.reshape(B, L, self.num_heads, self.head_dim).transpose(0,2,1,3)

    def __call__(self, q, k, v):
        B, Lq, _ = q.shape; Lk = k.shape[1]
        q = self.q_proj(q); k = self.k_proj(k); v = self.v_proj(v)
        q = self._sep(q,B,Lq); k = self._sep(k,B,Lk); v = self._sep(v,B,Lk)
        attn = mx.softmax((q @ k.transpose(0,1,3,2)) * (self.head_dim**-0.5), axis=-1)
        out  = (attn @ v).transpose(0,2,1,3).reshape(B, Lq, self.num_heads*self.head_dim)
        return self.out_proj(out)

    def load(self, weights, prefix):
        for name in ("q_proj","k_proj","v_proj","out_proj"):
            proj = getattr(self, name)
            w = _wopt(weights, prefix+name+".weight")
            b = _wopt(weights, prefix+name+".bias")
            if w is not None: proj.weight = w
            if b is not None: proj.bias   = b


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=8, mlp_dim=2048, skip_first_layer_pe=False):
        super().__init__()
        self.skip_first_layer_pe = skip_first_layer_pe
        self.self_attn      = SAM2Attention(embedding_dim, num_heads, downsample_rate=1)
        self.norm1          = nn.LayerNorm(embedding_dim)
        self.cross_attn_t2i = SAM2Attention(embedding_dim, num_heads, downsample_rate=2)
        self.norm2          = nn.LayerNorm(embedding_dim)
        self.mlp_lin1       = nn.Linear(embedding_dim, mlp_dim)
        self.mlp_lin2       = nn.Linear(mlp_dim, embedding_dim)
        self.norm3          = nn.LayerNorm(embedding_dim)
        self.cross_attn_i2t = SAM2Attention(embedding_dim, num_heads, downsample_rate=2)
        self.norm4          = nn.LayerNorm(embedding_dim)

    def __call__(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q       = queries + query_pe
            queries = queries + self.self_attn(q, q, queries)
        queries = self.norm1(queries)
        q       = queries + query_pe; k = keys + key_pe
        queries = queries + self.cross_attn_t2i(q, k, keys)
        queries = self.norm2(queries)
        queries = queries + self.mlp_lin2(nn.relu(self.mlp_lin1(queries)))
        queries = self.norm3(queries)
        q       = queries + query_pe; k = keys + key_pe
        keys    = keys + self.cross_attn_i2t(k, q, queries)
        keys    = self.norm4(keys)
        return queries, keys

    def load(self, weights, prefix):
        self.self_attn.load(weights,      prefix+"self_attn.")
        self.cross_attn_t2i.load(weights, prefix+"cross_attn_token_to_image.")
        self.cross_attn_i2t.load(weights, prefix+"cross_attn_image_to_token.")
        for norm, name in [(self.norm1,"norm1"),(self.norm2,"norm2"),
                           (self.norm3,"norm3"),(self.norm4,"norm4")]:
            norm.weight = _w(weights, prefix+name+".weight")
            norm.bias   = _w(weights, prefix+name+".bias")
        p = prefix+"mlp."
        # Handle multiple naming conventions: lin1/lin2 (original SAM2), 
        # proj_in/proj_out (HF format), or layers.0/layers.1 (MedSAM2)
        # Only load one set to avoid creating a 3-layer MLP from 2-layer weights
        # Try all naming conventions, raise error if none found
        w0 = None
        b0 = None
        w1 = None
        b1 = None
        
        # Try lin1/lin2 (original SAM2)
        w0 = _wopt(weights, p+"lin1.weight")
        b0 = _wopt(weights, p+"lin1.bias")
        w1 = _wopt(weights, p+"lin2.weight")
        b1 = _wopt(weights, p+"lin2.bias")
        
        # If not found, try proj_in/proj_out (HF format)
        if w0 is None or w1 is None:
            w0 = _wopt(weights, p+"proj_in.weight")
            b0 = _wopt(weights, p+"proj_in.bias")
            w1 = _wopt(weights, p+"proj_out.weight")
            b1 = _wopt(weights, p+"proj_out.bias")
        
        # If still not found, try layers.0/layers.1 (MedSAM2)
        if w0 is None or w1 is None:
            w0 = _wopt(weights, p+"layers.0.weight")
            b0 = _wopt(weights, p+"layers.0.bias")
            w1 = _wopt(weights, p+"layers.1.weight")
            b1 = _wopt(weights, p+"layers.1.bias")
        
        # Raise error if still not found
        if w0 is None:
            raise KeyError(f"MLP weight not found: tried {p}lin1.weight, {p}proj_in.weight, {p}layers.0.weight")
        if w1 is None:
            raise KeyError(f"MLP weight not found: tried {p}lin2.weight, {p}proj_out.weight, {p}layers.1.weight")
        
        self.mlp_lin1.weight = w0
        if b0 is not None: self.mlp_lin1.bias = b0
        self.mlp_lin2.weight = w1
        if b1 is not None: self.mlp_lin2.bias = b1


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act=nn.relu):
        super().__init__()
        self.act    = act
        dims        = [input_dim] + [hidden_dim]*(num_layers-1) + [output_dim]
        self.layers = [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1:
                x = self.act(x)
        return x

    def load(self, weights, prefix):
        for i, layer in enumerate(self.layers):
            w = _wopt(weights, prefix+f"layers.{i}.weight")
            b = _wopt(weights, prefix+f"layers.{i}.bias")
            if w is not None: layer.weight = w
            if b is not None: layer.bias   = b


# ---------------------------------------------------------------------------
# MaskDecoder — EXACT token order from official MedSAM2 / SAM2
# ---------------------------------------------------------------------------

class MaskDecoder(nn.Module):
    def __init__(self, cfg, transformer_dim=256,
                 num_multimask_outputs=3, iou_head_depth=3, iou_head_hidden_dim=256):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_mask_tokens = num_multimask_outputs + 1
        self.obj_score_token = mx.zeros((1, transformer_dim))
        self.iou_token       = mx.zeros((1, transformer_dim))
        self.mask_tokens     = mx.zeros((self.num_mask_tokens, transformer_dim))
        n_layers = cfg.get("decoder_transformer_layers", 2)
        self.transformer = [
            TwoWayAttentionBlock(transformer_dim, num_heads=8, mlp_dim=2048,
                                 skip_first_layer_pe=(i==0))
            for i in range(n_layers)
        ]
        self.final_attn = SAM2Attention(transformer_dim, num_heads=8, downsample_rate=2)
        self.norm_final = nn.LayerNorm(transformer_dim)
        self.dc1        = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.ln1        = LayerNorm2d(64)
        self.dc2        = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_s0    = nn.Conv2d(256, 32, kernel_size=1)
        self.conv_s1    = nn.Conv2d(256, 64, kernel_size=1)
        self.hyper_mlps = [
            MLP(transformer_dim, transformer_dim, transformer_dim//8, num_layers=3)
            for _ in range(self.num_mask_tokens)
        ]
        self.iou_head = MLP(transformer_dim, iou_head_hidden_dim,
                            self.num_mask_tokens, num_layers=iou_head_depth)

    def __call__(self, image_embeddings, image_pe,
                 sparse_prompt_embeddings, dense_prompt_embeddings,
                 high_res_features, multimask_output=False):
        N = image_embeddings.shape[0]
        s = 1  # iou_token at position 1 (after obj_score_token)

        obj_tok = mx.broadcast_to(self.obj_score_token.reshape(1,1,256), (N,1,256))
        iou_tok = mx.broadcast_to(self.iou_token.reshape(1,1,256),       (N,1,256))
        msk_tok = mx.broadcast_to(self.mask_tokens.reshape(1,self.num_mask_tokens,256),
                                   (N,self.num_mask_tokens,256))
        tokens  = mx.concatenate([obj_tok, iou_tok, msk_tok, sparse_prompt_embeddings], axis=1)

        src    = image_embeddings + dense_prompt_embeddings
        _, C, H, W = [int(x) for x in src.shape]
        src_flat = src.transpose(0,2,3,1).reshape(N, H*W, C)
        pe_flat  = image_pe.transpose(0,2,3,1).reshape(N, H*W, C)

        queries, keys = tokens, src_flat
        for block in self.transformer:
            queries, keys = block(queries, keys, query_pe=tokens, key_pe=pe_flat)
        q       = queries + tokens; k = keys + pe_flat
        queries = queries + self.final_attn(q, k, keys)
        queries = self.norm_final(queries)

        iou_token_out   = queries[:, s, :]
        mask_tokens_out = queries[:, s+1:s+1+self.num_mask_tokens, :]

        image_src = keys.reshape(N, H, W, C)
        up        = self.dc1(image_src) + high_res_features[1].transpose(0,2,3,1)
        up        = nn.gelu(self.ln1(up))
        up        = nn.gelu(self.dc2(up) + high_res_features[0].transpose(0,2,3,1))

        masks_list = []
        for i in range(self.num_mask_tokens):
            m = self.hyper_mlps[i](mask_tokens_out[:, i, :])
            masks_list.append((m[:, None, None, :] * up).sum(axis=-1, keepdims=True))
        masks    = mx.concatenate(masks_list, axis=-1).transpose(0,3,1,2)
        iou_pred = mx.sigmoid(self.iou_head(iou_token_out))  # SAM2.1: iou_prediction_use_sigmoid=True

        if not multimask_output:
            masks    = masks[:, 0:1]
            iou_pred = iou_pred[:, 0:1]
        return masks, iou_pred

    def load(self, weights):
        p = "sam_mask_decoder."
        found = [k for k in weights if k.startswith(p)]
        if not found:
            raise KeyError(f"No weights found with prefix '{p}'")
        for attr, key in [("obj_score_token","obj_score_token.weight"),
                           ("iou_token",      "iou_token.weight"),
                           ("mask_tokens",    "mask_tokens.weight")]:
            v = _wopt(weights, p+key)
            if v is not None: setattr(self, attr, v)
        for i, block in enumerate(self.transformer):
            block.load(weights, p+f"transformer.layers.{i}.")
        self.final_attn.load(weights, p+"transformer.final_attn_token_to_image.")
        fn_w = _wopt(weights, p+"transformer.norm_final_attn.weight")
        fn_b = _wopt(weights, p+"transformer.norm_final_attn.bias")
        if fn_w is not None:
            self.norm_final.weight = fn_w
            self.norm_final.bias   = fn_b
        for conv, idx in [(self.dc1,"0"), (self.dc2,"3")]:
            w = _wopt(weights, p+f"output_upscaling.{idx}.weight")
            b = _wopt(weights, p+f"output_upscaling.{idx}.bias")
            if w is not None: conv.weight = w
            if b is not None: conv.bias   = b
        self.ln1.load(weights, p+"output_upscaling.1.")
        self.iou_head.load(weights, p+"iou_prediction_head.")
        for i, mlp in enumerate(self.hyper_mlps):
            mlp.load(weights, p+f"output_hypernetworks_mlps.{i}.")
        for sfx, conv in [("conv_s0",self.conv_s0), ("conv_s1",self.conv_s1)]:
            w = _wopt(weights, p+f"{sfx}.weight")
            b = _wopt(weights, p+f"{sfx}.bias")
            if w is not None: conv.weight = w
            if b is not None: conv.bias   = b
        # Verify transformer blocks actually loaded
        b0_w = _wopt(weights, p+"transformer.layers.0.norm1.weight")
        if b0_w is None:
            # Print what transformer keys actually exist to aid diagnosis
            tx_keys = sorted(k for k in weights if "transformer.layers.0" in k)
            raise KeyError(f"MaskDecoder transformer block 0 norm1 not loaded. "
                           f"Actual layer 0 keys: {tx_keys}")


# ---------------------------------------------------------------------------
# MedSAM2Predictor
# ---------------------------------------------------------------------------

class MedSAM2Predictor:
    def __init__(self, model_dir):
        self.model_dir   = Path(model_dir)
        weights_file     = self.model_dir / "weights.safetensors"
        config_file      = self.model_dir / "model_config.json"
        if not weights_file.exists():
            raise FileNotFoundError(f"Weights not found: {weights_file}")
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        print(f"Loading weights: {weights_file}")
        self.weights = mx.load(str(weights_file))
        print(f"✓ Loaded {len(self.weights)} tensors")
        with open(config_file) as f:
            self.cfg = json.load(f)
        blks = self.cfg["hiera_blocks"]
        print(f"✓ Config: {self.cfg['num_blocks']} blocks | "
              f"window_sizes={[b['window_size'] for b in blks]}")
        self.image_encoder  = HieraEncoder(self.cfg)
        self.prompt_encoder = PromptEncoder()
        self.mask_decoder   = MaskDecoder(self.cfg)
        # Give encoder a reference to decoder's conv_s0/conv_s1 projections
        # so it can apply them and return properly-shaped high_res features
        self.image_encoder._decoder_ref = self.mask_decoder
        self._load_all_weights()
        self.features      = None
        self.original_size = None
        self.pixel_mean    = mx.array([123.675, 116.28,  103.53 ], dtype=mx.float32)
        self.pixel_std     = mx.array([ 58.395,  57.12,   57.375], dtype=mx.float32)
        print("✓ MedSAM2Predictor ready")

    def _load_all_weights(self):
        print("Loading weights into model...")
        errors = []
        for fn, name in [
            (lambda: self.image_encoder.load(self.weights),                        "HieraEncoder"),
            (lambda: self.prompt_encoder.load(self.weights, "sam_prompt_encoder."), "PromptEncoder"),
            (lambda: self.mask_decoder.load(self.weights),                          "MaskDecoder"),
        ]:
            try:
                fn(); print(f"  ✓ {name}")
            except Exception as e:
                msg = f"{name}: {e}"
                errors.append(msg)
                print(f"  ✗ {msg}")   # always visible — not just a warning
                import traceback; traceback.print_exc()
        self._load_errors = errors

    def verify_weights(self):
        print("\n--- Weight Verification ---")
        print(f"Total keys: {len(self.weights)}")
        if self._load_errors:
            print(f"  ✗ {len(self._load_errors)} load error(s):")
            for e in self._load_errors:
                print(f"    {e}")
            return False          # ← was missing
        else:
            print("  ✓ All components loaded cleanly")
            return True   

    def preprocess_image(self, image):
        H, W    = image.shape[:2]
        img_mx  = mx.array(image.astype(np.float32)[None])      # (1, H, W, 3)
        resized = _upsample(img_mx, 1024, 1024, mode="linear")[0]  # (1024, 1024, 3)
        norm    = (resized - self.pixel_mean) / self.pixel_std
        return norm.transpose(2, 0, 1)[None, ...]                # (1, 3, 1024, 1024)

    def set_image(self, image):
        self.original_size = image.shape[:2]
        inp = self.preprocess_image(image)
        print("Running image encoder...")
        self.features = self.image_encoder(inp)
        mx.eval(self.features["vision_features"])
        print(f"✓ Encoded: vision_features={list(self.features['vision_features'].shape)} "
              f"fpn={[list(f.shape) for f in self.features['backbone_fpn']]}")

    def predict(self, point_coords, point_labels, multimask_output=False):
        if self.features is None:
            raise ValueError("Call set_image() first")
        h_orig, w_orig = self.original_size
        coords = point_coords.copy().astype(np.float32)
        coords[:, 0] *= 1024.0 / w_orig
        coords[:, 1] *= 1024.0 / h_orig
        # Pre-pad to 2 points (real + padding) to match PyTorch SAM2 convention.
        # The prompt encoder then adds one more padding token → 3 sparse tokens,
        # matching the ONNX/CoreML decoder exactly.
        pad_coords = np.zeros((1, 2), dtype=np.float32)
        pad_labels = np.array([-1], dtype=np.float32)
        coords = np.concatenate([coords, pad_coords], axis=0)
        labels_np = np.concatenate([point_labels.astype(np.float32), pad_labels], axis=0)
        coords_mx = mx.array(coords[None, ...])
        labels_mx = mx.array(labels_np[None, ...])

        high_res = [self.features["backbone_fpn"][0], self.features["backbone_fpn"][1]]
        sparse_emb, dense_emb = self.prompt_encoder(points=(coords_mx, labels_mx))
        image_pe = self.prompt_encoder.get_dense_pe()
        low_res_masks, iou_pred = self.mask_decoder(
            image_embeddings         = self.features["vision_features"],
            image_pe                 = image_pe,
            sparse_prompt_embeddings = sparse_emb,
            dense_prompt_embeddings  = dense_emb,
            high_res_features        = high_res,
            multimask_output         = multimask_output,
        )
        mx.eval(low_res_masks, iou_pred)

        # Batched nearest-neighbour upsample back to original size
        masks_nhwc = low_res_masks[0].transpose(1, 2, 0)[None]              # (1, H, W, N_masks)
        masks_up   = _upsample(masks_nhwc, h_orig, w_orig, mode="nearest")[0]  # (h_orig, w_orig, N_masks)
        masks_np   = (np.array(masks_up) > 0.0).transpose(2, 0, 1)          # (N_masks, h_orig, w_orig)

        iou_np = np.array(iou_pred[0], dtype=np.float32)
        print(f"✓ Predict: masks={masks_np.shape} IoU={iou_np}")
        return masks_np, iou_np


# ---------------------------------------------------------------------------
# MLXPredictor - Wrapper for Blender extension compatibility
# ---------------------------------------------------------------------------

class MLXPredictor:
    """
    MedSAM2 predictor using MLX backend
    
    Wrapper around MedSAM2Predictor to match the expected interface
    for the Blender extension.
    """
    
    def __init__(self, models_dir):
        """
        Initialize the MLX predictor
        
        Args:
            models_dir: Path to directory containing medsam2_mlx/
        """
        self.models_dir = Path(models_dir)
        self.model_path = self.models_dir / "medsam2_mlx"
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MLX model directory not found: {self.model_path}\n"
                "Make sure medsam2_mlx/ folder exists with weights.safetensors and model_config.json"
            )
        
        print(f"Loading MLX MedSAM2 from {self.model_path}")
        self.predictor = MedSAM2Predictor(str(self.model_path))
        
        if not self.predictor.verify_weights():
            print("⚠️  Weight conversion issues detected")
        
        print("✓ MLX MedSAM2 loaded")
    
    def segment(self, image, points, labels=None):
        """
        Segment image with point prompts
        
        Args:
            image: Input image (H, W, 3) or (H, W) in [0, 255]
            points: List of (x, y) coordinates or array (N, 2)
            labels: List of labels (1=positive, 0=negative) or None (all positive)
            
        Returns:
            dict: Segmentation results
                'mask': Binary mask (H, W) as uint8
                'iou': IoU prediction score
        """
        # Convert to numpy arrays
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float32)
        
        # Default labels to all positive
        if labels is None:
            labels = np.ones(len(points), dtype=np.int32)
        elif not isinstance(labels, np.ndarray):
            labels = np.array(labels, dtype=np.int32)
        
        # Handle grayscale images
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Ensure uint8 in [0, 255]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Set image (encodes it)
        self.predictor.set_image(image)
        
        # Run prediction
        masks, iou_predictions = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )
        
        # Extract results
        mask = masks[0].astype(np.uint8)  # (H, W) binary
        iou = float(iou_predictions[0])
        
        return {
            'mask': mask,
            'iou': iou,
        }
