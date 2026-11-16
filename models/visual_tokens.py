"""
Helpers for turning CNN feature maps into a sequence of visual tokens that a
transformer decoder can attend over (instead of a single pooled vector).
"""

import math
import torch
import torch.nn as nn


def sinusoidal_2d(height, width, dim, device):
    """2D sinusoidal position encoding [height*width, dim]; works for any grid size."""
    assert dim % 4 == 0, "dim must be divisible by 4"
    quarter = dim // 4
    freq = torch.exp(torch.arange(quarter, device=device, dtype=torch.float32)
                     * (-math.log(10000.0) / quarter))
    ys = torch.arange(height, device=device, dtype=torch.float32)[:, None] * freq  # [H, q]
    xs = torch.arange(width, device=device, dtype=torch.float32)[:, None] * freq   # [W, q]
    pe_y = torch.cat([ys.sin(), ys.cos()], dim=-1)[:, None, :].expand(height, width, 2 * quarter)
    pe_x = torch.cat([xs.sin(), xs.cos()], dim=-1)[None, :, :].expand(height, width, 2 * quarter)
    return torch.cat([pe_y, pe_x], dim=-1).reshape(height * width, dim)


class FeatureTokens(nn.Module):
    """[B, C, H, W] feature map -> [B, H*W, dim] tokens with 2D position + modality embedding."""

    def __init__(self, in_channels, dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.modality_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, fmap):
        x = self.proj(fmap)                                   # [B, dim, H, W]
        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)                      # [B, H*W, dim]
        x = x + sinusoidal_2d(h, w, d, x.device).unsqueeze(0) + self.modality_embedding
        return self.dropout(self.norm(x))


def build_memory(token_groups):
    """
    Concatenate per-modality token groups into one memory sequence.

    token_groups: list of (tokens [B, N_i, D], present [B] bool or None)
    Returns (memory [B, N, D], padding_mask [B, N] bool, True = ignore).
    A sample whose modalities are all missing keeps every token visible, so attention
    never sees a fully masked row (which would produce NaNs).
    """
    tokens, masks = [], []
    for t, present in token_groups:
        if present is None:
            present = torch.ones(t.size(0), dtype=torch.bool, device=t.device)
        tokens.append(t)
        masks.append((~present.to(t.device).bool())[:, None].expand(-1, t.size(1)))
    memory = torch.cat(tokens, dim=1)
    mask = torch.cat(masks, dim=1)
    all_masked = mask.all(dim=1)
    if all_masked.any():
        mask = mask.clone()
        mask[all_masked] = False
    return memory, mask
