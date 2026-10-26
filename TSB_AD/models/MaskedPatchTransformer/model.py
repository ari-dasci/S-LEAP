"""Masked patch transformer: predict removed patches from spatio-temporal context."""

import torch
import torch.nn as nn

from .backbone import Encoder, sinusoidal_embedding
from .masking import mask_for_group


class MaskedPatchTransformer(nn.Module):
    """Predicts masked (channel, patch) tokens from the visible ones: x [B, L, C], mask [C, N] → pred [B, L, C]."""

    def __init__(self, n_channels, patch_size, win_size,
                 d_model=128, n_heads=8, depth=4, d_ff=None, dropout=0.0,
                 n_latents=None, gradient_checkpointing=False):
        super().__init__()
        if win_size % patch_size:
            raise ValueError(
                f'win_size={win_size} must be a multiple of patch_size={patch_size}')
        self.n_channels = int(n_channels)
        self.patch_size = int(patch_size)
        self.win_size = int(win_size)
        self.n_patches = self.win_size // self.patch_size
        d_ff = d_ff if d_ff is not None else 4 * d_model

        self.embed = nn.Linear(self.patch_size, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.channel_embed = nn.Parameter(
            torch.randn(self.n_channels, d_model) * 0.02)
        self.register_buffer(
            'pos_embed', sinusoidal_embedding(self.n_patches, d_model),
            persistent=False)

        self.encoder = Encoder(d_model, n_heads, d_ff, depth, dropout,
                               n_latents, gradient_checkpointing)
        self.head = nn.Linear(d_model, self.patch_size)

    # -- patching ---------------------------------------------------------

    def patchify(self, x):
        """[B, L, C] → [B, C, N, p]"""
        B, L, C = x.shape
        return (x.permute(0, 2, 1)
                 .reshape(B, C, self.n_patches, self.patch_size))

    def unpatchify(self, patches):
        """[B, C, N, p] → [B, L, C]"""
        B, C = patches.shape[0], patches.shape[1]
        return patches.reshape(B, C, self.win_size).permute(0, 2, 1)

    # -- forward ----------------------------------------------------------

    def forward(self, x, mask):
        """x: [B, L, C]; mask: [C, N] bool → pred [B, L, C]"""
        patches = self.patchify(x)                       # [B, C, N, p]
        h = self.embed(patches)                          # [B, C, N, D]

        # Masked tokens are replaced by the learned [MASK] embedding.
        m = mask.unsqueeze(0).unsqueeze(-1)              # [1, C, N, 1]
        h = torch.where(m, self.mask_token.expand_as(h), h)

        h = h + self.pos_embed.view(1, 1, self.n_patches, -1)
        h = h + self.channel_embed.view(1, self.n_channels, 1, -1)

        h = self.encoder(h)
        return self.unpatchify(self.head(h))             # [B, L, C]

    def masked_error(self, x, groups, block=1, rotation=0, reduction='abs'):
        """Prediction error over the G complementary passes, each token scored while hidden → [B, L, C]."""
        B, L, C = x.shape
        out = torch.zeros_like(x)
        for g in range(groups):
            mask = mask_for_group(C, self.n_patches, groups, g, block,
                                  rotation, device=x.device)      # [C, N]
            pred = self(x, mask)
            err = (x - pred).abs() if reduction == 'abs' else (x - pred) ** 2
            # Keep only the positions that were actually hidden in this pass.
            keep = (mask.repeat_interleave(self.patch_size, dim=1)  # [C, L]
                    .t().unsqueeze(0))                             # [1, L, C]
            out = torch.where(keep, err, out)
        return out
