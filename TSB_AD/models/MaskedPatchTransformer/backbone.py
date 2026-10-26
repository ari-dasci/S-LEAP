"""Factorized transformer: attention along time, then along channels (optional latent bottleneck)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head attention via scaled_dot_product_attention (fused kernels when available)."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f'd_model={d_model} is not divisible by n_heads={n_heads}')
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, kv=None):
        # x: [B, S, D]; kv: [B, T, D] for cross-attention (defaults to self)
        B, S, D = x.shape
        if kv is None:
            q, k, v = self.qkv(x).chunk(3, dim=-1)
        else:
            q = F.linear(x, self.qkv.weight[:D])
            k, v = F.linear(kv, self.qkv.weight[D:]).chunk(2, dim=-1)
        shape = lambda t: t.view(t.shape[0], t.shape[1], self.n_heads,
                                 self.d_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            shape(q), shape(k), shape(v),
            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, S, D))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TemporalAttention(nn.Module):
    """Attention over the patch axis, each channel independently."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(self, h):
        # h: [B, C, N, D]
        B, C, N, D = h.shape
        z = self.norm(h).reshape(B * C, N, D)
        return h + self.attn(z).view(B, C, N, D)


class ChannelAttention(nn.Module):
    """Attention over channels at each patch position; full C×C, or through K latents if n_latents is set."""

    def __init__(self, d_model, n_heads, dropout=0.0, n_latents=None):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.n_latents = n_latents
        if n_latents is None:
            self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        else:
            self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
            self.write = MultiHeadAttention(d_model, n_heads, dropout)
            self.read = MultiHeadAttention(d_model, n_heads, dropout)
            self.latent_norm = nn.LayerNorm(d_model)

    def forward(self, h):
        # h: [B, C, N, D] → attend across C for every (B, N)
        B, C, N, D = h.shape
        z = self.norm(h).permute(0, 2, 1, 3).reshape(B * N, C, D)
        if self.n_latents is None:
            out = self.attn(z)
        else:
            lat = self.latents.unsqueeze(0).expand(B * N, -1, -1)
            lat = lat + self.write(lat, kv=z)          # channels → latents
            out = self.read(z, kv=self.latent_norm(lat))   # latents → channels
        out = out.view(B, N, C, D).permute(0, 2, 1, 3)
        return h + out


class Block(nn.Module):
    """One encoder block: time attention → channel attention → FFN."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, n_latents=None):
        super().__init__()
        self.temporal = TemporalAttention(d_model, n_heads, dropout)
        self.channel = ChannelAttention(d_model, n_heads, dropout, n_latents)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, h):
        h = self.temporal(h)
        h = self.channel(h)
        return h + self.ff(self.ff_norm(h))


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, depth, dropout=0.0,
                 n_latents=None, gradient_checkpointing=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, n_latents)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, h):
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                h = torch.utils.checkpoint.checkpoint(block, h,
                                                      use_reentrant=False)
            else:
                h = block(h)
        return self.norm(h)


def sinusoidal_embedding(length, d_model, device=None):
    """Fixed sin/cos table [length, d_model]."""
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device,
                                 dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe
