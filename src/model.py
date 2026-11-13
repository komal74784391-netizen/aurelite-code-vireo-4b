"""Model definition for Vireo-4B.

Reference implementation used in the Aurelite Vireo-4B paper.
"""

import math

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(norm + self.eps)).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(y)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.ffn(self.ffn_norm(x))


class Vireo4bModel(nn.Module):
    """Vireo-4B decoder."""

    def __init__(self, vocab_size: int = 32000, dim: int = 2560, n_layers: int = 32,
                 n_heads: int = 20, hidden: int = 6912):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads, hidden) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        mask = torch.tril(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.norm(x))
