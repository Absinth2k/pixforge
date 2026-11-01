# SCUNet: Swin-Conv-UNet for blind real image denoising.
#
# Origin: https://github.com/cszn/SCUNet (models/network_scunet.py), also in cszn/KAIR
# Copyright (c) Kai Zhang et al., licensed under the Apache License 2.0.
#
# Modifications for Pixforge (also Apache-2.0):
#   - removed einops / timm / thop dependencies (reshape/permute instead of rearrange)
#   - shifted-window mask built with tensor ops from the (dynamic) window grid, so the
#     ONNX export accepts any tile size that is a multiple of 64
#   - padding to a multiple of 64 moved out of the model (the tiler aligns tiles)
#   - inference only, type hints
"""SCUNet network definition (inference only)."""

from __future__ import annotations

import torch
from torch import nn


def _shifted_mask(hw: torch.Tensor | int, ww: torch.Tensor | int, p: int) -> torch.Tensor:
    """Bool mask (1, 1, hw*ww, p*p, p*p), True = blocked. Same as upstream generate_mask."""
    s = p - p // 2
    a = torch.arange(p) < s
    cross = a[:, None] != a[None, :]  # (p_query, p_key)
    row_last = torch.arange(hw) == (hw - 1)
    col_last = torch.arange(ww) == (ww - 1)
    m1 = row_last[:, None, None, None, None, None] & cross[None, None, :, None, :, None]
    m2 = col_last[None, :, None, None, None, None] & cross[None, None, None, :, None, :]
    return (m1 | m2).reshape(1, 1, hw * ww, p * p, p * p)


class WMSA(nn.Module):
    """(Shifted) window multi-head self attention."""

    def __init__(self, input_dim: int, output_dim: int, head_dim: int, window_size: int,
                 type: str) -> None:  # noqa: A002 - upstream name
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(input_dim, 3 * input_dim, bias=True)
        self.relative_position_params = nn.Parameter(
            torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size - 1)
        )
        self.linear = nn.Linear(input_dim, output_dim)
        p = window_size
        cord = torch.stack(
            torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij"), -1
        ).reshape(-1, 2)
        relation = cord[:, None, :] - cord[None, :, :] + p - 1
        self.register_buffer("rel_index", relation, persistent=False)

    def relative_embedding(self) -> torch.Tensor:
        r = self.rel_index
        return self.relative_position_params[:, r[:, :, 0], r[:, :, 1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (b, h, w, c)."""
        p = self.window_size
        if self.type != "W":
            x = torch.roll(x, shifts=(-(p // 2), -(p // 2)), dims=(1, 2))
        b, h, w, c = x.shape
        hw, ww = h // p, w // p
        x = x.view(b, hw, p, ww, p, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hw * ww, p * p, c)
        qkv = self.embedding_layer(x)
        nh, hd = self.n_heads, self.head_dim
        qkv = qkv.view(b, hw * ww, p * p, 3 * nh, hd).permute(3, 0, 1, 2, 4)
        nw, n = hw * ww, p * p
        # DirectML's (Fused)MatMul rejects 5-D operands: fold heads into the batch dim.
        q = qkv[:nh].reshape(nh * b, nw, n, hd)
        k = qkv[nh : 2 * nh].reshape(nh * b, nw, n, hd)
        v = qkv[2 * nh :].reshape(nh * b, nw, n, hd)
        sim = ((q @ k.transpose(-1, -2)) * self.scale).view(nh, b, nw, n, n)
        sim = sim + self.relative_embedding()[:, None, None]
        if self.type != "W":
            sim = sim.masked_fill(_shifted_mask(hw, ww, p).to(sim.device), float("-inf"))
        probs = torch.softmax(sim, dim=-1).reshape(nh * b, nw, n, n)
        out = (probs @ v).view(nh, b, nw, n, hd)
        out = out.permute(1, 2, 3, 0, 4).reshape(b, nw, n, nh * hd)
        out = self.linear(out)
        out = out.view(b, hw, ww, p, p, -1).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)
        if self.type != "W":
            out = torch.roll(out, shifts=(p // 2, p // 2), dims=(1, 2))
        return out


class Block(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, head_dim: int, window_size: int,
                 type: str, input_resolution: int) -> None:  # noqa: A002
        super().__init__()
        if input_resolution <= window_size:
            type = "W"
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, type)
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim), nn.GELU(), nn.Linear(4 * input_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.msa(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim: int, trans_dim: int, head_dim: int, window_size: int,
                 type: str, input_resolution: int) -> None:  # noqa: A002
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        if input_resolution <= window_size:
            type = "W"
        self.trans_block = Block(trans_dim, trans_dim, head_dim, window_size, type,
                                 input_resolution)
        total = conv_dim + trans_dim
        self.conv1_1 = nn.Conv2d(total, total, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(total, total, 1, 1, 0, bias=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.conv1_2(torch.cat((conv_x, trans_x), dim=1))


class SCUNet(nn.Module):
    """Input/output: (b, 3, h, w) RGB 0..1, h and w multiples of 64."""

    alignment = 64

    def __init__(self, in_nc: int = 3, config: tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4),
                 dim: int = 64, input_resolution: int = 256) -> None:
        super().__init__()
        hd, ws = 32, 8

        def stage(n: int, d: int, res: int) -> list[nn.Module]:
            return [ConvTransBlock(d, d, hd, ws, "W" if i % 2 == 0 else "SW", res)
                    for i in range(n)]

        r = input_resolution
        self.m_head = nn.Sequential(nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False))
        self.m_down1 = nn.Sequential(*stage(config[0], dim // 2, r),
                                     nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False))
        self.m_down2 = nn.Sequential(*stage(config[1], dim, r // 2),
                                     nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False))
        self.m_down3 = nn.Sequential(*stage(config[2], 2 * dim, r // 4),
                                     nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False))
        self.m_body = nn.Sequential(*stage(config[3], 4 * dim, r // 8))
        self.m_up3 = nn.Sequential(nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False),
                                   *stage(config[4], 2 * dim, r // 4))
        self.m_up2 = nn.Sequential(nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False),
                                   *stage(config[5], dim, r // 2))
        self.m_up1 = nn.Sequential(nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False),
                                   *stage(config[6], dim // 2, r))
        self.m_tail = nn.Sequential(nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)
