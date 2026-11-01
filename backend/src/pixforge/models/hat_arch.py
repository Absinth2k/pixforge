# HAT: Hybrid Attention Transformer for image super-resolution.
#
# Origin: https://github.com/XPixelGroup/HAT  (hat/archs/hat_arch.py)
# Copyright (c) XPixelGroup, licensed under the Apache License 2.0.
# Parts derived from SwinIR (Apache-2.0) and timm (Apache-2.0).
#
# Modifications for Pixforge (also Apache-2.0):
#   - removed basicsr / einops dependencies (to_2tuple, trunc_normal_, rearrange inlined)
#   - removed training-only code paths (DropPath, checkpointing, PatchMerging, ape)
#   - attention mask is a registered buffer for a fixed tile size, so ONNX export
#     with a fixed input shape folds it into a constant
#   - type hints, English comments
"""HAT network definition (inference only)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from pixforge.models.hat_mask import shifted_window_mask


def _to_2tuple(x: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(x, Sequence):
        return int(x[0]), int(x[1])
    return int(x), int(x)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(b, h, w, c) -> (num_windows*b, ws, ws, c)."""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, b: int, h: int, w: int) -> torch.Tensor:
    """(num_windows*b, ws, ws, c) -> (b, h, w, c)."""
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN."""

    def __init__(self, num_feat: int, squeeze_factor: int = 16) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class CAB(nn.Module):
    """Channel attention block."""

    def __init__(self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30) -> None:
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    """Window based multi-head self attention with relative position bias."""

    def __init__(self, dim: int, window_size: tuple[int, int], num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, rpi: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        ws = self.window_size[0] * self.window_size[1]
        bias = self.relative_position_bias_table[rpi.view(-1)].view(ws, ws, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(x)


class HAB(nn.Module):
    """Hybrid attention block: (shifted) window attention plus a convolutional branch."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int,
        shift_size: int,
        compress_ratio: int,
        squeeze_factor: int,
        conv_scale: float,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, _to_2tuple(self.window_size), num_heads)
        self.conv_scale = conv_scale
        self.conv_block = CAB(dim, compress_ratio, squeeze_factor)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
        rpi_sa: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask: torch.Tensor | None = attn_mask
        else:
            shifted_x = x
            mask = None

        ws = self.window_size
        x_windows = window_partition(shifted_x, ws).view(-1, ws * ws, c)
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=mask).view(-1, ws, ws, c)
        shifted_x = window_reverse(attn_windows, ws, b, h, w)

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        x = shortcut + attn_x + conv_x * self.conv_scale
        return x + self.mlp(self.norm2(x))


class OCAB(nn.Module):
    """Overlapping cross-attention block."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        overlap_ratio: float,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size + self.overlap_win_size - 1) ** 2,
                num_heads,
            )
        )
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, x_size: tuple[int, int], rpi: torch.Tensor) -> torch.Tensor:
        h, w = x_size
        b, _, c = x.shape
        ws, ows = self.window_size, self.overlap_win_size

        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)  # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1)  # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1)  # b, 2c, h, w

        q_windows = window_partition(q, ws).view(-1, ws * ws, c)

        kv_windows = self.unfold(kv)  # b, 2*c*ows*ows, nw
        nw = kv_windows.shape[-1]
        # einops: 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch'
        kv_windows = (
            kv_windows.view(b, 2, c, ows, ows, nw)
            .permute(1, 0, 5, 3, 4, 2)
            .reshape(2, b * nw, ows * ows, c)
            .contiguous()
        )
        k_windows, v_windows = kv_windows[0], kv_windows[1]

        b_, nq, _ = q_windows.shape
        n = k_windows.shape[1]
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[rpi.view(-1)].view(ws * ws, ows * ows, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        attn = self.softmax(attn)

        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)
        attn_windows = attn_windows.view(-1, ws, ws, self.dim)
        x = window_reverse(attn_windows, ws, b, h, w).view(b, h * w, self.dim)

        x = self.proj(x) + shortcut
        return x + self.mlp(self.norm2(x))


class AttenBlocks(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        compress_ratio: int,
        squeeze_factor: int,
        conv_scale: float,
        overlap_ratio: float,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                HAB(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(depth)
            ]
        )
        self.overlap_attn = OCAB(dim, window_size, overlap_ratio, num_heads, mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
        rpi_sa: torch.Tensor,
        attn_mask: torch.Tensor,
        rpi_oca: torch.Tensor,
    ) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, x_size, rpi_sa, attn_mask)
        return self.overlap_attn(x, x_size, rpi_oca)


class RHAG(nn.Module):
    """Residual hybrid attention group."""

    def __init__(self, dim: int, **block_kwargs: object) -> None:
        super().__init__()
        self.dim = dim
        self.residual_group = AttenBlocks(dim=dim, **block_kwargs)  # type: ignore[arg-type]
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
        rpi_sa: torch.Tensor,
        attn_mask: torch.Tensor,
        rpi_oca: torch.Tensor,
    ) -> torch.Tensor:
        y = self.residual_group(x, x_size, rpi_sa, attn_mask, rpi_oca)
        y = y.transpose(1, 2).contiguous().view(x.shape[0], self.dim, x_size[0], x_size[1])
        y = self.conv(y).flatten(2).transpose(1, 2)
        return y + x


class Upsample(nn.Sequential):
    def __init__(self, scale: int, num_feat: int) -> None:
        m: list[nn.Module] = []
        if scale & (scale - 1) == 0:
            for _ in range(int(math.log2(scale))):
                m += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)]
        elif scale == 3:
            m += [nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"scale {scale} is not supported (2^n and 3 only)")
        super().__init__(*m)


@dataclass(frozen=True)
class HATConfig:
    upscale: int = 4
    embed_dim: int = 180
    depths: tuple[int, ...] = field(default=(6, 6, 6, 6, 6, 6))
    num_heads: tuple[int, ...] = field(default=(6, 6, 6, 6, 6, 6))
    window_size: int = 16
    compress_ratio: int = 3
    squeeze_factor: int = 30
    conv_scale: float = 0.01
    overlap_ratio: float = 0.5
    mlp_ratio: float = 2.0
    img_range: float = 1.0


# Variants as defined in the upstream options/test/*.yml files.
HAT_VARIANTS: dict[str, dict[str, object]] = {
    "HAT": {},
    "HAT-S": {"embed_dim": 144, "compress_ratio": 24, "squeeze_factor": 24},
    "HAT-L": {"depths": (6,) * 12, "num_heads": (6,) * 12},
}


def hat_config(variant: str, upscale: int) -> HATConfig:
    return HATConfig(upscale=upscale, **HAT_VARIANTS[variant])  # type: ignore[arg-type]


class HAT(nn.Module):
    """Hybrid Attention Transformer (pixelshuffle upsampler, RGB in/out, range 0..1)."""

    def __init__(self, cfg: HATConfig, tile_size: int = 64) -> None:
        super().__init__()
        self.cfg = cfg
        self.window_size = cfg.window_size
        self.shift_size = cfg.window_size // 2
        self.overlap_ratio = cfg.overlap_ratio
        self.upscale = cfg.upscale
        self.img_range = cfg.img_range
        num_feat = 64

        self.register_buffer(
            "mean", torch.tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer("relative_position_index_SA", self._calculate_rpi_sa())
        self.register_buffer("relative_position_index_OCA", self._calculate_rpi_oca())

        self.conv_first = nn.Conv2d(3, cfg.embed_dim, 3, 1, 1)
        # Upstream PatchEmbed with patch_size=1 and patch_norm=True: flatten + LayerNorm.
        self.patch_embed_norm = nn.LayerNorm(cfg.embed_dim)

        self.layers = nn.ModuleList(
            [
                RHAG(
                    dim=cfg.embed_dim,
                    input_resolution=(tile_size, tile_size),
                    depth=depth,
                    num_heads=heads,
                    window_size=cfg.window_size,
                    compress_ratio=cfg.compress_ratio,
                    squeeze_factor=cfg.squeeze_factor,
                    conv_scale=cfg.conv_scale,
                    overlap_ratio=cfg.overlap_ratio,
                    mlp_ratio=cfg.mlp_ratio,
                )
                for depth, heads in zip(cfg.depths, cfg.num_heads, strict=True)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.conv_after_body = nn.Conv2d(cfg.embed_dim, cfg.embed_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(cfg.embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(cfg.upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)

    # ---- index / mask helpers (identical math to upstream) ----
    def _calculate_rpi_sa(self) -> torch.Tensor:
        ws = self.window_size
        coords = torch.stack(torch.meshgrid([torch.arange(ws), torch.arange(ws)], indexing="ij"))
        flat = torch.flatten(coords, 1)
        rel = (flat[:, :, None] - flat[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        return rel.sum(-1)

    def _calculate_rpi_oca(self) -> torch.Tensor:
        ws = self.window_size
        wse = ws + int(self.overlap_ratio * ws)
        ori = torch.flatten(
            torch.stack(torch.meshgrid([torch.arange(ws), torch.arange(ws)], indexing="ij")), 1
        )
        ext = torch.flatten(
            torch.stack(torch.meshgrid([torch.arange(wse), torch.arange(wse)], indexing="ij")), 1
        )
        rel = (ext[:, None, :] - ori[:, :, None]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - wse + 1
        rel[:, :, 1] += ws - wse + 1
        rel[:, :, 0] *= ws + wse - 1
        return rel.sum(-1)

    def calculate_mask(self, h: int, w: int) -> torch.Tensor:
        return torch.from_numpy(shifted_window_mask(h, w, self.window_size).copy())

    def forward_features(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # No int() conversions here: under ONNX tracing these stay symbolic,
        # so the exported graph accepts any window-aligned tile size.
        b, h, w = x.shape[0], x.shape[2], x.shape[3]
        x = self.patch_embed_norm(x.flatten(2).transpose(1, 2))
        for layer in self.layers:
            x = layer(
                x,
                (h, w),
                self.relative_position_index_SA,
                attn_mask,
                self.relative_position_index_OCA,
            )
        x = self.norm(x)
        return x.transpose(1, 2).contiguous().view(b, self.cfg.embed_dim, h, w)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (b, 3, h, w) RGB in 0..1, h and w multiples of the window size.

        attn_mask: output of ``shifted_window_mask(h, w)``; computed on the fly if omitted.
        """
        if attn_mask is None:
            attn_mask = self.calculate_mask(int(x.shape[2]), int(x.shape[3])).to(x)
        mean = self.mean.to(x.dtype)
        x = (x - mean) * self.img_range
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x, attn_mask)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
        return x / self.img_range + mean


def load_hat_state_dict(model: HAT, state: dict[str, torch.Tensor]) -> None:
    """Load an upstream checkpoint (params_ema / params) into the Pixforge HAT module."""
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    remapped: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("patch_embed.norm."):
            k = k.replace("patch_embed.norm.", "patch_embed_norm.")
        if k.endswith("attn_mask") or ".patch_embed." in k or ".patch_unembed." in k:
            continue
        remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    missing = [m for m in missing if not m.endswith("attn_mask")]
    if missing or unexpected:
        raise RuntimeError(f"HAT checkpoint mismatch: missing={missing} unexpected={unexpected}")
