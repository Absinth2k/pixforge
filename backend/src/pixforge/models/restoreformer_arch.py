# RestoreFormer++: blind face restoration with a multi-scale cross-attention VQ autoencoder.
#
# Origin: https://github.com/wzhouxiff/RestoreFormerPlusPlus (gradio_demo/RestoreFormer_arch.py)
# Copyright (c) Zhouxia Wang et al., licensed under the Apache License 2.0. Encoder/decoder
# blocks derive from taming-transformers (MIT) and pytorch_diffusion.
#
# Modifications for Pixforge (also Apache-2.0): inference only (no losses / perplexity),
# vector quantisation as argmin + embedding lookup (same result as the one-hot matmul,
# exports cleanly to ONNX), no in-place ops, type hints, configuration fixed to the
# released RestoreFormer++ weights (head_size=4, one extra decoder attention scale).
"""RestoreFormer++ network definition (inference only). Input/output: (b, 3, 512, 512) in -1..1."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _norm(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=c, eps=1e-6, affine=True)


class VectorQuantizer(nn.Module):
    def __init__(self, n_e: int, e_dim: int) -> None:
        super().__init__()
        self.e_dim = e_dim
        self.embedding = nn.Embedding(n_e, e_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(-1, self.e_dim)
        e = self.embedding.weight
        d = (zf**2).sum(1, keepdim=True) + (e**2).sum(1) - 2 * zf @ e.t()
        idx = torch.argmin(d, dim=1)
        return self.embedding(idx).view(b, h, w, c).permute(0, 3, 1, 2)


class Upsample(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Downsample(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 2, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.norm1 = _norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = _norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(_swish(self.norm1(x)))
        h = self.conv2(_swish(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class MultiHeadAttnBlock(nn.Module):
    def __init__(self, c: int, head_size: int) -> None:
        super().__init__()
        self.head_size, self.att_size = head_size, c // head_size
        self.norm1, self.norm2 = _norm(c), _norm(c)
        self.q = nn.Conv2d(c, c, 1)
        self.k = nn.Conv2d(c, c, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.proj_out = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        h_ = self.norm1(x)
        y = h_ if y is None else self.norm2(y)
        q, k, v = self.q(y), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        hs, a = self.head_size, self.att_size
        q = q.reshape(b, hs, a, h * w).permute(0, 1, 3, 2)  # b, head, hw, att
        k = k.reshape(b, hs, a, h * w)  # b, head, att, hw
        v = v.reshape(b, hs, a, h * w).permute(0, 1, 3, 2)
        attn = torch.softmax((q * a**-0.5) @ k, dim=3)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x + self.proj_out(out)


class _Level(nn.Module):
    """Container with upstream attribute names (block, attn, downsample/upsample)."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.ModuleList()
        self.attn = nn.ModuleList()


class Encoder(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: tuple[int, ...],
        nrb: int,
        attn_res: tuple[int, ...],
        resolution: int,
        z_channels: int,
        head_size: int,
    ) -> None:
        super().__init__()
        self.nres, self.nrb = len(ch_mult), nrb
        self.conv_in = nn.Conv2d(3, ch, 3, 1, 1)
        in_mult = (1, *ch_mult)
        res = resolution
        self.down = nn.ModuleList()
        block_in = ch
        for i in range(self.nres):
            lvl = _Level()
            block_in, block_out = ch * in_mult[i], ch * ch_mult[i]
            for _ in range(nrb):
                lvl.block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
                if res in attn_res:
                    lvl.attn.append(MultiHeadAttnBlock(block_in, head_size))
            if i != self.nres - 1:
                lvl.downsample = Downsample(block_in)
                res //= 2
            self.down.append(lvl)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.norm_out = _norm(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hs: dict[str, torch.Tensor] = {}
        h = self.conv_in(x)
        for i in range(self.nres):
            lvl = self.down[i]
            for j in range(self.nrb):
                h = lvl.block[j](h)
                if len(lvl.attn) > 0:
                    h = lvl.attn[j](h)
            if i != self.nres - 1:
                hs[f"block_{i}"] = h
                h = lvl.downsample(h)
        h = self.mid.block_1(h)
        hs[f"block_{self.nres - 1}_atten"] = h
        h = self.mid.block_2(self.mid.attn_1(h))
        hs["mid_atten"] = h
        hs["out"] = self.conv_out(_swish(self.norm_out(h)))
        return hs


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: tuple[int, ...],
        nrb: int,
        attn_res: tuple[int, ...],
        resolution: int,
        z_channels: int,
        head_size: int,
    ) -> None:
        super().__init__()
        self.nres, self.nrb = len(ch_mult), nrb
        block_in = ch * ch_mult[-1]
        res = resolution // 2 ** (self.nres - 1)
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        levels: list[_Level] = []
        for i in reversed(range(self.nres)):
            lvl = _Level()
            block_out = ch * ch_mult[i]
            for _ in range(nrb + 1):
                lvl.block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
                if res in attn_res:
                    lvl.attn.append(MultiHeadAttnBlock(block_in, head_size))
            if i != 0:
                lvl.upsample = Upsample(block_in)
                res *= 2
            levels.insert(0, lvl)
        self.up = nn.ModuleList(levels)
        self.norm_out = _norm(block_in)
        self.conv_out = nn.Conv2d(block_in, 3, 3, 1, 1)

    def forward(self, z: torch.Tensor, hs: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h, hs["mid_atten"])
        h = self.mid.block_2(h)
        for i in reversed(range(self.nres)):
            lvl = self.up[i]
            for j in range(self.nrb + 1):
                h = lvl.block[j](h)
                if len(lvl.attn) > 0:
                    key = f"block_{i}_atten" if f"block_{i}_atten" in hs else f"block_{i}"
                    h = lvl.attn[j](h, hs[key])
            if i != 0:
                h = lvl.upsample(h)
        return self.conv_out(_swish(self.norm_out(h)))


class RestoreFormerPP(nn.Module):
    """RestoreFormer++ with the configuration of the released checkpoint."""

    size = 512

    def __init__(
        self,
        n_embed: int = 1024,
        embed_dim: int = 256,
        ch: int = 64,
        ch_mult: tuple[int, ...] = (1, 2, 2, 4, 4, 8),
        num_res_blocks: int = 2,
        z_channels: int = 256,
        head_size: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(ch, ch_mult, num_res_blocks, (16,), 512, z_channels, head_size)
        self.decoder = DecoderTransformer(
            ch, ch_mult, num_res_blocks, (16, 32), 512, z_channels, head_size
        )
        self.quantize = VectorQuantizer(n_embed, embed_dim)
        self.quant_conv = nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = self.encoder(x)
        quant = self.quantize(self.quant_conv(hs["out"]))
        return self.decoder(self.post_quant_conv(quant), hs)


def load_restoreformer_pp(model: RestoreFormerPP, ckpt: dict) -> None:
    """Load the official last.ckpt (Lightning checkpoint with a 'vqvae.' prefix)."""
    state = ckpt.get("state_dict", ckpt)
    own = model.state_dict()
    picked = {}
    for k, v in state.items():
        k = k.removeprefix("vqvae.")
        if k in own:
            picked[k] = v
    missing = [k for k in own if k not in picked]
    if missing:
        raise RuntimeError(
            f"RestoreFormer++ checkpoint is missing {len(missing)} keys: {missing[:5]}"
        )
    model.load_state_dict(picked, strict=True)
