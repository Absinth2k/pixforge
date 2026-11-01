# FBCNN: Flexible Blind CNN for JPEG artifact removal.
#
# Origin: https://github.com/jiaxi-jiang/FBCNN (models/network_fbcnn.py), building blocks
# from cszn/KAIR and xinntao/BasicSR. Licensed under the Apache License 2.0.
#
# Modifications for Pixforge: only the configuration used by the released weights
# (strideconv down, convtranspose up, ReLU), no torchvision import, padding moved out of
# the model (tiles aligned to 8), explicit quality-factor input so all tiles of an image
# can share one estimate (avoids seams), type hints.
"""FBCNN network definition (inference only)."""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.res = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=False),
                                 nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res(x)


class QFAttention(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.res = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=False),
                                 nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return x + gamma[..., None, None] * self.res(x) + beta[..., None, None]


class FBCNN(nn.Module):
    """forward(x, qf) -> (restored, predicted_qf).

    x: (b, 3, h, w) RGB 0..1, h and w multiples of 8.
    qf: (b, 1) in 0..1 = 1 - jpeg_quality/100. Pass a negative value to use the model's
        own per-tile prediction instead.
    """

    alignment = 8

    def __init__(self, nc: tuple[int, int, int, int] = (64, 128, 256, 512), nb: int = 4) -> None:
        super().__init__()
        self.nb = nb
        self.m_head = nn.Conv2d(3, nc[0], 3, 1, 1)

        def down(i: int) -> nn.Sequential:
            return nn.Sequential(*[ResBlock(nc[i]) for _ in range(nb)],
                                 nn.Conv2d(nc[i], nc[i + 1], 2, 2, 0))

        self.m_down1, self.m_down2, self.m_down3 = down(0), down(1), down(2)
        self.m_body_encoder = nn.Sequential(*[ResBlock(nc[3]) for _ in range(nb)])
        self.m_body_decoder = nn.Sequential(*[ResBlock(nc[3]) for _ in range(nb)])

        def up(i: int) -> nn.ModuleList:
            return nn.ModuleList([nn.ConvTranspose2d(nc[i + 1], nc[i], 2, 2, 0),
                                  *[QFAttention(nc[i]) for _ in range(nb)]])

        self.m_up3, self.m_up2, self.m_up1 = up(2), up(1), up(0)
        self.m_tail = nn.Conv2d(nc[0], 3, 3, 1, 1)
        self.qf_pred = nn.Sequential(
            *[ResBlock(nc[3]) for _ in range(nb)], nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid(),
        )
        self.qf_embed = nn.Sequential(nn.Linear(1, 512), nn.ReLU(), nn.Linear(512, 512),
                                      nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        for i, c in ((3, nc[2]), (2, nc[1]), (1, nc[0])):
            setattr(self, f"to_gamma_{i}", nn.Sequential(nn.Linear(512, c), nn.Sigmoid()))
            setattr(self, f"to_beta_{i}", nn.Sequential(nn.Linear(512, c), nn.Tanh()))

    def forward(self, x: torch.Tensor, qf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body_encoder(x4)
        qf_pred = self.qf_pred(x)
        x = self.m_body_decoder(x)
        qf_used = torch.where(qf < 0, qf_pred, qf)
        emb = self.qf_embed(qf_used)
        for i, (ups, skip) in enumerate(((self.m_up3, x4), (self.m_up2, x3), (self.m_up1, x2))):
            level = 3 - i
            gamma = getattr(self, f"to_gamma_{level}")(emb)
            beta = getattr(self, f"to_beta_{level}")(emb)
            x = ups[0](x + skip)
            for blk in ups[1:]:
                x = blk(x, gamma, beta)
        return self.m_tail(x + x1), qf_pred
