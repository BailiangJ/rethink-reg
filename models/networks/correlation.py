import torch
import torch.nn as nn


class WinCorrTorch3D(nn.Module):
    def __init__(self, radius: int = 3):
        super().__init__()
        self.win_size = 2 * radius + 1
        self.radius = radius
        self.padding = nn.ConstantPad3d(radius, 0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape
        b, c, d, h, w = x.shape  # depth, height, width
        y_padded = self.padding(y)
        offset = torch.meshgrid(
            [torch.arange(0, self.win_size) for _ in range(3)]
        )
        corr = torch.cat(
            [
                torch.sum(x * y_padded[:, :, dz:dz + d, dy:dy + h, dx:dx + w], dim=1, keepdim=True)
                for dz, dy, dx in zip(offset[0].flatten(), offset[1].flatten(), offset[2].flatten())
            ], dim=1
        )
        corr *= (c ** -0.5)
        return corr


class GlobalCorrTorch3D(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape
        b, c, d, h, w = x.shape
        x_flat = x.view(b, c, -1)
        y_flat = y.view(b, c, -1)
        corr = torch.einsum('bci, bcj -> bji', x_flat, y_flat)
        corr *= (c ** -0.5)
        return corr.view(b,-1,d,h,w)
