import torch
from torch import nn


class MakeAnchors(nn.Module):
    def __init__(self, x0, y0, sx, sy, anchors):
        super(MakeAnchors, self).__init__()
        self.x0 = x0
        self.y0 = y0
        self.sx = sx
        self.sy = sy
        self.anchors = anchors.clone()

    def forward(self, input):
        N, H, W = input.size(0), input.size(2), input.size(3)
        k = self.anchors.size(1)
        x_centers = torch.arange(0, W).type_as(input)
        x_centers = x_centers * self.sx + self.x0
        y_centers = torch.arange(0, H).type_as(input)
        y_centers = y_centers * self.sy + self.y0

        self.output = torch.zeros(N, k, 4, H, W, dtype=input.dtype, device=input.device)

        self.output[:, :, 0] = x_centers.view(1, 1, 1, W).expand(N, k, H, W)
        self.output[:, :, 1] = y_centers.view(1, 1, H, 1).expand(N, k, H, W)
        self.output[:, :, 2] = self.anchors[0].view(1, k, 1, 1).expand(N, k, H, W)
        self.output[:, :, 3] = self.anchors[1].view(1, k, 1, 1).expand(N, k, H, W)

        self.output = self.output.reshape(N, 4*k, H, W)

        return self.output