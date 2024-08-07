import torch
from torch import nn


class MakeBoxes(nn.Module):
    def __init__(self, x0, y0, sx, sy, anchors):
        super(MakeBoxes, self).__init__()
        self.x0 = x0
        self.y0 = y0
        self.sx = sx
        self.sy = sy
        self.anchors = anchors.clone()

        self.input_perm = torch.Tensor()
        self.boxes = torch.Tensor()
        self.raw_anchors = torch.Tensor()
        self.wa_expand = None
        self.ha_expand = None

        self.dtx = torch.Tensor()
        self.dty = torch.Tensor()
        self.dtw = torch.Tensor()
        self.dth = torch.Tensor()

    def forward(self, input):
        N, H, W = input.size(0), input.size(2), input.size(3)
        k = input.size(1) // 4
        self.boxes = torch.zeros(N, k, H, W, 4)

        x = self.boxes[:, :, :, :, 0]
        y = self.boxes[:, :, :, :, 1]
        w = self.boxes[:, :, :, :, 2]
        h = self.boxes[:, :, :, :, 3]

        input_view = input.view(N, k, 4, H, W)

        tx = input_view[:, :, 0]
        ty = input_view[:, :, 1]
        tw = input_view[:, :, 2]
        th = input_view[:, :, 3]

        wa, ha = self.anchors[0], self.anchors[1]
        self.wa_expand = wa.view(1, k, 1, 1).expand(N, k, H, W)
        self.ha_expand = ha.view(1, k, 1, 1).expand(N, k, H, W)

        xa = torch.arange(0, W).type_as(input).mul_(self.sx).add_(self.x0)
        ya = torch.arange(0, H).type_as(input).mul_(self.sy).add_(self.y0)
        xa_expand = xa.view(1, 1, 1, W).expand(N, k, H, W)
        ya_expand = ya.view(1, 1, H, 1).expand(N, k, H, W)

        self.raw_anchors.resize_as_(self.boxes)
        self.raw_anchors.select(4, 0).copy_(xa_expand)
        self.raw_anchors.select(4, 1).copy_(ya_expand)
        self.raw_anchors.select(4, 2).copy_(self.wa_expand)
        self.raw_anchors.select(4, 3).copy_(self.ha_expand)
        self.raw_anchors = self.raw_anchors.view(N, k * H * W, 4)

        x.mul_(self.wa_expand).add_(xa_expand)
        y.mul_(self.ha_expand).add_(ya_expand)

        w.exp_().mul_(self.wa_expand)
        h.exp_().mul_(self.ha_expand)

        self.output = self.boxes.view(N, k * H * W, 4)
        return self.output

    # def backward(self, input, gradOutput):
    #     N, H, W = input.size(0), input.size(2), input.size(3)
    #     k = input.size(1) // 4
    #
    #     self.gradInput.resize_as_(input).zero_()
    #
    #     dboxes = gradOutput.view(N, k, H, W, 4)
    #     dx = dboxes[:, :, :, :, 0]
    #     dy = dboxes[:, :, :, :, 1]
    #     dw = dboxes[:, :, :, :, 2]
    #     dh = dboxes[:, :, :, :, 3]
    #
    #     self.dtx.mul_(self.wa_expand, dx)
    #     self.dty.mul_(self.ha_expand, dy)
    #
    #     w = self.boxes[:, :, :, :, 2]
    #     h = self.boxes[:, :, :, :, 3]
    #     self.dtw.mul_(w, dw)
    #     self.dth.mul_(h, dh)
    #
    #     gradInput_view = self.gradInput.view(N, k, 4, H, W)
    #     gradInput_view[:, :, 0].copy_(self.dtx)
    #     gradInput_view[:, :, 1].copy_(self.dty)
    #     gradInput_view[:, :, 2].copy_(self.dtw)
    #     gradInput_view[:, :, 3].copy_(self.dth)
    #
    #     return self.gradInput
