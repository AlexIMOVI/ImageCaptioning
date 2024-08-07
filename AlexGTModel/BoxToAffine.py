import torch
import torch.nn as nn


class BoxToAffine(nn.Module):
    """
    Convert bounding box coordinates to affine parameter matrices
    for bilinear interpolation.
    """

    def __init__(self):
        super(BoxToAffine, self).__init__()
        self.H = None
        self.W = None

    def setSize(self, H, W):
        """
        Set the size of the input image.
        """
        self.H = H
        self.W = W

    def forward(self, input):
        """
        Forward pass.
        """
        assert input.dim() == 2, 'Expected 2D input'
        B = input.size(0)
        assert input.size(1) == 4, 'Expected input of shape B x 4'

        assert self.H and self.W, 'Need to call setSize before calling forward'

        xc = input[:, 0]
        yc = input[:, 1]
        w = input[:, 2]
        h = input[:, 3]

        self.output = torch.zeros(B, 2, 3, device=input.device)

        self.output[:, 0, 2] = (xc * 2 - 1 - self.W) / (self.W - 1)
        self.output[:, 1, 2] = (yc * 2 - 1 - self.H) / (self.H - 1)
        self.output[:, 0, 0] = w / self.W
        self.output[:, 1, 1] = h / self.H
        return self.output

