import torch.nn as nn
import torch
import torch.nn.functional as F


class AffineGridGeneratorBHWD(nn.Module):
    """
    Generates a 2D flow field (sampling grid) for bilinear sampling given a batch of affine transformation matrices.
    """

    def __init__(self, height, width, device):
        super(AffineGridGeneratorBHWD, self).__init__()
        self.height = height
        self.width = width
        self.device = device

    def forward(self, affine_matrix):
        B = affine_matrix.size(0)
        grid = F.affine_grid(affine_matrix.to(self.device), [B, 1, self.height, self.width], align_corners=False)
        return grid


class BatchBilinearSamplerBHWD(nn.Module):
    """
    BatchBilinearSamplerBHWD efficiently performs bilinear sampling to pull out
    multiple patches from a single input image.
    """

    def __init__(self):
        super(BatchBilinearSamplerBHWD, self).__init__()
        self.inputImageView = torch.Tensor()

    def check(self, input):
        inputImages, grids = input[0], input[1]

        assert inputImages.dim() == 4
        assert grids.dim() == 4
        assert inputImages.size(0) == grids.size(0)  # batch
        assert grids.size(3) == 2  # coordinates

    def forward(self, input):
        inputImages, grids = input[0], input[1]

        assert grids.dim() == 4
        B = grids.size(0)

        assert inputImages.dim() == 3
        self.inputImageView = inputImages.expand(B, -1, -1, -1)

        self.check([self.inputImageView, grids])
        self.output = F.grid_sample(self.inputImageView, grids, align_corners=False)

        return self.output