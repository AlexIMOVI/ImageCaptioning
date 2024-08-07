import torch
import torch.nn as nn

class PosSlicer(nn.Module):
    def __init__(self):
        super(PosSlicer, self).__init__()

    def forward(self, input):
        features, gt_features = input
        if gt_features.numel() == 0:
            output = features
        else:
            P = gt_features.size(0)
            assert P <= features.size(0), "Must have P <= N"
            output = features[:P]
        return output

    # def backward(self, input, gradOutput):
    #     features, gt_features = input
    #     self.grad_gt_features = torch.zeros_like(gt_features)
    #     if gt_features.numel() == 0:
    #         self.gradInput = (gradOutput, self.grad_gt_features)
    #     else:
    #         P = gt_features.size(0)
    #         self.grad_features = torch.zeros_like(features)
    #         self.grad_features[:P] = gradOutput
    #         self.gradInput = (self.grad_features, self.grad_gt_features)
    #     return self.gradInput
