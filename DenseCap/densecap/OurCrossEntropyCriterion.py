import torch
import torch.nn as nn

class OurCrossEntropyCriterion(nn.Module):
    def __init__(self, weights=None):
        super(OurCrossEntropyCriterion, self).__init__()
        self.lsm = nn.LogSoftmax(dim=1)
        self.nll = nn.NLLLoss(weight=weights)

    def forward(self, input, target):
        # No need to squeeze in PyTorch
        self.lsm_output = self.lsm(input)
        self.output = self.nll(self.lsm_output, target)
        return self.output
