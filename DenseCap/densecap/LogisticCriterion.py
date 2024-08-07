import torch
import torch.nn as nn
from densecap_utils import utils


class LogisticCriterion(nn.Module):
    def __init__(self):
        super(LogisticCriterion, self).__init__()
        self.offsets = None
        self.buffer = None
        self.log_den = None
        self.losses = None
        self.mask = None
        self.target_nonzero = None
        self.target_zero_mask = None

    def forward(self, input, target):
        self.offsets = torch.minimum(input, torch.zeros_like(input))
        self.buffer = torch.exp(self.offsets - input)
        self.log_den = torch.log(torch.exp(self.offsets) + self.buffer) - self.offsets
        self.target_nonzero = target.unsqueeze(1)

        self.target_zero_mask = target.eq(0)

        self.losses = self.log_den.clone()

        self.losses[self.target_zero_mask] = self.losses[self.target_zero_mask] + input[self.target_zero_mask]
        self.losses = self.losses / input.numel()
        self.output = torch.sum(self.losses)
        return self.output

