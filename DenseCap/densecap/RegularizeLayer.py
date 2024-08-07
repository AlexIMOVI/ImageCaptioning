import torch
import torch.nn as nn

class RegularizeLayer(nn.Module):
    def __init__(self, weight):
        super(RegularizeLayer, self).__init__()
        assert weight is not None, 'RegularizeLayer needs its weight passed in'
        self.w = weight

    def forward(self, input):
        loss = 0.5 * self.w * torch.norm(input, p=2)**2
        self.loss = loss
        self.output = input.clone()  # noop forward
        return self.output

    # def backward(self, input, gradOutput):
    #     gradInput = input.clone().mul(self.w)
    #     gradInput.add_(gradOutput)
    #     return gradInput
