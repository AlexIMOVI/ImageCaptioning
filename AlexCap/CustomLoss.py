import torch
import torch.nn as nn

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0, label_smoothing=0.1)

    def forward(self, input, target):
        N, T, C = input.size(0), input.size(1), input.size(2)
        target = target.long()
        input = input.view(-1, C)
        target = target.view(-1)
        loss = self.criterion(input, target)
        return loss
