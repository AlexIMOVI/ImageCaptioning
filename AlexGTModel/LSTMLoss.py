import torch
import torch.nn as nn

class TemporalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(TemporalCrossEntropyLoss, self).__init__()
        self.lsm = nn.LogSoftmax(dim=-1)
        self.crit = nn.CTCLoss()

    def forward(self, input, target):
        N, T, C = input.size(0), input.size(1), input.size(2)
        input_lengths = torch.ones(N) * T
        input = self.lsm(input.permute(1, 0, 2))
        target_lengths = target.min(dim=1)
        target = target.long()
        null_mask = torch.eq(target, 0)

        logprobs = self.lsm.forward(input)
        losses = torch.gather(logprobs, 2, target.view(N, T, 1)).mul(-1)
        losses[null_mask] = 0

        if self.batch_average:
            losses = losses.div(N)
        if self.time_average:
            losses = losses.div(T)

        return losses.sum()

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    def forward(self, input, target):
        N, T, C = input.size(0), input.size(1), input.size(2)
        target = target.long()
        # size = target.nonzero().numel() / 2
        input = input.view(-1, C)
        target = target.view(-1)
        loss = self.criterion(input, target)# / size
        return loss
