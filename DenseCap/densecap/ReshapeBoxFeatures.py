import torch
import torch.nn as nn

class ReshapeBoxFeatures(nn.Module):
    def __init__(self, k):
        super(ReshapeBoxFeatures, self).__init__()
        self.k = k

    def forward(self, input):
        N,  H, W= input.size(0), input.size(2), input.size(3)
        D = input.size(1) // self.k
        k = self.k
        input_perm = input.view(N, k, D, H, W).permute(0, 1, 3, 4, 2)
        output = input_perm.reshape(N, k * H * W, D)
        return output

