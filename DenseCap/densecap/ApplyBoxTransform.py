import torch
import torch.nn as nn
import math


class ApplyBoxTransform(nn.Module):

    def __init__(self):
        super(ApplyBoxTransform, self).__init__()

    def forward(self, input):
        """
        Args:
            boxes (Tensor): Tensor of shape (..., 4) giving coordinates of boxes in
                            (xc, yc, w, h) format.
            trans (Tensor): Tensor of shape (..., 4) giving box transformations in the form
                            (tx, ty, tw, th).
        Returns:
            Tensor: Transformed boxes of shape (..., 4) in (xc, yc, w, h) format.
        """

        boxes, trans = input
        minibatch=True
        if boxes.dim() == 2:
            minibatch = False
            boxes = boxes.unsqueeze(0)
        if trans.dim() == 2:
            minibatch = False
            trans = trans.unsqueeze(0)
        assert boxes.shape[-1] == 4, 'Last dim of boxes must be 4'
        assert trans.shape[-1] == 4, 'Last dim of trans must be 4'

        xa = boxes[:, :, 0]
        ya = boxes[:, :, 1]
        wa = boxes[:, :, 2]
        ha = boxes[:, :, 3]
        tx = trans[:, :, 0]
        ty = trans[:, :, 1]
        tw = trans[:, :, 2]
        th = trans[:, :, 3]

        ret = torch.zeros_like(boxes, device=trans.device)
        ret[:, :, 0] = (tx * wa) + xa
        ret[:, :, 1] = (ty * ha) + ya
        ret[:, :, 2] = wa * torch.exp(tw)
        ret[:, :, 3] = ha * torch.exp(th)

        if not minibatch:
            ret = ret.squeeze(0)

        return ret
