import torch
import torch.nn as nn

class InvertBoxTransform(nn.Module):
    def __init__(self):
        super(InvertBoxTransform, self).__init__()

    def forward(self, anchor_boxes, target_boxes):

        xa = anchor_boxes[:, 0]
        ya = anchor_boxes[:, 1]
        wa = anchor_boxes[:, 2]
        ha = anchor_boxes[:, 3]

        xt = target_boxes[:, 0]
        yt = target_boxes[:, 1]
        wt = target_boxes[:, 2]
        ht = target_boxes[:, 3]

        x = (xt - xa) / wa
        y = (yt - ya) / ha
        w = torch.log(wt/wa)
        h = torch.log(ht/ha)

        self.output = torch.stack([x, y, w, h], dim=-1)

        return self.output