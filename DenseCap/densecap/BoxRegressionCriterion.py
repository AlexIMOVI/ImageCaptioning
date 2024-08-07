import torch
import torch.nn as nn
from InvertBoxTransform import InvertBoxTransform


class BoxRegressionCriterion(nn.Module):
    def __init__(self, device, w=1.0):
        super(BoxRegressionCriterion, self).__init__()
        self.w = w
        self.device = device
        self.invert_transform = InvertBoxTransform().to(device)
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, input, target_boxes):
        anchor_boxes, transforms = input
        self.target_transforms = self.invert_transform(anchor_boxes, target_boxes)

        # DIRTY DIRTY HACK: Ignore loss for boxes whose transforms are too big
        max_trans = torch.abs(self.target_transforms).max(dim=1).values.unsqueeze(1)
        mask = torch.gt(max_trans, 10).expand_as(self.target_transforms)
        mask_sum = mask.sum() / 4
        if mask_sum > 0:
            print(f'WARNING: Ignoring {mask_sum} boxes in BoxRegressionCriterion')
            transforms[mask] = 0
            self.target_transforms[mask] = 0

        loss = self.w * self.smooth_l1(transforms, self.target_transforms)
        return loss
